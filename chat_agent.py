import base64
import os
import re
from enum import Enum
from typing import TypedDict, Annotated, Literal
import weaviate
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_neo4j import Neo4jGraph
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_weaviate import WeaviateVectorStore
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from redisvl.utils.vectorize import OpenAITextVectorizer
from redisvl.extensions.llmcache import SemanticCache
import redis

load_dotenv(".env", override=True)

redis_client = redis.Redis(host="localhost", port=6379, db=0)
redis_client.ping()

langcache_embed = OpenAITextVectorizer(
    model="text-embedding-3-small"
)
recipe_cache = SemanticCache(
    name="recipe-cache",
    vectorizer=langcache_embed,
    redis_url="redis://localhost:6379",
    distance_threshold=0.3,
    ttl=3600
)

weaviate_client = weaviate.connect_to_local()

class CypherQuery(BaseModel):
    cypher: str = Field(description="The Cypher query to execute")

def _weaviate_image_search_impl(img_b64: str):
    """
    Search Weaviate using image vector similarity.

    Args:
        img_b64: base64 encoded image string
    """
    results = weaviate_client.collections.get("Recipe").query.near_image(
        near_image=img_b64, limit=2
    )
    print(f"Found {len(results.objects)} results")
    return [o.properties["recipeId"] for o in results.objects]

def weaviate_image_search(img_b64: str):
    """
    Search Weaviate using image vector similarity.

    Args:
        img_b64: base64 encoded image string
    """
    return _weaviate_image_search_impl(img_b64)


def _weaviate_text_search_impl(query_text: str):
    """
    Search Weaviate using text semantic search.

    Args:
        query_text: The text query to search for
    """
    embeddings = OpenAIEmbeddings()
    vectorstore = WeaviateVectorStore(
        client=weaviate_client,
        embedding=embeddings,
        index_name="Recipe",
        text_key="text",
    )

    results = vectorstore.similarity_search(query_text, k=2)
    return results


async def _get_recipe_ids_from_text_search(query_text: str):
    """
    Get recipe IDs from Weaviate text search for use in hybrid search.

    Args:
        query_text: The text query to search for
    """
    response = await recipe_cache.acheck(prompt=query_text, distance_threshold=0.3)
    if response:
        return response
    
    collection = weaviate_client.collections.get("Recipe")
    results = collection.query.near_text(
        query=query_text, limit=2, return_metadata=["distance"]
    )
    recipe_ids = [o.properties["recipeId"] for o in results.objects]
    await recipe_cache.acache(prompt=query_text, response=recipe_ids)

    print(f"Found {len(results.objects)} results")
    return recipe_ids


def weaviate_text_search(query_text: str):
    """
    Search Weaviate using text semantic search.

    Args:
        query_text: The text query to search for
        limit: Number of results to return
    """
    return _weaviate_text_search_impl(query_text)


def _neo4j_search_impl(query_text: str, recipe_ids: list[str] | None = None):
    """
    Search Neo4j using text semantic search.

    Args:
        query_text: The text query to search for
        recipe_ids: The list of recipe IDs to search for
    """
    NEO4J_URI = os.getenv("NEO4J_URI")
    NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
    NEO4J_DATABASE = os.getenv("NEO4J_DATABASE")

    neo4j_client = Neo4jGraph(
        url=NEO4J_URI,
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD,
        database=NEO4J_DATABASE,
    )

    schema = neo4j_client.get_schema
    recipe_ids_str = str(recipe_ids) if recipe_ids else "None"
    print("recipe_ids_str: ", recipe_ids_str)
    print("query_text: ", query_text)

    search_terms = query_text.lower()
    
    system_prompt = f"""
    You are a Cypher Query Generator for a Neo4j database.

    You MUST follow these rules:
    1. Only output Cypher. No explanations.
    2. Use ONLY the entities and relationships from the schema.
    3. If a query cannot be answered using the schema, return: "NO_CYPHER_POSSIBLE".
    4. Use parameterized values only for ingredient names. Do NOT use parameters for search terms (use string literals).
    5. Do NOT invent properties or nodes that do not exist.
    6. When `Current Recipe IDs` is NOT "None":
       - You MUST start your query with: `MATCH (r:Recipe) WHERE r.recipeId IN [...]` using the provided IDs.
       - Do NOT filter `r.name` or `r.description` using the query text (because the IDs already come from a text search).
       - DO filter other properties (ingredients, instructions, userName) if the query specifies them.
       - If the user query is generic (e.g. "show me", "similar to this") and does not specify other filters, just return the recipes in the IDs list.


    Examples:
    - If Current Recipe IDs = ['id1', 'id2'] and user asks about "tacos":
      MATCH (r:Recipe) WHERE r.recipeId IN ['id1', 'id2'] RETURN r

    - If Current Recipe IDs = ['id1', 'id2'] and user asks "with garlic":
      MATCH (r:Recipe) WHERE r.recipeId IN ['id1', 'id2'] AND EXISTS {{ (r)-[:HAS_INGREDIENT]->(i) WHERE i.name CONTAINS 'garlic' }} RETURN r

    - If Current Recipe IDs = None and user asks about "tacos":
      MATCH (r:Recipe) WHERE r.name CONTAINS 'tacos' OR r.description CONTAINS 'tacos' RETURN r

    Neo4j Schema:
    {schema}

    Current Recipe IDs:
    {recipe_ids_str}

    User question:
    "{query_text}"

    Output:
    Only Cypher.

    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    llm_structured = llm.with_structured_output(CypherQuery)

    result = llm_structured.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Generate Cypher for this question: {query_text}"),
        ]
    )

    cypher_query = result.cypher
    print("cypher_query: ", cypher_query)

    if "NO_CYPHER_POSSIBLE" in cypher_query:
        if recipe_ids:
            # Fallback: just get the recipes
            cypher_query = f"MATCH (r:Recipe) WHERE r.recipeId IN {str(recipe_ids)} RETURN r"
            print("Fallback cypher_query: ", cypher_query)
        else:
            return []

    results = neo4j_client.query(cypher_query)
    return results


def neo4j_search(query_text: str, recipe_ids: list[str] | None = None):
    """
    Search Neo4j using text semantic search.

    Args:
        query_text: The text query to search for
        recipe_ids: The list of recipe IDs to search for
    """
    return _neo4j_search_impl(query_text, recipe_ids)


def hybrid_search_image(img_b64: str):
    """
    Search both Weaviate and Neo4j for image-based search.
    """
    w_results = _weaviate_image_search_impl(img_b64)
    # TODO - move
    recipe_ids = [o.properties["recipeId"] for o in w_results.objects]
    if recipe_ids:
        results = _neo4j_search_impl("Find recipes matching these IDs", recipe_ids)
        return results
    else:
        return "No recipe IDs found"


def hybrid_search_text(query_text: str):
    """
    Search both Weaviate and Neo4j using text based search.
    """
    recipe_ids = _get_recipe_ids_from_text_search(query_text)
    if recipe_ids:
        results = _neo4j_search_impl(query_text, recipe_ids)
    else:
        return "No recipe IDs found"


class QueryRoute(str, Enum):
    """Possible routing destinations for queries"""

    NEO4J_SEARCH = "neo4j_search"
    HYBRID_SEARCH = "hybrid_search_text"
    HYBRID_SEARCH_IMAGE = "hybrid_search_image"


class RoutingDecision(BaseModel):
    """LLM decision on how to route a query"""

    route: QueryRoute = Field(
        description="The routing destination: 'hybrid_search_image' for image-based search, 'hybrid_search_text' for semantic text search, 'neo4j_search' for structured graph queries"
    )
    reasoning: str = Field(description="Brief explanation of why this route was chosen")


class GraphState(TypedDict):
    """State for the recipe search graph"""

    messages: Annotated[list[BaseMessage], add_messages]
    original_question: str
    route: str | None
    results: any
    img_b64: str | None


def route_query_node(state: GraphState) -> GraphState:
    """
    Route the query to the appropriate search system.
    This node has access to state['original_question'] directly!
    """
    original_question = state["original_question"]
    img_b64 = state["img_b64"]
    print(f"Routing query: {original_question}")

    if img_b64:
        return {
            "route": QueryRoute.HYBRID_SEARCH_IMAGE.value,
            "reasoning": "Image-based search is available",
        }

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    llm_structured = llm.with_structured_output(RoutingDecision)

    system_prompt = """You are a query routing expert for a recipe search system with two databases:

1. **neo4j_search** (Graph Database):
   - Use for: exact ingredient matching with specific constraints, quantity filters, exclusions, user ownership
   - Use for: queries that need structured filtering like "recipes with ingredient X but NOT ingredient Y"
   - Examples: "recipes with chicken but no butter", "recipes with >300g chicken", "recipes by user Julia", "recipes that use exactly these ingredients: chicken, salt, garlic"

2. **hybrid_search_text** (Both):
   - Use for: ANY query that searches by recipe description, name, or text content
   - Use for: semantic similarity, fuzzy matching, "similar to" queries, recommendations, finding recipes by keywords
   - Use for: queries with words like "mention", "about", "containing", "with", "recipes that", "find recipes"
   - Examples: "creamy pasta recipes", "recipes that mention tacos", "recipes similar to this", "show me dinner ideas", "find recipes about chicken"

CRITICAL ROUTING RULES:
- "recipes that mention X" → ALWAYS use hybrid_search_text (semantic text search)
- "find recipes about X" → ALWAYS use hybrid_search_text
- "recipes with X" (when X is a general term) → use hybrid_search_text
- "recipes with X but not Y" (exact exclusion) → use neo4j_search
- "recipes with >X grams" (quantity filter) → use neo4j_search

Route the query to the most appropriate system(s).
"""

    result = llm_structured.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Route this query: {original_question}"),
        ]
    )

    return {
        "route": result.route.value,
    }


def neo4j_search_node(state: GraphState) -> GraphState:
    """
    Perform Neo4j search. Has direct access to state['original_question']!
    """
    original_question = state["original_question"]
    print(f"Performing Neo4j search for: {original_question}")

    results = _neo4j_search_impl(original_question, None)

    return {
        "results": results,
    }


async def hybrid_search_text_node(state: GraphState) -> GraphState:
    """
    Perform hybrid text search. Has direct access to state['original_question']!
    """
    original_question = state["original_question"]
    print(f"Performing hybrid text search for: {original_question}")

    recipe_ids = await _get_recipe_ids_from_text_search(original_question)
    if recipe_ids:
        results = _neo4j_search_impl(original_question, recipe_ids)
        return {
            "results": results,
        }
    else:
        return {}


def hybrid_search_image_node(state: GraphState) -> GraphState:
    """
    Perform hybrid image search. Has direct access to state['img_b64'] and state['original_question']!
    """
    img_b64 = state.get("img_b64")
    original_question = state["original_question"]

    if not img_b64:
        return {
            "results": "Error: No image provided for image search",
        }

    print(f"Performing hybrid image search")
    recipe_ids = _weaviate_image_search_impl(img_b64)
    if recipe_ids:
        results = _neo4j_search_impl(original_question, recipe_ids)
        return {
            "results": results,
        }
    else:
        return {}


def format_response_node(state: GraphState) -> GraphState:
    """
    Format the final response based on search results.
    """
    results = state.get("results")
    original_question = state["original_question"]

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    response_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant that provides recipe search results. Format the search results in a clear and helpful way.",
            ),
            (
                "human",
                "User question: {question}\n\nSearch results: {results}\n\nProvide a helpful response based on these results.",
            ),
        ]
    )

    formatted_response = llm.invoke(
        response_prompt.format_messages(
            question=original_question, results=str(results)
        )
    )

    return {
        **state,
        "messages": [AIMessage(content=formatted_response.content)],
    }


def should_route(
    state: GraphState,
) -> Literal["neo4j_search", "hybrid_search_text", "hybrid_search_image"]:
    """
    Conditional routing function based on route decision.
    """
    route = state.get("route")
    if route == QueryRoute.NEO4J_SEARCH.value:
        return "neo4j_search"
    elif route == QueryRoute.HYBRID_SEARCH.value:
        return "hybrid_search_text"
    elif route == QueryRoute.HYBRID_SEARCH_IMAGE.value:
        return "hybrid_search_image"
    else:
        return "hybrid_search_text"


# Build the LangGraph
workflow = StateGraph(GraphState)

# Add nodes
workflow.add_node("route_query", route_query_node)
workflow.add_node("neo4j_search", neo4j_search_node)
workflow.add_node("hybrid_search_text", hybrid_search_text_node)
workflow.add_node("hybrid_search_image", hybrid_search_image_node)
workflow.add_node("format_response", format_response_node)

# Set entry point
workflow.set_entry_point("route_query")

# Add conditional edges from route_query
workflow.add_conditional_edges(
    "route_query",
    should_route,
    {
        "neo4j_search": "neo4j_search",
        "hybrid_search_text": "hybrid_search_text",
        "hybrid_search_image": "hybrid_search_image",
    },
)

# All search nodes go to format_response
workflow.add_edge("neo4j_search", "format_response")
workflow.add_edge("hybrid_search_text", "format_response")
workflow.add_edge("hybrid_search_image", "format_response")

# format_response is the end
workflow.add_edge("format_response", END)

# Compile the graph
app = workflow.compile()
# graph_image = app.get_graph(xray=True).draw_mermaid_png()
# with open("recipe_keeper.png", "wb") as f:
#     f.write(graph_image)

async def search_recipes(question: str, img_b64: str | None = None) -> str:
    print(f"Searching for: {question}")
    if img_b64:
        print(f"with image (length: {len(img_b64)})")
    print("=", 60)
    initial_state = {
        "messages": [HumanMessage(content=question)],
        "original_question": question,
        "route": None,
        "results": None,
        "img_b64": img_b64,
    }

    result = await app.ainvoke(initial_state)
    return result["messages"][-1].content


# Example usage
if __name__ == "__main__":
    import asyncio
    async def main():
        #response = await search_recipes("Find recipes that mention tacos?")
        #response = await search_recipes("Recipes that use these ingredients: mushroom")
        # response = await search_recipes("Show me recipes using chicken but NOT butter")
        response = await search_recipes("Find recipes with >10 grams of sugar")
        # response = await search_recipes("Show me a recipe with creamy chicken and mushrooms")
        # response = await search_recipes("Give me quick dinner ideas under 20 minutes")
        # response = await search_recipes("Show me recipes that look similar to this dish")
        # response = await search_recipes("Find recipes that look like this image but also mention spicy ingredients")

        print("\n=== Final Result ===")
        print(response)
        weaviate_client.close()

    asyncio.run(main())
