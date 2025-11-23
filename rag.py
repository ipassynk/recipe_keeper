import base64
import json
import os
import sys
import uuid
from datetime import datetime
from enum import Enum
import requests
import weaviate
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_neo4j import GraphCypherQAChain, Neo4jGraph
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langchain_weaviate import WeaviateVectorStore
from pydantic import BaseModel, Field
from weaviate.classes.config import Configure, DataType, Multi2VecField, Property


load_dotenv(".env", override=True)

weaviate_client = weaviate.connect_to_local()
print(weaviate_client.is_ready())

class RecipeStep(str, Enum):
    MAIN = "main"
    SERVING = "serving"


class Ingredient(BaseModel):
    quantity: float | None = Field(
        default=None, description="The quantity of the ingredient"
    )
    unit: str | None = Field(
        default=None, description="The unit of measurement of the ingredient"
    )
    ingredient: str = Field(description="The ingredient name")
    step: RecipeStep = Field(
        description="The step of the recipe where the ingredient is used (either 'main' or 'serving')"
    )


class Recipe(BaseModel):
    name: str = Field(description="The name of the recipe")
    description: str = Field(description="The description of the recipe")
    ingredients: list[Ingredient] = Field(
        description="The ingredients of the recipe with quantity and unit of measurement"
    )
    instructions: list[str] = Field(description="The instructions of the recipe")
    image: str = Field(description="The image of the recipe")
    recipeId: str = Field(description="The id of the recipe")
    url: str = Field(description="The url of the recipe")
    userName: str = Field(description="The name of the user who created the recipe")
    createdAt: str = Field(description="The date and time the recipe was created")
    summary: str = Field(description="The summary of the recipe")


def get_text_from_html(html: str) -> str:
    """
    Extract all image URLs abd pure text from html
    Args:
        html: html string
        name: name of the recipe
    Returns:
        text: pure text from html
        images: list of image URLs
    """
    soup = BeautifulSoup(html, "html.parser")
    images = []
    for img in soup.find_all("img"):
        img_url = img.get("src")
        if img_url:
            images.append(img_url)
        img.decompose()
    text = soup.get_text()
    return text, images


def extract_recipe(file_name: str, userName: str) -> tuple[Recipe, str]:
    with open(file_name, "r") as f:
        data = json.load(f)

    html = data["html"]
    url = data["url"]
    text, images = get_text_from_html(html)

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    llm_structured = llm.with_structured_output(Recipe)

    system_prompt = """"
    You are an expert in extracting recipe information from text.

    I will provide you with:
    1. The text content of a recipe page
    2. A list of image URLs found on the page
    3. The url of the recipe page
    4. The id of the recipe

    You will need to extract the following information:
        - Recipe Name
        - Recipe Description
        - Recipe Ingredients in a list
        - Recipe Instructions in a list
        - Recipe Image url (choose the main recipe image from the provided list of image URLs)
        - Recipe Summary (a summary of the recipe in max 100 words)

    Intruction about recipe name:
    Extract the recipe name from the text. Recipe name should be lowercase.

    Intruction about images:
    From the list of images, select the one that best represents the finished recipe dish. 
    If there are no images, leave it blank.
    Ignore decorative images, icons, logos, or images that are clearly not the main recipe photo.
    If there are multiple resolutions of the same image, use the lowest resolution image.

    Intruction about ingredients:
    Use the Ingredient model to format the ingredients list.
    Extract ingredient name from the text and quantity and unit of measurement from the text. Ingradient name should be lowercase.
    Include the quantity and unit of measurement for each ingredient.
    Include the step of the recipe where the ingredient is used - "main" or "serving".
    If you can't find the quantity and unit of measurement, leave it blank.
    Example:
    {
        "quantity": 1,
        "unit": "cup",
        "ingredient": "flour",
        "step": "main"
    }
    {
        "quantity": 2,
        "unit": "tablespoons",
        "ingredient": "olive oil",
        "step": "main"
    }
    {
        "quantity": 0.5,
        "unit": "teaspoon",
        "ingredient": "black pepper",
        "step": "main"
    }
    {
        "quantity": 500,
        "unit": "g",
        "ingredient": "beef brisket",
        "step": "main"
    }
    {
        "quantity": 750,
        "unit": "ml",
        "ingredient": "beef stock",
        "step": "main"
    }
    {
        "quantity": 2,
        "unit": "large",
        "ingredient": "onions",
        "step": "main"
    }
    {
        "quantity": 1.5,
        "unit": "kg",
        "ingredient": "potatoes",
        "step": "main"
    }
    {
        "quantity": 0.25,
        "unit": "cup",
        "ingredient": "butter",
        "step": "main"
    }
    {
        "quantity": 1,
        "unit": "pound",
        "ingredient": "ground beef",
        "step": "main"
    }
    {
        "quantity": 1,
        "unit": "pinch",
        "ingredient": "cayenne pepper",
        "step": "main"
    }
    {
        "quantity": 1,
        "unit": "",
        "ingredient": "corn tortillas",
        "step": "serving"
    }
    {
        "quantity": 1,
        "unit": "",
        "ingredient": "lime wedges",
        "step": "serving"
    }

    Example Output:
    {
        "name": "Recipe Name",
        "description": "Recipe Description",
        "ingredients": [
            {
                "quantity": 1,
                "unit": "cup",
                "ingredient": "flour",
                "step": "main"
            },
            {
                "quantity": 2,
                "unit": "tablespoons",
                "ingredient": "olive oil",
                "step": "main"
            }
        ],
        "instructions": ["Instruction 1", "Instruction 2", ..., "Instruction N"],
        "image": "Recipe Image url"
    }
    """
    recipeId = uuid.uuid4().hex
    result = llm_structured.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(
                content=f"Recipe Text:\n{text}\n\nAvailable Images:\n"
                + "\n".join(f"- {img}" for img in images)
                + f"\n\nRecipe URL:\n{url}"
                + f"\n\nRecipe ID:\n{recipeId}"
                + f"\n\nUser Name:\n{userName}"
            ),
        ]
    )
    print("=" * 100)
    print(json.dumps(result.model_dump(), indent=2, ensure_ascii=False))
    # Set recipeId and other fields that need to be set
    result.recipeId = recipeId
    result.userName = userName
    result.createdAt = datetime.now().isoformat()
    return result, text


def save_recipe_image(image_url: str, recipe_id: str) -> str:
    response = requests.get(image_url)
    img_b64 = base64.b64encode(response.content).decode("utf-8")
    os.makedirs("recipe_images", exist_ok=True)
    with open(f"recipe_images/{recipe_id}.jpg", "wb") as f:
        f.write(response.content)
    return img_b64


def save_recipe_as_graph(recipe: Recipe):
    NEO4J_URI = os.getenv("NEO4J_URI")
    NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
    NEO4J_DATABASE = os.getenv("NEO4J_DATABASE")

    kg = Neo4jGraph(
        url=NEO4J_URI,
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD,
        database=NEO4J_DATABASE,
    )

    # Save Recipe
    kg.query(
        """
    MERGE (r:Recipe {recipeId: $recipeId})
    SET r.name = $name, r.description = $description, r.instructions = $instructions, r.image = $image, r.userName = $userName, r.createdAt = $createdAt, r.url = $url
    """,
        {
            "recipeId": recipe.recipeId,
            "name": recipe.name,
            "description": recipe.description,
            "instructions": recipe.instructions,
            "image": recipe.image,
            "userName": recipe.userName,
            "createdAt": recipe.createdAt,
            "url": recipe.url,
        },
    )

    # Save Ingredients
    for ingredient in recipe.ingredients:
        kg.query(
            """
        MERGE (i:Ingredient {name: $name})
        SET i.name = $name
        """,
            {"name": ingredient.ingredient.lower()},
        )

    # Save Relationship between Recipe and Ingredients
    for ingredient in recipe.ingredients:
        kg.query(
            """
        MATCH (r:Recipe {recipeId: $recipeId})
        MATCH (i:Ingredient {name: $ingredientName})
        MERGE (r)-[rel:HAS_INGREDIENT]->(i)
        SET rel.quantity = $quantity, rel.unit = $unit
        """,
            {
                "recipeId": recipe.recipeId,
                "ingredientName": ingredient.ingredient.lower(),
                "quantity": ingredient.quantity,
                "unit": ingredient.unit,
            },
        )


def save_recipe_to_weaviate(recipe: Recipe, img_b64: str):
    weaviate_client.collections.get("Recipe").data.insert(
        {
            "name": recipe.name,
            "text": recipe.summary,
            "image": img_b64,
            "recipeId": recipe.recipeId,
        }
    )

def create_recipe_collection_in_weaviate():
    """
    Create recipe collection in Weaviate that is multi2vec-clip compatible for image and text search.
    """
    if not weaviate_client.collections.exists("Recipe"):
        weaviate_client.collections.create(
            "Recipe",
            properties=[
                Property(name="recipeId", data_type=DataType.TEXT),  
                Property(name="name", data_type=DataType.TEXT),      
                Property(name="text", data_type=DataType.TEXT),      
                Property(name="image", data_type=DataType.BLOB),     
            ],
            vector_config=[
                Configure.Vectors.multi2vec_clip(
                    name="recipe_vector",
                    image_fields=[Multi2VecField(name="image", weight=0.5)],
                    text_fields=[
                        Multi2VecField(name="name", weight=0.2),
                        Multi2VecField(name="text", weight=0.3)
                    ]
                )
            ],
        )



def main():
    #weaviate_client.collections.delete("Recipe")
    create_recipe_collection_in_weaviate()
    # TODO - get recipe from user name
    for file_name in os.listdir("recipes"):
        print(f"Processing {file_name}")
        print("=" * 100)
        recipe, text = extract_recipe(f"recipes/{file_name}", "Julia Passynkova")
        if recipe.image:
            img_b64 = save_recipe_image(recipe.image, recipe.recipeId)
      
        else:
            img_b64 = None
        
        save_recipe_as_graph(recipe)
        save_recipe_to_weaviate(recipe, img_b64)

    weaviate_client.close()


if __name__ == "__main__":
    main()
