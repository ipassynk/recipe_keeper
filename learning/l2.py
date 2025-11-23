from dotenv import load_dotenv
import os

from langchain_neo4j import Neo4jVector
from langchain_neo4j import GraphCypherQAChain, Neo4jGraph

load_dotenv('.env', override=True)
NEO4J_URI = os.getenv('NEO4J_URI')
NEO4J_USERNAME = os.getenv('NEO4J_USERNAME')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')
NEO4J_DATABASE = os.getenv('NEO4J_DATABASE')

kg = Neo4jGraph(
    url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD, database=NEO4J_DATABASE
)

# Global constants
VECTOR_INDEX_NAME = 'form_10k_chunks'
VECTOR_NODE_LABEL = 'Chunk'
VECTOR_SOURCE_PROPERTY = 'text'
VECTOR_EMBEDDING_PROPERTY = 'textEmbedding'

first_file_name = "./data.json"
first_file_as_object = json.load(open(first_file_name))

cyther = """
MATCH (n:Person {name: "Julia"})
RETURN n
"""

result = kg.query(cyther)
print(result)