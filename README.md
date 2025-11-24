# Recipe Keeper

Hope you like this app and bon appétit!

## Prerequisites

To install the application, you need:

1. Python3 and uv package manager.
2. Docker installed to run Neo4j and Weaviate containers.
3. Node.js (optional): Required if you want to use the extension to populate recipes. Otherwise, you can use the test recipes in the recipes folder.
4. Your OPENAI_API_KEY. Get it from http://openai.com

## System Architecture

The application implements a Retrieval-Augmented Generation (RAG) pipeline using FastAPI for orchestration.

* **Neo4j**: Serves as the graph database to model structured relationships between recipes, ingredients. This enables complex traversal and filtering that flat data structures cannot efficiently handle. We can add users to the graph to track their preferences and history.
* **Weaviate**: Used as the vector database for its native multi-modal support (via `multi2vec-clip`). This allows for hybrid search capabilities, enabling the system to understand and retrieve recipes based on both text and image inputs.
* **LangGraph**: Used for orchestration chat flows.
* **OpenAI**: Used as the LLM for inference during chat and text embeddings for Weaviate vectorization.
* **Chainlit**: Provides the interactive chat interface.
* **FastAPI**: Used for recipe ingestion.
* **Chrome**:  Used to inject recipes from the browser to the database.
* **CLIP**: Used to generate embeddings for images.




![System Architecture](docs/system_architecture.png)

Here is the chat flow:
![alt text](docs/recipe_keeper.png)

## Getting Started

1. Clone the project
```
git clone https://github.com/your-username/recipe-keeper.git
cd recipe-keeper
```

2. Set up your environment and update OPENAI_API_KEY key

```
cp env.example .env
```

3. Install python packages
    ```
    uv install
    ```
    
4. Start Neo4j and Weaviate containers:

    ```bash
    docker compose up -d
    ```
![alt text](docs/dockers.png)

5. Start the Chainlit UI app:

    ```bash
    make chat
    ```
    Open http://localhost:8002 in your browser to see the chatting app.

6a. (Either 6a or b6) Initialize the FastAPI backend if you want to use chrome extension to inject your recipes:
    ```bash
    make server
    ```
 You need to install the browser extension to send recipes from your browser to the database:

    ```bash
    cd extension
    npm install
    npm run build
    ```
    See the README.md in the extension folder how to install it.

    After you install the extension, you can inject your recipes into the system. Check that recipes are added to recipes folder.
    
6b. (Either 6a or b6) You can use the existing test recipes saved in the recipes folder:

    ```bash
     cp test-recipes/* recipes/
    ```

7. Run RAG to populate Neo4j and Weaviate with recipes:
    ```bash
    make rag
    ```

8. Check if the graph is created in Neo4j. Use the password defined in the .env file.
    ```bash
    http://localhost:7474
    ```

    Run the following Cypher query to check if the graph is created:
    ```bash
    MATCH (n)
    RETURN n
    ```

    You should see something like this:

    ![alt text](docs/graph.png)

9. Check that the recipes are added to Weaviate:
    TODO - need to add a script to check if the recipes are added to Weaviate

10. Check that images are added to local folder:
    ```bash
    ls images
    ```
    For now the system does not use them duing chatting but it will be added in the future.

11. You are now ready to chat with the Recipe Keeper.
    Here are example messages:

    ```
    Recipes that use these ingredients: mushrooms
    Show me recipes using chicken but NOT butter
    Find recipes with sugar
    Show me a recipe with creamy chicken and mushrooms
    Give me quick dinner ideas under 20 minutes
    Show me recipes that look similar to this dish
    Find recipes that look like this image but also mention spicy ingredients
    What can I use instead of ricotta?
    List all ingredients for Recipe_123
    What recipes did user Julia create?
    Find spicy chicken dishes with less than 10g carbs.
    ```

    Here is an example of chatting with the Recipe Keeper:
    ![alt text](docs/image_query.png)

## TODO:
    1. Add more recipes to test folder
    2. Implement evals to check if RAG and chat are working
    3. Allow injecting new recipes via chat
    4. Add authentication
    5. Add Airflow to schedule RAG
    6. Handle all errors
    7. Add telemetry for logging
    8. Add guardrails to prevent abuse
    9. Tested on macOS. Windows and Linux support needs verification.
    10. Cache results in redis
    11. Make a routing to different LLMs based on the query 