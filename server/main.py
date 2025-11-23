from datetime import datetime
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from pathlib import Path
import json
import os

app = FastAPI(title="Recipe Saver API")

class RecipePage(BaseModel):
    url: str
    html: str

# Endpoint for Chrome extension to send recipe page
@app.post("/api/save-recipe")
async def save_recipe(recipe: RecipePage):
    print(f"Received recipe URL: {recipe.url}")

    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    with open(f"recipes/{now}.json", "w") as f:
        f.write(json.dumps({
            "url": recipe.url,
            "html": recipe.html,
            "date": now
        }))

    return {"status": "success", "url": recipe.url}
