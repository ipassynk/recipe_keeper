import chainlit as cl
import base64
from chat_agent import search_recipes

@cl.on_message
async def main(message: cl.Message):
    #check for image attachments
    img_b64 = None
    if message.elements:
        for element in message.elements:
            if "image" in element.mime:
                with open(element.path, "rb") as f:
                    img_b64 = base64.b64encode(f.read()).decode("utf-8")
                break
    
    if img_b64:
        print(f"Image processed, length: {len(img_b64)}")
    else:
        print("No image processed")

    msg = cl.Message(content="Searching...")
    await msg.send()

    response_content = await search_recipes(message.content, img_b64=img_b64)

    msg.content = response_content
    await msg.update()