.PHONY: server format rag chat-agent chat

server:
	uv run uvicorn server.main:app --reload --port 8001

rag:
	uv run rag.py 

chat-agent:
	uv run chat_agent.py

chat:
	uv run chainlit run chat.py -w --port 8002

format:
	uv run black .