install:
	pip install -r requirements.txt

run:
	uvicorn src.app:app --reload --port 8001