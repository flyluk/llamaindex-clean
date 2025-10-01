FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y iproute2 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
COPY django_app/requirements.txt django_requirements.txt
RUN pip install -r requirements.txt
RUN pip install -r django_requirements.txt

COPY . .

# Pre-download docling models during build
RUN docling-tools models download

RUN python -c "import chromadb;client = chromadb.Client();collection = client.create_collection('all-my-documents');collection.add(documents=['This is document1'], ids=['doc1']);results = collection.query(query_texts=['This is a query document'],n_results=1)"

EXPOSE 8000

WORKDIR /app/django_app
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]