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

# Test docling conversion with a simple script
RUN python -c "from docling.document_converter import DocumentConverter; converter = DocumentConverter(); print('Docling initialized successfully')"
# Test chromadb functionality with a simple script
RUN python -c "import chromadb;client = chromadb.Client();collection = client.create_collection('all-my-documents');collection.add(documents=['This is document1'], ids=['doc1']);results = collection.query(query_texts=['This is a query document'],n_results=1)"

COPY b202412061.pdf .
RUN python doc_converter.py markdown b202412061.pdf b202412061.md
RUN rm b202412061.pdf b202412061.md

EXPOSE 8000

WORKDIR /app/django_app
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]