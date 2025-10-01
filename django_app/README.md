# Django LlamaIndex App

Minimal Django conversion of the Streamlit LlamaIndex application.

## Setup

```bash
cd django_app
pip install -r requirements.txt
python manage.py migrate
python manage.py runserver
```

## Features

- **Search**: Document search with LlamaIndex
- **Upload**: File upload and processing
- **Models**: Ollama model management (pull/delete)
- **Admin**: Database administration

## URLs

- `/` - Main search page
- `/models/` - Model management
- `/admin_panel/` - Admin panel
- `/upload/` - File upload (POST)
- `/search/` - Search API (POST)
- `/pull_model/` - Pull model API (POST)
- `/delete_model/` - Delete model API (POST)

## Notes

- Uses Bootstrap for UI
- AJAX for dynamic operations
- Reuses existing DocumentIndexer class
- Minimal conversion focusing on core functionality