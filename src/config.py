import os
from dotenv import load_dotenv

load_dotenv()

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
if not MISTRAL_API_KEY:
    raise ValueError("No se encontró MISTRAL_API_KEY en .env")

# Directorio raíz del proyecto
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))

LLM_MODEL = "mistral-small-latest"
LLM_TEMPERATURE = 0.1