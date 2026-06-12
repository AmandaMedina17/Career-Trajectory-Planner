import json
from mistralai.client import Mistral

def create_client(api_key: str) -> Mistral:
    """
    Creates and returns the Mistral client.
    It is called from main.py to centralize credential management
    """
    return Mistral(api_key=api_key)
