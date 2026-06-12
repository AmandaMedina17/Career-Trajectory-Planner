import os
import json
from src.config import BASE_DIR

SKILLS_FILE = os.path.join(BASE_DIR, "data", "skills.json")
COURSES_FILE = os.path.join(BASE_DIR, "data", "courses.json")

def load_data() -> tuple[list, list]:
    """
    Loads the skill catalog and the course catalog from JSON files.
    
    Returns:
      - skills_catalog: list of strings (skill names)
      - courses_catalog: list of dicts (data for each course)
    """
    # Load skill catalog
    with open(SKILLS_FILE, "r", encoding="utf-8") as f:
        skills_catalog = json.load(f)
    
    # Load course catalog with prerequisites
    with open(COURSES_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
        courses_catalog = data["cursos"]
    
    return skills_catalog, courses_catalog