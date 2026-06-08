import json
from typing import List
from .config import SKILLS_FILE

class SkillCatalog:
    """Carga y proporciona acceso al catálogo de habilidades."""
    
    def __init__(self):
        self._skills: List[str] = []
        self.load()
    
    def load(self):
        with open(SKILLS_FILE, "r", encoding="utf-8") as f:
            self._skills = json.load(f)
    
    def get_all(self) -> List[str]:
        return self._skills.copy()
    
    def exists(self, skill: str) -> bool:
        return skill in self._skills
    
    def __len__(self):
        return len(self._skills)