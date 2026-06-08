from typing import List, Dict, Any
from .skill_catalog import SkillCatalog
from .llm_client import MistralSkillExtractor

class LearningPlanner:
    """Orquesta la extracción de habilidades y calcula las faltantes."""
    
    def __init__(self):
        self.catalog = SkillCatalog()
        self.extractor = MistralSkillExtractor(self.catalog.get_all())
    
    def plan(self, goal: str, base_description: str) -> Dict[str, Any]:
        """
        Retorna un diccionario con:
        - goal: la meta original
        - base_skills: lista de habilidades reconocidas como base
        - necessary_skills: lista de habilidades necesarias para la meta
        - missing_skills: habilidades que faltan
        """
        # Extraer habilidades base
        base_skills = []
        if base_description.strip():
            base_skills = self.extractor.extract_skills(base_description, "base")
        
        # Extraer habilidades necesarias
        necessary_skills = self.extractor.extract_skills(goal, "goal")
        
        # Calcular faltantes
        missing_skills = [s for s in necessary_skills if s not in base_skills]
        
        return {
            "goal": goal,
            "base_skills": base_skills,
            "necessary_skills": necessary_skills,
            "missing_skills": missing_skills
        }