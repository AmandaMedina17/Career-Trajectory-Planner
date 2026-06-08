import json
from typing import List
from mistralai.client import Mistral
from .config import MISTRAL_API_KEY, LLM_MODEL, LLM_TEMPERATURE

class MistralSkillExtractor:
    """Cliente para extraer habilidades del catálogo usando Mistral AI."""
    
    def __init__(self, catalog: List[str]):
        self.client = Mistral(api_key=MISTRAL_API_KEY)
        self.catalog = catalog
        self.catalog_str = ", ".join(catalog)
    
    def extract_skills(self, text: str, text_type: str) -> List[str]:
        """
        text_type: 'base' (habilidades ya poseídas) o 'goal' (habilidades necesarias para la meta)
        """
        prompt = self._build_prompt(text, text_type)
        try:
            completion = self.client.chat.complete(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=LLM_TEMPERATURE
            )
            response = json.loads(completion.choices[0].message.content)
            skills = response.get("habilidades", [])
            # Filtrar solo habilidades que existen en el catálogo
            return [s for s in skills if s in self.catalog]
        except Exception as e:
            print(f"Error en la extracción de habilidades: {e}")
            return []
    
    def _build_prompt(self, text: str, text_type: str) -> str:
        if text_type == "base":
            return f"""
            Eres un sistema experto en mapear habilidades. Tu tarea es interpretar la siguiente descripción del usuario sobre las habilidades que YA POSEE.

            Descripción del usuario: "{text}"

            Debes devolver ÚNICAMENTE un objeto JSON con la lista de habilidades que coincidan con el catálogo oficial que se proporciona.
            Catálogo oficial: {self.catalog_str}

            Reglas:
            - Usa EXACTAMENTE los nombres del catálogo.
            - Si el usuario menciona una habilidad de forma genérica (ej: "python"), debes mapearla a la habilidad más adecuada del catálogo (ej: "Programación en Python").
            - Si una habilidad no tiene una coincidencia clara, no la incluyas.
            - Si el usuario quiere decir que no tiene habilidades básicas, devuelve un json vacío.
            - Responde SOLO con JSON, sin texto adicional.
            Formato: {{"habilidades": ["nombre exacto 1", "nombre exacto 2", ...]}}
            """
        else:  # goal
            return f"""
            Dada la siguiente meta profesional: "{text}"

            **REGLAS ESTRICTAS**:
            1. Solo puedes devolver habilidades que EXISTAN EXACTAMENTE en el catálogo oficial que se proporciona.
            2. NO puedes inventar habilidades. Si la meta profesional requiere una habilidad que no está en el catálogo, NO la incluyas.
            3. Excluye explícitamente habilidades blandas genéricas (comunicación, trabajo en equipo, empatía, pensamiento crítico, etc.) a menos que sean indispensables y estén en el catálogo.
            4. Si no encuentras ninguna habilidad del catálogo relevante para la meta, devuelve {{"habilidades": []}}.
            5. Responde SOLO con JSON.
            6.**Céntrate exclusivamente** en:
            - Habilidades técnicas concretas (programación, manejo de herramientas, metodologías)
            - Conocimientos científicos o técnicos (estadística, álgebra, biología, anatomía, etc.)
            - Dominios específicos de la profesión (diagnóstico veterinario, cirugía, análisis de datos, etc.)
            - Certificaciones o técnicas especializadas
            
            Catálogo oficial: {self.catalog_str}
            """