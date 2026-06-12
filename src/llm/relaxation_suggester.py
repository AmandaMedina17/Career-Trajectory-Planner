import json
from mistralai.client import Mistral
from src.planning_engine.load_data import load_data

def suggest_relaxation(
    client: Mistral,
    goal: str,
    constraints: dict,
    warnings: list[str]
) -> str:
    """
    When the engine detects that the trajectory does not meet the user's
    constraints (e.g., it takes too long or costs too much), the LLM suggests
    which constraints to relax and how.
    
    Returns: string with the LLM's advice.
    """
    skills_catalog, courses_catalog = load_data()
    warnings_str = "\n".join(warnings)
    constraints_str = json.dumps(constraints, ensure_ascii=False, indent=2)
    
    prompt = f"""
        Un estudiante quiere alcanzar esta meta profesional: "{goal}"

        Ha establecido estas restricciones:
        {constraints_str}

        El sistema de planificación ha detectado estos problemas:
        {warnings_str}

        Como asesor experto, sugiere de forma concisa (máximo 4-5 frases) qué restricciones
        podría relajar el usuario y cómo hacerlo de forma razonable para poder generar
        un plan factible. Sé específico y práctico segun los cursos: {courses_catalog}, saca un promedio para mas o menos 15 cursos.
        No menciones el id de los cursos.

        Responde solo con el texto de consejo, sin JSON. No uses markdown ni formato especial.
    """
    
    try:
        completion = client.chat.complete(
            model="mistral-small-latest",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"   ❌ Error in LLM suggestion: {e}")
        return "Could not generate an automatic suggestion."