import json
from mistralai.client import Mistral

def compare_trajectories(
    client: Mistral,
    goal: str,
    primary_trajectory: dict,
    alternative_trajectory: dict
) -> dict:
    """
    The LLM compares two trajectories generated with different criteria and
    recommends which one is better for the user's profile, explaining why.
    
    Returns dict with:
      - 'recommended': 'primary' or 'alternative'
      - 'reason': text explaining the recommendation
      - 'comparison_table': list of strings with comparison points
    """
    def summary(t: dict, name: str) -> str:
        courses_str = ", ".join([c["nombre"] for c in t["path"]])
        return (
            f"--- {name} ---\n"
            f"  Courses: {courses_str}\n"
            f"  Total duration: {t['total_weeks']} weeks\n"
            f"  Total hours: {t['total_hours']}h\n"
            f"  Total cost: {t['total_cost']}€\n"
            f"  Selection criterion: {t.get('criterion', 'balanced')}"
        )
    
    prompt = f"""
    Eres un asesor de carrera. Compara estas dos trayectorias de aprendizaje para la meta:

    META PROFESIONAL: "{goal}"

    {summary(primary_trajectory, "TRAYECTORIA PRINCIPAL (criterio equilibrado)")}

    {summary(alternative_trajectory, "TRAYECTORIA ALTERNATIVA (criterio: " + alternative_trajectory.get("criterion", "distinto") + ")")}

    Analiza ambas y recomienda la más adecuada. Responde SOLO con JSON:
    {{
    "recomendada": "principal" o "alternativa",
    "razon": "explicación de por qué es mejor (2-3 frases)",
    "tabla_comparativa": [
        "Punto 1: la principal tiene X, la alternativa tiene Y",
        "Punto 2: ..."
    ]
    }}
"""
    
    try:
        completion = client.chat.complete(
            model="mistral-small-latest",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.2
        )
        response = completion.choices[0].message.content
        return json.loads(response)
    except Exception as e:
        print(f"   ❌ Error in LLM comparison: {e}")
        return {}