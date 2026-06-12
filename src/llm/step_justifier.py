import json
from mistralai.client import Mistral

def justify_steps(
    client: Mistral,
    goal: str,
    trajectory: list[dict]
) -> list[dict]:
    """
    For each course in the trajectory, the LLM generates a personalised justification
    of why that course is necessary to achieve the user's specific goal.
    
    This adds EXPLAINABILITY to the system: the user not only sees "what to learn",
    but also "why that specific course for their specific objective".
    
    For efficiency, a SINGLE call to the LLM is made with all courses,
    instead of one call per course (which would be costly and slow).
    
    Returns:
      List of dicts, one per course, with:
        - 'id': course id
        - 'justification': explanatory text (1-2 sentences)
    """
    lista_cursos = "\n".join([
        f'  - ID: {c["id"]} | Curso: "{c["nombre"]}" | Aporta: {", ".join(c["habilidades_aportadas"])}'
        for c in trajectory
    ])
    
    course_list = [c["id"] for c in trajectory]
    
    prompt = f"""
    Eres un asesor de carrera experto. Un estudiante quiere alcanzar esta meta profesional:

    META: "{goal}"

    Ha generado esta trayectoria de aprendizaje:
    {course_list}

    Para CADA curso de la lista, escribe una justificación concisa (1-2 frases) de por qué
    ese curso específico es necesario o útil para alcanzar esa meta concreta.
    La justificación debe ser personalizada para la meta, no una descripción genérica del curso.

    Responde ÚNICAMENTE con un JSON con este formato exacto:
    {{
    "justificaciones": [
        {{"id": "C001", "justificacion": "texto explicativo..."}},
        {{"id": "C002", "justificacion": "texto explicativo..."}},
        ...
    ]
    }}

    Los IDs que debes justificar son: {course_list}
    Responde SOLO con JSON.
    """
    
    try:
        completion = client.chat.complete(
            model="mistral-small-latest",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.3
        )
        response = completion.choices[0].message.content
        data = json.loads(response)
        return data.get("justificaciones", [])
    except Exception as e:
        print(f"   ❌ Error in LLM justification: {e}")
        return []