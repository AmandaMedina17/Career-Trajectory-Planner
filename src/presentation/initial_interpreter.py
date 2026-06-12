import json
from mistralai.client import Mistral

SEP = "=" * 70
SEP_LIGHT = "-" * 70

def header():
    """Displays the program header"""
    print(f"\n{SEP}")
    print("  SISTEMA DE PLANIFICACION DE TRAYECTORIA PROFESIONAL")
    print("  Motor: Grafos DAG + Busqueda BFS + LLM (Mistral AI)")
    print(SEP)

def extract_skills_from_text(client: Mistral, text: str, text_type: str, catalog: list) -> list:
    """
    Use Mistral AI to extract skills from the catalog based on text.

    This is your original function. Its logic remains unchanged. It has only been adapted to receive the client as a parameter (instead of creating it within the function) to avoid creating multiple connections.

    Parameters:
    client: an already initialized Mistral instance
    text: user-generated text (skill goal or description)
    text_type: 'base' (skills already possessed) or 'goal' (professional goal)
    catalog: a list of strings containing the exact names from the catalog

    Returns:
    A list of strings containing the exact names of skills from the catalog.
    """
    catalog_str = ", ".join(catalog)

    if text_type == "base":
        prompt = f"""
        Eres un sistema experto en mapear habilidades. Tu tarea es interpretar la siguiente descripción del usuario sobre las habilidades que YA POSEE.

        Descripción del usuario: "{text}"

        Debes devolver ÚNICAMENTE un objeto JSON con la lista de habilidades que coincidan con el catálogo oficial que se proporciona.
        Catálogo oficial: {catalog_str}

        Reglas:
        - Usa EXACTAMENTE los nombres del catálogo.
        - Si el usuario menciona una habilidad de forma genérica (ej: "python"), debes mapearla a la habilidad más adecuada del catálogo (ej: "Programación en Python").
        - Si una habilidad no tiene una coincidencia clara, no la incluyas.
        - Si el usuario quiere decir que no tiene habilidades básicas, devuelve un json vacío.
        - Responde SOLO con JSON, sin texto adicional.
        Formato: {{"habilidades": ["nombre exacto 1", "nombre exacto 2", ...]}}
        """
    else:  # goal
        prompt = f"""
        Dada la siguiente meta profesional: "{text}"

        **REGLAS ESTRICTAS**:
        1. Solo puedes devolver habilidades que EXISTAN EXACTAMENTE en el catálogo oficial que se proporciona.
        2. NO puedes inventar habilidades. Si la meta profesional requiere una habilidad que no está en el catálogo, NO la incluyas.
        3. Excluye explícitamente habilidades blandas genéricas (comunicación, trabajo en equipo, empatía, pensamiento crítico, etc.) a menos que sean indispensables y estén en el catálogo.
        4. Si no encuentras ninguna habilidad del catálogo relevante para la meta, devuelve {{"habilidades": []}}.
        5. Responde SOLO con JSON.
        6. Céntrate exclusivamente en:
        - Habilidades técnicas concretas (programación, manejo de herramientas, metodologías)
        - Conocimientos científicos o técnicos (estadística, álgebra, biología, anatomía, etc.)
        - Dominios específicos de la profesión (diagnóstico veterinario, cirugía, análisis de datos, etc.)
        - Certificaciones o técnicas especializadas
        
        Catálogo oficial: {catalog_str}
        """
    try:
        completion = client.chat.complete(
            model="mistral-small-latest",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.1
        )
        response_content = completion.choices[0].message.content
        data = json.loads(response_content)
        return data.get("habilidades", [])
    except Exception as e:
        print(f"   ❌ Error en la extracción de habilidades: {e}")
        return []


def show_recognized_skills(type: str, skills: list):
    """
    Demonstrates the skills recognized by the LLM.
    
    type: 'base' o 'goal'
    """
    if type == "base":
        print("\n  [OK] Habilidades base reconocidas por el LLM:")
        if skills:
            for h in skills:
                print(f"       + {h}")
        else:
            print("       (ninguna habilidad base reconocida)")
    else:
        print("\n  [OK] Habilidades objetivo identificadas por el LLM:")
        if skills:
            for h in skills:
                print(f"       > {h}")
        else:
            print("       (no se identificaron habilidades objetivo)")


def show_gap(missing_skills: list, skills_already_posessed: list):
    """
    It shows the gap between what the user has and what they need
    """
    print(f"\n{SEP_LIGHT}")
    print("  HABILIDADES FALTANTES PARA ALCANZAR TU META")
    print(SEP_LIGHT)
    
    if skills_already_posessed:
        print(f"\n  [TIENES] Habilidades del objetivo que ya posees ({len(skills_already_posessed)}):")
        for h in skills_already_posessed:
            print(f"           [*] {h}")
    
    if missing_skills:
        print(f"\n  [FALTAN] Habilidades que necesitas adquirir ({len(missing_skills)}):")
        for h in missing_skills:
            print(f"           [ ] {h}")
    else:
        print("\n  [FELICITACIONES] Ya posees todas las habilidades necesarias para tu meta.")


def ask_for_restrictions() -> dict:
    """
    Collects user restrictions interactively.

    Returns a dict with:

    - max_total_hours: int or None

    - max_cost: float or None

    - available_hours_week: int or None

    - exclude_payment: bool
    """
    print(f"\n{SEP_LIGHT}")
    print("  CONFIGURACION DE RESTRICCIONES (pulsa Enter para omitir)")
    print(SEP_LIGHT)
    
    restrictions = {}
    
    # Available hours per week
    raw = input("\n  ¿Cuántas horas/semana puedes dedicar al estudio? (ej: 10): ").strip()
    if raw.isdigit():
        restrictions["available_hours_week"] = int(raw)
    else:
        restrictions["available_hours_week"] = None
    
    # Term in weeks
    raw = input("  ¿En cuántas semanas puedes completar el plan? (ej: 20): ").strip()
    if raw.isdigit():
        semanas = int(raw)
        horas_sem = restrictions.get("available_hours_week")
        if horas_sem:
            restrictions["max_total_hours"] = semanas * horas_sem
        else:
            restrictions["max_total_hours"] = None
        restrictions["plazo_semanas"] = semanas
    else:
        restrictions["max_total_hours"] = None
        restrictions["plazo_semanas"] = None
    
    # Maximum budget
    raw = input("  ¿Presupuesto máximo en usd? (ej: 200): ").strip()
    try:
        restrictions["max_cost"] = float(raw)
    except ValueError:
        restrictions["max_cost"] = None
    
    # Exclude paid courses
    raw = input("  ¿Solo cursos gratuitos? (s/n, default: n): ").strip().lower()
    restrictions["exclude_payment"] = (raw == "s")

    # Exclude in-person courses
    raw = input("  ¿Solo modalidad a distancia? (s/n, default: n): ").strip().lower()
    restrictions["online_mode"] = (raw == "s")
    
    return restrictions