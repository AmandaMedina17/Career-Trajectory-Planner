import os
import json
from dotenv import load_dotenv
from mistralai.client import Mistral

load_dotenv()
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
if not MISTRAL_API_KEY:
    raise ValueError("No se encontró MISTRAL_API_KEY en .env")

client = Mistral(api_key=MISTRAL_API_KEY)

# Cargar catálogo de habilidades
with open("skills.json", "r", encoding="utf-8") as f:
    SKILLS_CATALOG = json.load(f)

def extract_skills_from_text(text, text_type, catalog):
    """Usa Mistral AI para extraer habilidades del catálogo a partir de un texto."""
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
        6.**Céntrate exclusivamente** en:
        - Habilidades técnicas concretas (programación, manejo de herramientas, metodologías)
        - Conocimientos científicos o técnicos (estadística, álgebra, biología, anatomía, etc.)
        - Dominios específicos de la profesión (diagnóstico veterinario, cirugía, análisis de datos, etc.)
        - Certificaciones o técnicas especializadas
        
        Catálogo oficial: {catalog_str}
        """
    try:
        # CORRECCIÓN AQUÍ: Se pasa el prompt como un mensaje con 'role' y 'content'
        completion = client.chat.complete(
            model="mistral-small-latest",
            messages=[{"role": "user", "content": prompt}],  # <-- Esta es la línea corregida
            response_format={"type": "json_object"},
            temperature=0.1
        )
        response_content = completion.choices[0].message.content
        data = json.loads(response_content)
        return data.get("habilidades", [])
    except Exception as e:
        print(f"Error en la extracción de habilidades: {e}")
        return []

def main():
    print("=" * 70)
    print("🎯 PLANIFICADOR DE APRENDIZAJE (basado completamente en LLM)")
    print("=" * 70)
    
    # 1. Meta profesional
    goal = input("\n📌 Ingresa tu meta profesional: ").strip()
    if not goal:
        print("❌ Meta vacía.")
        return
    
    # 2. Descripción de habilidades base (texto libre)
    print("\n🧠 Describe las habilidades que YA POSEES (puedes escribir frases completas):")
    base_description = input("> ").strip()
    if not base_description:
        base_skills = []
    else:
        print("   🔍 Interpretando tus habilidades base con IA...")
        base_skills = extract_skills_from_text(base_description, "base", SKILLS_CATALOG)
        if base_skills:
            print("\n✅ Habilidades base reconocidas por Mistral:")
            for s in base_skills:
                print(f"   • {s}")
        else:
            print("   ⚠ No se reconoció ninguna habilidad base específica.")
    
    # 3. Habilidades necesarias para la meta
    print("\n🤖 Analizando qué habilidades necesitas para tu meta...")
    necessary_skills = extract_skills_from_text(goal, "goal", SKILLS_CATALOG)
    if not necessary_skills:
        print("❌ No se pudieron determinar las habilidades necesarias. Abortando.")
        return
    
    # 4. Calcular faltantes
    missing = [s for s in necessary_skills if s not in base_skills]
    
    # 5. Mostrar resultados
    print("\n" + "=" * 70)
    print("📊 RESULTADOS")
    print("=" * 70)
    print(f"\n🎯 Meta: {goal}")
    print("\n✅ Habilidades base (según tu descripción):")
    if base_skills:
        for s in base_skills:
            print(f"   • {s}")
    else:
        print("   (ninguna)")
    
    print("\n🆘 Habilidades que te faltan por adquirir:")
    if missing:
        for s in missing:
            print(f"   • {s}")
        print(f"\n💰 Total: {len(missing)} habilidades a adquirir.")
    else:
        print("   ¡Felicidades! Ya posees todas las habilidades necesarias.")
    print("=" * 70)

if __name__ == "__main__":
    main()