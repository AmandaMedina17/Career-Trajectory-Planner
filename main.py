#!/usr/bin/env python3
import sys
from src.planner import LearningPlanner

def main():
    print("=" * 70)
    print("PLANIFICADOR DE APRENDIZAJE")
    print("=" * 70)
    
    # Entrada de la meta
    goal = input("\n Ingresa tu meta profesional: ").strip()
    if not goal:
        print("❌ Meta vacía. Saliendo.")
        sys.exit(1)
    
    # Entrada de habilidades base (texto libre)
    print("\n Describe las habilidades que YA POSEES:")
    base_description = input("> ").strip()
    
    # Planificar
    print("\n Procesando con IA... (puede tomar unos segundos)")
    planner = LearningPlanner()
    result = planner.plan(goal, base_description)
    
    # Mostrar resultados
    print("\n" + "=" * 70)
    print("RESULTADOS")
    print("=" * 70)
    print(f"\n🎯 Meta: {result['goal']}")
    
    print("\nHabilidades base reconocidas:")
    if result['base_skills']:
        for s in result['base_skills']:
            print(f"   • {s}")
    else:
        print("   (ninguna)")
    
    print("\nHabilidades que te faltan por adquirir:")
    missing = result['missing_skills']
    if missing:
        for s in missing:
            print(f"   • {s}")
        print(f"\n Total: {len(missing)} habilidades a adquirir.")
    else:
        print("   ¡Felicidades! Ya posees todas las habilidades necesarias.")
    
    print("=" * 70)

if __name__ == "__main__":
    main()