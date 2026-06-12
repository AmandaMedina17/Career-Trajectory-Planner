"""
This is the piece that connects the user's constraints to the correct engine.

Decision logic:
─────────────────────────────────────────────────────────────────────
    In any case (no restrictions, no cost preference):
        → Engine 1: BFS (minimum number of courses)

    Does the user want to optimize only ONE metric?
        "cheapest" → Engine 2: A* with g(n) = cost
        "fastest" → Engine 2: A* with g(n) = time

    Did the user set HARD constraints (budget, weeks, hours/week)?
        YES → Engine 3: CSP (hardly constrained search)

    Did the user assign weights (α and β) to cost and time?
        YES → Engine 4: Metaheuristic (Simulated Annealing)

─────────────────────────────────────────────────────────────────────
"""

from .graph_manager import SkillsGraph
from . import strategy_hard_constraints, strategy_min_courses, strategy_multi_objective, strategy_single_objective


def select_and_run(
    graph: SkillsGraph,
    missing_skills: list[str],
    possessed_skills: list[str],
    constraints: dict
) -> dict:
    """
    Choose the appropriate motor according to the user's restrictions and run it.

    
    """
    max_cost          = constraints.get("max_cost")
    max_weeks         = constraints.get("max_weeks")
    max_hours_per_week= constraints.get("max_hours_per_week")
    only_free         = constraints.get("only_free", False)
    optimize          = constraints.get("optimize")      # 'cost' | 'time' | None
    weight_cost       = constraints.get("weight_cost")   # float | None
    weight_time       = constraints.get("weight_time")   # float | None

   
    if only_free:
        max_cost = 0.0
        constraints = dict(constraints)
        constraints["max_cost"] = 0.0
        print("  [ROUTER] Filtro: solo cursos gratuitos → Motor 3 (CSP)")


    # CASE 1: Metaheuristics
    if weight_cost is not None and weight_time is not None:
        print(f"  [ROUTER] Función compuesta (coste {weight_cost:.0%} / tiempo {weight_time:.0%})"
              f" → Motor 4: Recocido Simulado")
        return strategy_multi_objective.run(
            graph=graph,
            missing_skills=missing_skills,
            possessed_skills=possessed_skills,
            weight_cost=weight_cost,
            weight_time=weight_time
        )

    # CASE 2: CSP
    has_hard_constraints = (
        (max_cost is not None) or
        (max_weeks is not None) or
        (max_hours_per_week is not None)
    )
    if has_hard_constraints:
        print(f"  [ROUTER] Restricciones duras detectadas → Motor 3: CSP")
        return strategy_hard_constraints.run(
            graph=graph,
            missing_skills=missing_skills,
            possessed_skills=possessed_skills,
            max_cost=max_cost,
            max_weeks=max_weeks,
            max_hours_per_week=max_hours_per_week
        )

    # CASE 3: A* / UCS
    if optimize in ("cost", "time"):
        label = "coste mínimo" if optimize == "cost" else "tiempo mínimo"
        print(f"  [ROUTER] Optimización de {label} → Motor 2: A* / UCS")
        return strategy_single_objective.run(
            graph=graph,
            missing_skills=missing_skills,
            possessed_skills=possessed_skills,
            criterion=optimize
        )

    # CASE 4: BFS 
    print("  [ROUTER] Sin restricciones → Motor 1: BFS (mínimo número de cursos)")
    return strategy_min_courses.run(
        graph=graph,
        missing_skills=missing_skills,
        possessed_skills=possessed_skills
    )