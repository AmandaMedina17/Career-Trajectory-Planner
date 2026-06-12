from .graph_manager import SkillsGraph

ENGINE_NAME = "CSP — Búsqueda con restricciones duras"

def run(
    graph: SkillsGraph,
    missing_skills: list[str],
    possessed_skills: list[str],
    max_cost: float = None,
    max_weeks: int = None,
    max_hours_per_week: int = None,
    **kwargs
) -> dict:
    """
    Performs DFS search with CSP pruning to find a valid path
    that respects the user's hard limits.

    """

    if not missing_skills:
        return graph.build_result([], engine_name=ENGINE_NAME)

    target = set(missing_skills)

    max_cost = float("inf") if max_cost is None else float(max_cost)
    max_weeks = float("inf") if max_weeks is None else float(max_weeks)
    max_hpw = float("inf") if max_hours_per_week is None else float(max_hours_per_week)

   
    candidate_ids = [
        cid for cid, c in graph.courses.items()
        if any(sk in target for sk in c["habilidades_aportadas"])
        and c["horas_semana"] <= max_hpw     
    ]

    
    candidate_ids.sort(key=lambda cid: (
        -len(set(graph.courses[cid]["habilidades_aportadas"]) & target),  
         graph.courses[cid]["coste"],                                     
         graph.courses[cid]["duracion_semanas"]  
    ))

    if not candidate_ids:
        return graph.build_result(
            [],
            warnings=["Ningún curso del catálogo cumple las restricciones de horas/semana."],
            engine_name=ENGINE_NAME
        )

   
    best_solution: tuple | None = None

    for depth_limit in range(1, len(candidate_ids) + 1):
        result = _dfs_csp(
            graph=graph,
            candidates=candidate_ids,
            target=target,
            max_cost=max_cost,
            max_weeks=max_weeks,
            depth_limit=depth_limit,
            cost_so_far=0.0,
            weeks_so_far=0.0,
            covered=set(),
            taken=()
        )
        if result is not None:
            best_solution = result
            break

    warnings = []
    if best_solution is None:
        mensaje_error = (
            f"❌ Imposible generar un plan válido. Las restricciones establecidas "
            f"(Presupuesto: {max_cost}€, Tiempo: {max_weeks} semanas) son demasiado estrictas "
            f"para cubrir todas las habilidades que te faltan."
        )
        return graph.build_result(
            [], 
            warnings=[mensaje_error], 
            engine_name=ENGINE_NAME, 
            criterion="csp_hard_failed"
        )

    courses_set = set(best_solution)
    all_needed = graph._expand_prerequisites(courses_set, set(possessed_skills))


    valid_needed = set()
    running_cost = 0.0
    running_weeks = 0.0
    for cid in all_needed:
        c = graph.courses[cid]
        
        if (running_cost + c["coste"] > max_cost or
            running_weeks + c["duracion_semanas"] > max_weeks or
            c["horas_semana"] > max_hpw):
            
            mensaje_error = (
                f"❌ Imposible generar el plan. Aunque los cursos principales entran en tu límite, "
                f"sus prerrequisitos ocultos (como '{c['nombre']}') hacen que superes "
                f"las restricciones de tiempo o dinero."
            )
            return graph.build_result(
                [], 
                warnings=[mensaje_error], 
                engine_name=ENGINE_NAME, 
                criterion="csp_hard_failed"
            )
            
        valid_needed.add(cid)
        running_cost += c["coste"]
        running_weeks += c["duracion_semanas"]

    ordered = graph._topological_order(valid_needed)
    return graph.build_result(ordered, warnings=warnings, engine_name=ENGINE_NAME, criterion="csp_hard")


def _dfs_csp(
    graph: SkillsGraph,
    candidates: list[str],
    target: set[str],
    max_cost: float,
    max_weeks: float,
    depth_limit: int,
    cost_so_far: float,
    weeks_so_far: float,
    covered: set[str],
    taken: tuple
) -> tuple | None:
    """
    DFS recursivo limitado en profundidad con poda por restricciones CSP.
    """
    if target.issubset(covered):
        return taken

    if depth_limit == 0:
        return None

    last_cid = taken[-1] if taken else ""

    for cid in candidates:
        if cid <= last_cid:
            continue

        course = graph.courses[cid]

        if cost_so_far + course["coste"] > max_cost:
            continue   

        if weeks_so_far + course["duracion_semanas"] > max_weeks:
            continue   

        nuevas = set(course["habilidades_aportadas"]) - covered
        if not nuevas:
            continue  

        new_covered = covered | nuevas
        new_taken = taken + (cid,)

        result = _dfs_csp(
            graph=graph,
            candidates=candidates,
            target=target,
            max_cost=max_cost,
            max_weeks=max_weeks,
            depth_limit=depth_limit - 1,
            cost_so_far=cost_so_far + course["coste"],
            weeks_so_far=weeks_so_far + course["duracion_semanas"],
            covered=new_covered,
            taken=new_taken
        )

        if result is not None:
            return result 

    return None 

def _best_partial(
    graph: SkillsGraph,
    candidates: list[str],
    target: set[str],
    max_cost: float,
    max_weeks: float
) -> tuple:
    
    remaining = set(target)
    chosen = []
    cost_acc = 0.0
    weeks_acc = 0.0

    # Ordenar por cobertura descendente
    sorted_candidates = sorted(
        candidates,
        key=lambda cid: -len(set(graph.courses[cid]["habilidades_aportadas"]) & remaining)
    )

    for cid in sorted_candidates:
        c = graph.courses[cid]
        if (cost_acc + c["coste"] <= max_cost and
                weeks_acc + c["duracion_semanas"] <= max_weeks):
            nuevas = set(c["habilidades_aportadas"]) & remaining
            if nuevas:
                chosen.append(cid)
                cost_acc += c["coste"]
                weeks_acc += c["duracion_semanas"]
                remaining -= nuevas
                if not remaining:
                    break

    return tuple(chosen)