import heapq
from .graph_manager import SkillsGraph

ENGINE_NAME_UCS   = "UCS — Coste/Tiempo mínimo garantizado"
ENGINE_NAME_ASTAR = "A* — Optimización de recurso con heurística"

def run(
    graph: SkillsGraph,
    missing_skills: list[str],
    possessed_skills: list[str],
    criterion: str = "cost",   # 'cost' | 'time'
    **kwargs
) -> dict:
    if not missing_skills:
        return graph.build_result([], engine_name=ENGINE_NAME_ASTAR)

    target = frozenset(missing_skills)

    def evaluate_path(path_tuple):
        if not path_tuple:
            return frozenset(), 0.0
        
        expanded_cids = graph._expand_prerequisites(set(path_tuple), set(possessed_skills))
        
        covered_skills = set()
        for c in expanded_cids:
            covered_skills.update(graph.courses[c]["habilidades_aportadas"])
            
        if criterion == "cost":
            g_val = sum(float(graph.courses[c]["coste"]) for c in expanded_cids)
        else:
            g_val = sum(float(graph.courses[c]["duracion_semanas"]) for c in expanded_cids)
            
        return frozenset(covered_skills), g_val

    def step_cost_simple(cid: str) -> float:
        return float(graph.courses[cid]["coste"]) if criterion == "cost" else float(graph.courses[cid]["duracion_semanas"])

    best_value_per_skill = {}
    for skill in missing_skills:
        providers = graph.get_providers(skill)
        if providers:
            best_value_per_skill[skill] = min(step_cost_simple(cid) for cid in providers)
        else:
            best_value_per_skill[skill] = 0.0

    def heuristic(covered):
        missing = [sk for sk in target if sk not in covered]
        if not missing:
            return 0.0
        return max(best_value_per_skill.get(sk, 0.0) for sk in missing)

    candidate_ids = sorted([
        cid for cid, course in graph.courses.items()
        if any(sk in target for sk in course["habilidades_aportadas"])
    ])
    
    if not candidate_ids:
        return graph.build_result(
            [], warnings=["No hay cursos para las habilidades faltantes."],
            engine_name=ENGINE_NAME_ASTAR
        )

    tie = 0
    initial_covered = frozenset()
    h0 = heuristic(initial_covered)
    
    heap = [(h0, 0.0, 0, tie, initial_covered, ())]
    best_g = {initial_covered: 0.0}
    
    solution = None
    solution_cost = float("inf")

    while heap:
        f, g, num_courses, _, covered, path = heapq.heappop(heap)
        
        if f >= solution_cost:
            continue
            
        if target.issubset(covered):
            if g < solution_cost:
                solution = path
                solution_cost = g
            continue  

   
        last_cid = path[-1] if path else ""

        for cid in candidate_ids:
            if cid <= last_cid:
                continue
                
            new_path = path + (cid,)
            
            new_covered, new_g = evaluate_path(new_path)
            
            if new_g < best_g.get(new_covered, float("inf")):
                best_g[new_covered] = new_g
                new_h = heuristic(new_covered)
                new_f = new_g + new_h
                tie += 1
                
                heapq.heappush(heap, (new_f, new_g, num_courses + 1, tie, new_covered, new_path))

    if solution is None:
        solution = ()

    courses_set = set(solution)
    all_needed = graph._expand_prerequisites(courses_set, set(possessed_skills))
    ordered = graph._topological_order(all_needed)

    used_heuristic = any(v > 0 for v in best_value_per_skill.values())
    engine_label = ENGINE_NAME_ASTAR if used_heuristic else ENGINE_NAME_UCS
    label = "min_cost" if criterion == "cost" else "min_time"
    
    return graph.build_result(ordered, engine_name=engine_label, criterion=label)