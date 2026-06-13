import heapq
from .graph_manager import SkillsGraph

def run(
    graph: SkillsGraph,
    missing_skills: list[str],
    possessed_skills: list[str],
    criterion: str = "cost",   # 'cost' | 'time' | 'courses'
    **kwargs
) -> dict:
    """
    A universal search engine based on f(n) = g(n) + h(n).
    It dynamically adjusts its weights and heuristics according to the chosen criteria.
    """
    
    # Initial Validations
    todas_posibles = set().union(*[set(c["habilidades_aportadas"]) for c in graph.courses.values()])
    if not set(missing_skills).issubset(todas_posibles):
        raise ValueError("Es imposible cubrir todas las habilidades con el catálogo actual.")

    if not missing_skills:
        return graph.build_result([], engine_name="Motor Unificado")

    target = frozenset(missing_skills)

    # Dynamic Evaluation Functions
    def evaluate_path(path_tuple):
        """
        Expand the prerequisites of the selected courses and calculate
        the actual cost g(n) according to the active criterion.
        """
        if not path_tuple:
            return frozenset(), 0.0
        
        # Expansion of prerequisites within the loop
        expanded_cids = graph._expand_prerequisites(set(path_tuple), set(possessed_skills))
        
        covered_skills = set()
        for c in expanded_cids:
            covered_skills.update(graph.courses[c]["habilidades_aportadas"])
            
        # Cost allocation g(n)
        if criterion == "cost":
            g_val = sum(float(graph.courses[c]["coste"]) for c in expanded_cids)
        elif criterion == "time":
            g_val = sum(float(graph.courses[c]["duracion_semanas"]) for c in expanded_cids)
        else: # criterion == "courses"
            g_val = float(len(expanded_cids))
            
        return frozenset(covered_skills), g_val

    def step_cost_simple(cid: str) -> float:
        """Determine the individual step cost for heuristic calculation."""
        if criterion == "cost":
            return float(graph.courses[cid]["coste"])
        elif criterion == "time":
            return float(graph.courses[cid]["duracion_semanas"])
        else:
            return 1.0

    # Preparation of the Heuristic h(n) 
    best_value_per_skill = {}
    for skill in missing_skills:
        providers = graph.get_providers(skill)
        if providers:
            best_value_per_skill[skill] = min(step_cost_simple(cid) for cid in providers)
        else:
            best_value_per_skill[skill] = 0.0

    def heuristic(covered):
        """
        Calculate h(n). If the criterion is the number of courses, it behaves
        like pure UCS (h=0) to guarantee optimality as if it were BFS.
        """
        if criterion == "courses":
            return 0.0 
            
        missing = [sk for sk in target if sk not in covered]
        if not missing:
            return 0.0
        return max(best_value_per_skill.get(sk, 0.0) for sk in missing)

    # Initialization of the Search Space
    candidate_ids = sorted([
        cid for cid, course in graph.courses.items()
        if any(sk in target for sk in course["habilidades_aportadas"])
    ])
    
    if not candidate_ids:
        return graph.build_result(
            [], warnings=["No hay cursos en el catálogo para las habilidades faltantes."],
            engine_name="Motor Unificado"
        )

    tie = 0
    initial_covered = frozenset()
    h0 = heuristic(initial_covered)
    
    # Heap structure: (f(n), g(n), tie, covered_skills, path_tuple)
    heap = [(h0, 0.0, tie, initial_covered, ())]
    best_g = {initial_covered: 0.0}
    
    solution = None
    solution_cost = float("inf")

    # Main Search Loop 
    while heap:
        f, g, _, covered, path = heapq.heappop(heap)
        
        # Pruning: if f(n) is already worse than our best found solution, ignore
        if f >= solution_cost:
            continue
            
        # Goal test
        if target.issubset(covered):
            if g < solution_cost:
                solution = path
                solution_cost = g
            continue  

        last_cid = path[-1] if path else ""

        for cid in candidate_ids:
            # Maintain lexicographical order to avoid redundant permutations

            if cid <= last_cid:
                continue
                
            new_path = path + (cid,)
            new_covered, new_g = evaluate_path(new_path)
            
            # STRICT RULE OF PROGRESS: Avoids infinite stagnation
            if not (new_covered - covered):
                continue
            
            if new_g < best_g.get(new_covered, float("inf")):
                best_g[new_covered] = new_g
                new_h = heuristic(new_covered)
                new_f = new_g + new_h
                tie += 1
                
                heapq.heappush(heap, (new_f, new_g, tie, new_covered, new_path))

    #  Reconstruction and Formatting
    if solution is None:
        # extreme case safety fallback
        solution = _greedy_fallback(graph, missing_skills)

    courses_set = set(solution)
    all_needed = graph._expand_prerequisites(courses_set, set(possessed_skills))
    ordered = graph._topological_order(all_needed)

    # Determine the correct label for the logs
    if criterion == "courses":
        engine_label = "BFS/UCS — Mínimo número de cursos"
    else:
        used_heuristic = any(v > 0 for v in best_value_per_skill.values())
        engine_label = "A* — Optimización" if used_heuristic else "UCS — Optimización Garantizada"
    
    return graph.build_result(ordered, engine_name=engine_label, criterion=criterion)


def _greedy_fallback(graph: SkillsGraph, missing_skills: list[str]) -> tuple:
    """Strict greedy fallback in case of disconnected graphs."""
    remaining = set(missing_skills)
    chosen = []

    while remaining:
        best_cid = max(
            graph.courses.keys(),
            key=lambda cid: len(set(graph.courses[cid]["habilidades_aportadas"]) & remaining),
            default=None
        )
        if best_cid is None or not (set(graph.courses[best_cid]["habilidades_aportadas"]) & remaining):
            break
        chosen.append(best_cid)
        remaining -= set(graph.courses[best_cid]["habilidades_aportadas"])

    return tuple(chosen)