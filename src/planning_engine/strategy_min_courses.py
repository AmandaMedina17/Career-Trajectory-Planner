import heapq
from .graph_manager import SkillsGraph

ENGINE_NAME = "BFS/UCS — Mínimo número de cursos"


def run(
    graph: SkillsGraph,
    missing_skills: list[str],
    possessed_skills: list[str],
    **kwargs
) -> dict:
    
    todas_posibles = set().union(*[set(c["habilidades_aportadas"]) for c in graph.courses.values()])
    if not set(missing_skills).issubset(todas_posibles):
        raise ValueError("Es imposible cubrir todas las habilidades con el catálogo actual.")

    if not missing_skills:
        return graph.build_result([], engine_name=ENGINE_NAME)

    target = frozenset(missing_skills)

   
    def evaluate_path_size(path_tuple):
        if not path_tuple:
            return frozenset(), 0
        
        expanded_cids = graph._expand_prerequisites(set(path_tuple), set(possessed_skills))
        
        covered_skills = set()
        for c in expanded_cids:
            covered_skills.update(graph.courses[c]["habilidades_aportadas"])
            
        return frozenset(covered_skills), len(expanded_cids)

    candidate_ids = sorted([
        cid for cid, course in graph.courses.items()
        if any(sk in target for sk in course["habilidades_aportadas"])
    ])

    if not candidate_ids:
        return graph.build_result(
            [], warnings=["No hay cursos en el catálogo para las habilidades faltantes."],
            engine_name=ENGINE_NAME
        )

    tie = 0
    initial_covered = frozenset()
    
    # heap: (num_cursos_reales, tie, cubiertas, path)
    heap = [(0, tie, initial_covered, ())]
    best_g = {initial_covered: 0}
    
    solution = None
    solution_cost = float("inf")

    while heap:
        g, _, covered, path = heapq.heappop(heap)
        
        # Poda 
        if g >= solution_cost:
            continue
            
        if target.issubset(covered):
            if g < solution_cost:
                solution = path
                solution_cost = g
            continue

        last_cid = path[-1] if path else ""

        for cid in candidate_ids:
            # Evitar permutaciones: explorar en orden id
            if cid <= last_cid:
                continue
                
            new_path = path + (cid,)
            
            new_covered, new_g = evaluate_path_size(new_path)
            
            if new_g < best_g.get(new_covered, float("inf")):
                best_g[new_covered] = new_g
                tie += 1
                heapq.heappush(heap, (new_g, tie, new_covered, new_path))

    if solution is None:
        solution = _greedy_fallback(graph, missing_skills)

    courses_set = set(solution)
    all_needed = graph._expand_prerequisites(courses_set, set(possessed_skills))
    ordered = graph._topological_order(all_needed)

    return graph.build_result(ordered, engine_name=ENGINE_NAME, criterion="min_courses")


def _greedy_fallback(graph: SkillsGraph, missing_skills: list[str]) -> tuple:
    """Fallback greedy: Choose the course that covers the most missing skills."""
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