import random
import math
from .graph_manager import SkillsGraph

ENGINE_NAME = "Optimización multi-objetivo"


def run(
    graph: SkillsGraph,
    missing_skills: list[str],
    possessed_skills: list[str],
    weight_cost: float = 0.5,     
    weight_time: float = 0.5,     
    seed: int = 42,               
    **kwargs
) -> dict:

    if not missing_skills:
        return graph.build_result([], engine_name=ENGINE_NAME)

    random.seed(seed)
    target = set(missing_skills)

    total_w = weight_cost + weight_time
    if total_w == 0:
        weight_cost = weight_time = 0.5
    else:
        weight_cost /= total_w
        weight_time /= total_w

    all_costs = [c["coste"] for c in graph.courses.values()]
    all_weeks = [c["duracion_semanas"] for c in graph.courses.values()]
    max_possible_cost = sum(sorted(all_costs, reverse=True)[:len(missing_skills) + 5]) or 1
    max_possible_weeks = sum(sorted(all_weeks, reverse=True)[:len(missing_skills) + 5]) or 1

   
    current = _greedy_solution(graph, target, weight_cost, weight_time, max_possible_cost, max_possible_weeks)

    T = 10.0           
    T_min = 0.0001     
    alpha = 0.99      
    max_iter = 5000   
    patience = 500    

    best = set(current)
    best_score = _objective(graph, best, target, weight_cost, weight_time,
                            max_possible_cost, max_possible_weeks, possessed_skills)
    current_score = best_score
    no_improve = 0

    iteration = 0
    while T > T_min and no_improve < patience:
        for _ in range(max_iter):
            neighbor = _get_neighbor(graph, current, target)
            if not neighbor:
                continue

            neighbor_score = _objective(graph, neighbor, target, weight_cost, weight_time,
                                        max_possible_cost, max_possible_weeks, possessed_skills)

            delta = neighbor_score - current_score

            if delta < 0:
                current = neighbor
                current_score = neighbor_score
                if current_score < best_score:
                    best = set(current)
                    best_score = current_score
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                prob = math.exp(-delta / T) if T > 0 else 0
                if random.random() < prob:
                    current = neighbor
                    current_score = neighbor_score
                no_improve += 1

            iteration += 1
        T *= alpha

    all_needed = graph._expand_prerequisites(best, set(possessed_skills))
    ordered = graph._topological_order(all_needed)

    criterion_label = f"cost={weight_cost:.0%} time={weight_time:.0%}"
    return graph.build_result(
        ordered,
        engine_name=ENGINE_NAME,
        criterion=criterion_label
    )


def _objective(
    graph: SkillsGraph,
    solution: set[str],
    target: set[str],
    weight_cost: float,
    weight_time: float,
    max_cost: float,
    max_weeks: float,
    possessed_skills: list[str]
) -> float:
    
    if not solution:
        return float("inf")

    expanded_cids = graph._expand_prerequisites(solution, set(possessed_skills))

    total_cost = sum(graph.courses[cid]["coste"] for cid in expanded_cids)
    total_weeks = sum(graph.courses[cid]["duracion_semanas"] for cid in expanded_cids)

    covered = set()
    for cid in expanded_cids:
        covered.update(graph.courses[cid]["habilidades_aportadas"])

    uncovered_fraction = len(target - covered) / max(len(target), 1)
    penalty = 1000.0 * uncovered_fraction

    cost_norm = total_cost / max_cost
    time_norm = total_weeks / max_weeks

    return weight_cost * cost_norm + weight_time * time_norm + penalty


def _get_neighbor(graph: SkillsGraph, current: set[str], target: set[str]) -> set[str]:
    ops = ["swap", "add", "remove"]
    random.shuffle(ops)

    for op in ops:
        result = None

        if op == "swap" and current:
            victim = random.choice(list(current))
            victim_skills = set(graph.courses[victim]["habilidades_aportadas"])

            substitutes = [
                cid for cid in graph.courses
                if cid != victim
                and cid not in current
                and victim_skills.issubset(graph.courses[cid]["habilidades_aportadas"])
            ]
            if substitutes:
                replacement = random.choice(substitutes)
                result = (current - {victim}) | {replacement}

        elif op == "add":
            covered = set()
            for cid in current:
                covered.update(graph.courses[cid]["habilidades_aportadas"])
            uncovered = target - covered

            if uncovered:
                skill_to_add = random.choice(list(uncovered))
                providers = graph.get_providers(skill_to_add)
                if providers:
                    new_course = random.choice(providers)
                    result = current | {new_course}

        elif op == "remove" and len(current) > 1:
            victim = random.choice(list(current))
            remaining = current - {victim}

            covered_without = set()
            for cid in remaining:
                covered_without.update(graph.courses[cid]["habilidades_aportadas"])

            victim_skills = set(graph.courses[victim]["habilidades_aportadas"])
            if victim_skills.issubset(covered_without):
                result = remaining

        if result and result != current:
            return result

    return set(current)


def _greedy_solution(
    graph: SkillsGraph, 
    target: set[str], 
    weight_cost: float, 
    weight_time: float, 
    max_cost: float, 
    max_weeks: float
) -> set[str]:
    """
    Construye una solución inicial que respeta los pesos del usuario
    en lugar de irse siempre por la opción más barata por defecto.
    """
    remaining = set(target)
    chosen = set()
    covered = set()

    sorted_skills = sorted(remaining, key=lambda sk: len(graph.get_providers(sk)))

    for skill in sorted_skills:
        if skill in covered:
            continue
        providers = graph.get_providers(skill)
        if not providers:
            continue
            
        def provider_score(cid):
            c_norm = graph.courses[cid]["coste"] / max_cost if max_cost else 0
            w_norm = graph.courses[cid]["duracion_semanas"] / max_weeks if max_weeks else 0
            return weight_cost * c_norm + weight_time * w_norm

        best = min(providers, key=provider_score)
        chosen.add(best)
        covered.update(graph.courses[best]["habilidades_aportadas"])

    return chosen