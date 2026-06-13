"""
We simulated N_AGENTS, an additional number of students who are also trying to complete career paths in the same course market.


Each agent has:
- Current skills (random)
- Career goal (random from a list)
- Budget (random variable)
- Available hours/week (random variable)
- Behavioral profile (conservative, normal, aggressive)

And reacts according to these rules:
    R1: IF course.cost > agent.remaining_budget
    THEN find_free_alternative(course)

    R2: IF market.demand(skill) > SATURATION_THRESHOLD
    THEN prioritize_skill_alternative()

    R3: IF agent.weeks_without_progress > STAGNATION_THRESHOLD
    THEN reduce_goal_to_subset()

    R4: IF agent.available_hours < course.hours_week
    THEN find_less_intensive_course()

    R5: IF agent.number_of_dropouts > MAX_DROPPOUTS
    THEN pause_career_path(weeks=4)

USE OF RESULTS
---------------------
The agent simulation generates MARKET STATISTICS:

- Which skills are most in demand (everyone is looking for them)
- Which courses have the longest virtual "waiting list"
- What percentage of agents achieve their goal
- Which skills are most competitive

These statistics are passed on to the LLM to contextualize their recommendation:
"Skill X is in high demand in the virtual market (70% of agents are looking for it),
consider differentiating yourself with Y, which has less competition."
"""

import random
import math
from collections import defaultdict


# AGENT PROFILE TYPES
PERFIL_CONSERVADOR  = "conservador"  # Fewer courses, cheaper, slower
PERFIL_NORMAL       = "normal"        # Standard behavior
PERFIL_AGRESIVO     = "agresivo"      # Many courses, more expensive, faster

PERFILES = [PERFIL_CONSERVADOR, PERFIL_NORMAL, PERFIL_AGRESIVO]
PERFIL_PESOS = [0.30, 0.50, 0.20]   # Distribution in the population


# REACTIVE AGENT CLASS
class ReactiveAgent:
    """
    Agente reactivo que simula un estudiante compitiendo en el mercado.

    Arquitectura: condición-acción pura.
    No tiene estado interno más allá de sus atributos actuales y el historial
    simplificado de acciones.
    """

    #Threshold of weeks without progress to activate R3
    STAGNATION_THRESHOLD = 4
    # Market saturation threshold (fraction of agents seeking the skill)
    SATURATION_THRESHOLD = 0.65
    # Maximum number of quits before pausing (R5)
    MAX_DROPOUTS = 3

    def __init__(self, agent_id: int, available_courses: list[dict], seed: int = None):
        """
        Initialize the agent with random attributes.

        Each agent has a different profile generated randomly according to a distribution that simulates a real student population.
        """
        if seed is not None:
            random.seed(seed + agent_id * 17)   # unique seed por agent

        self.id = agent_id
        self.courses_catalog = {c["id"]: c for c in available_courses}

        # Agent profile
        self.profile = random.choices(PERFILES, weights=PERFIL_PESOS, k=1)[0]

        # Total budget 
        mu_log    = 4.5   # ln(90€) ≈ 4.5
        sigma_log = 0.8
        self.budget = max(0.0, math.exp(random.gauss(mu_log, sigma_log)))
        if self.profile == PERFIL_CONSERVADOR:
            self.budget *= 0.6
        elif self.profile == PERFIL_AGRESIVO:
            self.budget *= 1.8

        # Available hours per week 
        base_hours = random.gauss(8, 3)
        self.hours_per_week = max(2.0, base_hours)
        if self.profile == PERFIL_CONSERVADOR:
            self.hours_per_week *= 0.7
        elif self.profile == PERFIL_AGRESIVO:
            self.hours_per_week *= 1.4

        # Goal: random set of skills from the catalog
        all_skills = list({
            skill
            for c in available_courses
            for skill in c["habilidades_aportadas"]
        })
        n_target = random.randint(2, min(8, len(all_skills)))
        self.target_skills = set(random.sample(all_skills, n_target))

        # Initial skills (it might already have some)
        n_initial = random.randint(0, max(1, n_target // 3))
        initial_candidates = list(set(all_skills) - self.target_skills)
        if initial_candidates and n_initial > 0:
            self.owned_skills = set(random.sample(initial_candidates,
                                                   min(n_initial, len(initial_candidates))))
        else:
            self.owned_skills = set()

        #  Internal state of the agent
        self.budget_spent = 0.0
        self.weeks_elapsed = 0.0
        self.weeks_without_progress = 0
        self.n_dropouts = 0
        self.n_paused_weeks = 0
        self.courses_completed: list[str] = []
        self.action_log: list[str] = []
        self.is_paused = False
        self.goal_reached = False

    @property
    def budget_remaining(self) -> float:
        return self.budget - self.budget_spent

    @property
    def missing_skills(self) -> set[str]:
        return self.target_skills - self.owned_skills

    def perceive(self, market_stats: dict) -> dict:
        """
        The agent perceives the environment: its own attributes plus market statistics. 
        It has no memory of past perceptions (purely reactive).
        """
        return {
            "budget_remaining": self.budget_remaining,
            "hours_per_week": self.hours_per_week,
            "missing_skills": self.missing_skills,
            "weeks_without_progress": self.weeks_without_progress,
            "n_dropouts": self.n_dropouts,
            "market_demand": market_stats.get("skill_demand", {}),
            "is_paused": self.is_paused,
        }

    def select_action(self, perception: dict) -> tuple[str, dict]:
        """
        Select the action to perform based on the condition-action rules.

        The rules are evaluated in order of priority (the first ones carry more weight).

        Returns (action_name, parameters).
        """

        # R5 (highest priority): too many dropouts → pause
        if perception["n_dropouts"] >= self.MAX_DROPOUTS:
            return ("PAUSE", {"weeks": 3})

        # If it is paused, do nothing
        if perception["is_paused"]:
            return ("WAIT", {})

        # R3: no progress for too long → reduce goal
        if perception["weeks_without_progress"] >= self.STAGNATION_THRESHOLD:
            return ("REDUCE_GOAL", {})

        missing = perception["missing_skills"]
        if not missing:
            return ("DONE", {})

        # select goal skill
        demand = perception["market_demand"]

        # R2: Avoid highly saturated skills if alternatives exist.
        unsaturated = {
            s for s in missing
            if demand.get(s, 0) < self.SATURATION_THRESHOLD
        }
        skill_target = random.choice(list(unsaturated)) if unsaturated \
                       else random.choice(list(missing))

        # Search for a course for the skill
        providers = self._find_providers(skill_target)
        if not providers:
            return ("SKIP_SKILL", {"skill": skill_target})

        # R1: if the best course is too expensive, search for a free alternative
        cheapest = min(providers, key=lambda c: c["coste"])
        if cheapest["coste"] > perception["budget_remaining"]:
            free_alternatives = [c for c in providers if c["coste"] == 0]
            if free_alternatives:
                chosen = random.choice(free_alternatives)
                return ("TAKE_COURSE_FREE", {"course": chosen})
            else:
                return ("SKIP_SKILL", {"skill": skill_target})

        # R4: if the course is too intensive, search for a lighter one
        if cheapest["horas_semana"] > perception["hours_per_week"] * 1.3:
            lighter = [c for c in providers
                       if c["horas_semana"] <= perception["hours_per_week"]]
            if lighter:
                chosen = min(lighter, key=lambda c: c["coste"])
                return ("TAKE_COURSE_LIGHT", {"course": chosen})

        return ("TAKE_COURSE", {"course": cheapest})

    def execute_action(self, action: str, params: dict) -> dict:
        """
        Ejecuta la acción seleccionada y actualiza el estado del agente.
        Retorna un resumen de lo que ocurrió.
        """
        summary = {"action": action, "agent_id": self.id}

        if action == "DONE":
            self.goal_reached = True
            self.action_log.append("META ALCANZADA")
            summary["detail"] = "Objetivo completado"

        elif action == "PAUSE":
            self.is_paused = True
            self.n_paused_weeks += params.get("weeks", 3)
            self.action_log.append(f"PAUSA ({params.get('weeks')} sem.)")
            summary["detail"] = "Agente en pausa por exceso de abandonos"

        elif action == "WAIT":
            self.weeks_without_progress += 1
            # After the break, resume
            if self.weeks_without_progress >= params.get("pause_duration", 3):
                self.is_paused = False
                self.weeks_without_progress = 0
            summary["detail"] = "Esperando"

        elif action == "REDUCE_GOAL":
            # Remove the most difficult skill (most prerequisites)
            if self.missing_skills:
                hardest = max(
                    self.missing_skills,
                    key=lambda s: sum(
                        len(c.get("prerequisitos", []))
                        for c in self.courses_catalog.values()
                        if s in c.get("habilidades_aportadas", [])
                    )
                )
                self.target_skills.discard(hardest)
                self.weeks_without_progress = 0
                self.action_log.append(f"REDUCIR META: quita {hardest}")
                summary["detail"] = f"Objetivo reducido: eliminada '{hardest}'"

        elif action in ("TAKE_COURSE", "TAKE_COURSE_FREE", "TAKE_COURSE_LIGHT"):
            course = params.get("course")
            if course:
                # Simulate whether you complete or drop out of the course
                p_success = 0.85 if self.profile == PERFIL_AGRESIVO else \
                            0.90 if self.profile == PERFIL_NORMAL else 0.80

                if random.random() < p_success:
                    # Success
                    self.owned_skills.update(course.get("habilidades_aportadas", []))
                    self.budget_spent += course.get("coste", 0)
                    duration = course.get("duracion_semanas", 4)
                    self.weeks_elapsed += duration
                    self.weeks_without_progress = 0
                    self.courses_completed.append(course["id"])
                    self.action_log.append(f"COMPLETADO: {course['nombre']}")
                    summary["detail"] = f"Completado: {course['nombre']}"
                    summary["skill_gained"] = course.get("habilidades_aportadas", [])
                else:
                    # Abandonment

                    self.n_dropouts += 1
                    self.weeks_elapsed += course.get("duracion_semanas", 4) * 0.5
                    self.weeks_without_progress += 2
                    self.action_log.append(f"ABANDONO: {course['nombre']}")
                    summary["detail"] = f"Abandono: {course['nombre']}"

        elif action == "SKIP_SKILL":
            self.weeks_without_progress += 1
            summary["detail"] = f"Sin recursos para: {params.get('skill')}"

        return summary

    def _find_providers(self, skill: str) -> list[dict]:
        """Devuelve todos los cursos del catálogo que aportan esta habilidad."""
        return [
            c for c in self.courses_catalog.values()
            if skill in c.get("habilidades_aportadas", [])
        ]

#Simulate the agent population

def run_reactive_agents(
    available_courses: list[dict],
    n_agents: int = 100,
    n_weeks: int = 52,
    seed: int = 42
) -> dict:
    """
    Simulate N reactive agents over N weeks in the course marketplace.

    Parameters:
    available_courses: complete course catalog
    n_agents: number of agents to simulate
    n_weeks: simulation duration in weeks
    seed: seed for reproducibility

    Returns a dictionary with market statistics:
    - skill_demand: fraction of agents seeking each skill
    - course_demand: fraction of agents who took each course
    - success_rate: fraction of agents who achieved their goal
    - avg_completion: average percentage of goal completed
    - most_competitive_skills: top 5 most sought-after skills
    - least_competitive_skills: top 5 least sought-after skills
    - profile_stats: statistics by profile
    - market_insights: list of textual insights for the LLM
    """
    random.seed(seed)

    # Create agents
    agents = [
        ReactiveAgent(i, available_courses, seed=seed + i)
        for i in range(n_agents)
    ]

    # Comulative Statistics
    skill_demand_count: dict[str, int] = defaultdict(int)
    for agent in agents:
        for skill in agent.target_skills:
            skill_demand_count[skill] += 1

    course_taken_count: dict[str, int] = defaultdict(int)
    profile_success   = defaultdict(lambda: {"total": 0, "success": 0})

    #Week by week simulation 
    # Calculate demand statistics (static during the simulation)
    skill_demand_frac = {
        skill: count / n_agents
        for skill, count in skill_demand_count.items()
    }
    market_stats = {"skill_demand": skill_demand_frac}

    for week in range(n_weeks):
        for agent in agents:
            if agent.goal_reached:
                continue

            # 1. Perceive
            perception = agent.perceive(market_stats)

            # 2. Select action
            action, params = agent.select_action(perception)

            # 3. Execute action
            result = agent.execute_action(action, params)

            # 4. Update cumulative statistics
            for cid in result.get("courses_completed", []):
                course_taken_count[cid] += 1
            for skill in result.get("skill_gained", []):
                pass  

    # Final statistics calculation
    n_success = sum(1 for a in agents if a.goal_reached)
    success_rate = n_success / n_agents

    # medium completeness of goals (fraction of target skills acquired) 
    completions = []
    for agent in agents:
        if len(agent.target_skills) > 0:
            frac = len(agent.owned_skills & agent.target_skills) / len(agent.target_skills)
            completions.append(frac)
    avg_completion = sum(completions) / max(len(completions), 1)

    # Statistics by profile
    for agent in agents:
        pf = agent.profile
        profile_success[pf]["total"] += 1
        if agent.goal_reached:
            profile_success[pf]["success"] += 1

    profile_stats = {
        pf: {
            "total": data["total"],
            "success": data["success"],
            "success_rate": round(data["success"] / max(data["total"], 1), 3)
        }
        for pf, data in profile_success.items()
    }

    # Top 5 most competitive skills (highest demand fraction)
    sorted_skills = sorted(skill_demand_frac.items(), key=lambda x: x[1], reverse=True)
    most_competitive  = sorted_skills[:5]
    least_competitive = sorted_skills[-5:]

    # Top 5 most taken courses
    course_name_map = {c["id"]: c["nombre"] for c in available_courses}
    sorted_courses = sorted(course_taken_count.items(), key=lambda x: x[1], reverse=True)
    top_courses = [(course_name_map.get(cid, cid), count) for cid, count in sorted_courses[:5]]

    # Generate textual insights for the LLM based on the statistics
    insights = []

    if most_competitive:
        top_skill, top_frac = most_competitive[0]
        insights.append(
            f"La habilidad más demandada en el mercado simulado es "
            f"'{top_skill}' ({top_frac*100:.0f}% de agentes la buscan)."
        )

    if least_competitive:
        low_skill, low_frac = least_competitive[-1]
        insights.append(
            f"La habilidad menos competida es '{low_skill}' "
            f"({low_frac*100:.0f}% de agentes). Diferenciarse aquí puede ser ventajoso."
        )

    if success_rate < 0.3:
        insights.append(
            f"Solo el {success_rate*100:.0f}% de agentes alcanzó su meta en {n_weeks} semanas. "
            f"Los objetivos en este mercado son difíciles de completar a corto plazo."
        )
    elif success_rate > 0.7:
        insights.append(
            f"El {success_rate*100:.0f}% de agentes logró su meta. "
            f"El mercado es accesible para perfiles motivados."
        )

    cons_rate = profile_stats.get(PERFIL_CONSERVADOR, {}).get("success_rate", 0)
    agr_rate  = profile_stats.get(PERFIL_AGRESIVO, {}).get("success_rate", 0)
    if agr_rate > cons_rate + 0.2:
        insights.append(
            "Los agentes con perfil agresivo (más horas, más cursos) tienen "
            f"significativamente más éxito ({agr_rate*100:.0f}% vs {cons_rate*100:.0f}%). "
            "Considera intensificar tu plan de estudio."
        )

    return {
        "skill_demand": skill_demand_frac,
        "course_demand": {course_name_map.get(cid, cid): count for cid, count in sorted_courses[:10]},
        "success_rate": round(success_rate, 3),
        "avg_completion": round(avg_completion, 3),
        "n_agents": n_agents,
        "n_weeks_simulated": n_weeks,
        "most_competitive_skills": most_competitive,
        "least_competitive_skills": least_competitive,
        "top_courses": top_courses,
        "profile_stats": profile_stats,
        "market_insights": insights,
    }
