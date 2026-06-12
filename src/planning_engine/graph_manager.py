"""
skills_graph.py
================
Graph engine for the professional trajectory planner.

Responsibilities:
  - Build a directed acyclic graph (DAG) of prerequisites between courses
  - Find ALL courses needed to cover the missing skills
  - Sort those courses respecting dependencies (topological order)
  - Apply user constraints (time, cost, already possessed skills)
  - Return candidate trajectories for the LLM to evaluate

External dependencies: only Python standard library (no networkx).
"""

from collections import defaultdict, deque


# =============================================================================
# MAIN CLASS: SkillsGraph
# =============================================================================

class SkillsGraph:
    """
    Represents the DAG of courses and their prerequisites.
    
    Internally uses adjacency lists:
      - forward_graph:   course -> list of courses that depend on it
      - reverse_graph:   course -> list of its prerequisites
    """

    def __init__(self, courses: list):
        """
        Builds the graph from the course catalog (list of dicts).
        
        Parameter:
          courses: list of dicts with keys 'id', 'nombre', 'prerequisitos',
                   'habilidades_aportadas', 'duracion_semanas', 'horas_semana', 'coste'
        """
        # Dictionary course_id -> full course data
        self.courses = {c["id"]: c for c in courses}
        
        # Auxiliary maps to find courses by the skills they provide
        # skill -> list of course ids that provide it
        self.skill_to_courses: dict[str, list[str]] = defaultdict(list)
        
        # Dependency graph: node A -> nodes that depend on A
        self.forward_graph: dict[str, list[str]] = defaultdict(list)
        
        # Reverse graph: node A -> prerequisites of A
        self.reverse_graph: dict[str, list[str]] = defaultdict(list)
        
        # Build the maps
        self._build_graph()

    def _build_graph(self):
        """
        Step 1: Populate skill_to_courses.
        Step 2: Translate prerequisites (skill names) to course ids.
        Step 3: Add edges to the DAG.
        """
        # map skill -> courses that provide it
        for course_id, course in self.courses.items():
            for skill in course["habilidades_aportadas"]:
                self.skill_to_courses[skill].append(course_id)

        # for each course, translate its prerequisites
        for course_id, course in self.courses.items():
            for prereq_skill in course["prerequisitos"]:
                # A prerequisite is a skill; there may be several courses that provide it.
                # We take ALL possible "providers" of that skill.
                providers = self.skill_to_courses.get(prereq_skill, [])
                for provider_id in providers:
                    if provider_id != course_id:  # avoid self-loops
                        # Edge: provider -> course (provider must be taken first)
                        self.forward_graph[provider_id].append(course_id)
                        self.reverse_graph[course_id].append(provider_id)

    def get_providers(self, skill: str) -> list[str]:
        """Devuelve los ids de cursos que aportan una habilidad dada."""
        return self.skill_to_courses.get(skill, [])
    
    def _expand_prerequisites(
        self,
        initial_courses: set[str],
        possessed_skills: set[str]
    ) -> set[str]:
        """
        BFS inverso: dado un conjunto de cursos objetivo, añade todos sus
        prerequisitos transitivos que el usuario aún no posee.
 
        Parámetros:
          initial_courses:  set de ids de cursos objetivo ya seleccionados
          possessed_skills: set de habilidades que el usuario ya tiene
 
        Retorna:
          set con todos los ids de cursos necesarios (objetivo + prerequisitos)
        """
        covered = set(possessed_skills)
        queue = deque(initial_courses)
        all_needed = set(initial_courses)
        visited = set(initial_courses)
 
        while queue:
            cid = queue.popleft()
            course = self.courses[cid]
 
            for prereq_skill in course["prerequisitos"]:
                if prereq_skill in covered:
                    continue
 
                providers = self.skill_to_courses.get(prereq_skill, [])
                if not providers:
                    continue
 
                # Elige el proveedor de menor coste efectivo como heurística
                # de desempate (el motor de búsqueda ya habrá elegido los
                # principales; esto es solo para prerequisitos de soporte)
                best = min(
                    providers,
                    key=lambda x: (
                        self.courses[x]["coste"] +
                        self.courses[x]["duracion_semanas"] * self.courses[x]["horas_semana"] * 2
                    )
                )
 
                if best not in visited:
                    visited.add(best)
                    all_needed.add(best)
                    queue.append(best)
                    covered.update(self.courses[best]["habilidades_aportadas"])
 
        return all_needed

   

    def _topological_order(self, subset_ids: set[str]) -> list[dict]:
        """
        Sorts the courses in the subset respecting their prerequisites.
        
        Kahn's algorithm (BFS topological sort):
          1. Compute the in-degree of each node in the subgraph.
          2. Initialize a queue with all nodes with in-degree 0 (no pending prerequisites).
          3. While the queue is not empty:
             a. Pop a node, add it to the result.
             b. For each successor: decrease its in-degree by 1.
             c. If a successor's in-degree becomes 0 → add it to the queue.
          4. If not all nodes were processed → there is a cycle (error in data).
        
        Advantage of Kahn over DFS: it naturally produces the order starting from
        nodes with no dependencies, which directly corresponds to the learning order.
        """
        # --- Compute in-degree only within the subset ---
        in_degree: dict[str, int] = {cid: 0 for cid in subset_ids}
        
        for cid in subset_ids:
            for successor in self.forward_graph.get(cid, []):
                if successor in subset_ids:
                    in_degree[successor] += 1
        
        # --- Initial queue: nodes without prerequisites in the subgraph ---
        # Sort by effective cost to process shorter/cheaper courses first when tie in degree
        queue = deque(sorted(
            [cid for cid, deg in in_degree.items() if deg == 0],
            key=lambda cid: self.courses[cid]["duracion_semanas"]
        ))
        
        result: list[str] = []
        
        # --- Process the queue ---
        while queue:
            node = queue.popleft()
            result.append(node)
            
            # Decrease in-degree of successors within the subgraph
            for successor in self.forward_graph.get(node, []):
                if successor in subset_ids:
                    in_degree[successor] -= 1
                    if in_degree[successor] == 0:
                        queue.append(successor)
        
        # --- Detect cycles (should not happen with correct data) ---
        if len(result) != len(subset_ids):
            # There is a cycle: return what we could process + problematic ones at the end
            processed = set(result)
            cyclic = [cid for cid in subset_ids if cid not in processed]
            result.extend(cyclic)
        
        # Return the full course dicts in the correct order
        return [self.courses[cid] for cid in result]


    def build_result(
        self,
        ordered_path: list[dict],
        warnings: list[str] = None,
        engine_name: str = "",
        criterion: str = ""
    ) -> dict:
        """
        Package an ordered path into the standard result dictionary
        that the rest of the system expects
        """
        total_hours = sum(c["duracion_semanas"] * c["horas_semana"] for c in ordered_path)
        total_cost = sum(c["coste"] for c in ordered_path)
        total_weeks = sum(c["duracion_semanas"] for c in ordered_path)
        covered = set()
        for c in ordered_path:
            covered.update(c["habilidades_aportadas"])
 
        return {
            "path": ordered_path,
            "total_hours": total_hours,
            "total_weeks": total_weeks,
            "total_cost": total_cost,
            "covered_skills": covered,
            "warnings": warnings or [],
            "engine_used": engine_name,
            "criterion": criterion,
        }