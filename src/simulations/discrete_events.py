"""
This module models a student's actual week-by-week progress as a sequence of events:

EVENT DESCRIPTION
───────────────────────────────────────────────
    COURSE_START The student begins a new course.
    COURSE_END The student completes a course. Successful course
    COURSE_DROPOUT The student drops out of a course midway
    UNFORESEEN EVENT An external event occurs that interrupts learning
    UNFORESEEN_END The unforeseen event ends, the student resumes
    GOAL_ACHIEVED All target skills covered
    SIMULATION_END Time ran out without reaching the goal

ARCHITECTURE
------------
Event queue (priority queue ordered by time):[(time, event_type, data), ...]

Event processor:
While the queue is not empty and time < limit:
1. Extract the event with the shortest time
2. Update the system state
3. Generate new events based on transitions
4. Advance the clock to the time of the processed event

System states:
- clock: current week
- covered_skills: skills already acquired
- active_course: course in progress (only one at a time)
- waiting_courses: pending courses (prerequisites not yet released)
- completed: list of courses completed
- event_log: event history
"""

import heapq
import random
import math


# TYPES OF EVENTS

# Event type constants (also priorities: lower = more urgent)
IMPREVISTO_FIN  = 0
CURSO_FIN       = 1
PREREQ_OK       = 2
CURSO_INICIO    = 3
IMPREVISTO      = 4
CURSO_ABANDONO  = 5
META_ALCANZADA  = 6
SIMULACION_FIN  = 99

EVENT_NAMES = {
    IMPREVISTO_FIN : "IMPREVISTO_FIN",
    CURSO_FIN      : "CURSO_FIN",
    PREREQ_OK      : "PREREQ_OK",
    CURSO_INICIO   : "CURSO_INICIO",
    IMPREVISTO     : "IMPREVISTO",
    CURSO_ABANDONO : "CURSO_ABANDONO",
    META_ALCANZADA : "META_ALCANZADA",
    SIMULACION_FIN : "SIMULACION_FIN",
}



# RANDOM TIME GENERATORS 

def _normal_sample(mu: float, sigma: float, lo: float, hi: float) -> float:
    for _ in range(50):
        u1 = max(random.random(), 1e-10)
        u2 = random.random()
        z  = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
        v  = mu + sigma * z
        if lo <= v <= hi:
            return v
    return mu   # fallback


def _exponential_sample(lam: float) -> float:
    u = max(random.random(), 1e-10)
    return -math.log(u) / lam


def _uniform(a: float, b: float) -> float:
    return a + (b - a) * random.random()


# SYSTEM STATE

class SimState:
    """
    Complete state of the system at any instant of the simulation.
    It is updated as events are processed.
    """

    def __init__(self, trajectory: list[dict], target_skills: set[str], possessed_skills: set[str]):
        self.clock = 0.0                         # Virtual clock (weeks)
        self.covered_skills = set(possessed_skills)  # Skills already acquired
        self.target_skills  = set(target_skills)     # Target skills

        # Queue of pending courses (not yet started) indexed by id
        self.pending: dict[str, dict] = {c["id"]: c for c in trajectory}
        # Course currently in progress (only one at a time)
        self.active_course: dict | None = None
        # Completed courses (in order)
        self.completed: list[dict] = []
        # If there is an active incident
        self.in_incident = False
        # Log of events to display the timeline
        self.event_log: list[dict] = []

    @property
    def remaining_skills(self) -> set[str]:
        """Target skills that have not yet been covered."""
        return self.target_skills - self.covered_skills

    @property
    def goal_reached(self) -> bool:
        return self.target_skills.issubset(self.covered_skills)

    def courses_available(self) -> list[dict]:
        """
        Returns the list of pending courses whose prerequisites are already covered.
        A course is available if all its prerequisite skills are in covered_skills.
        """
        available = []
        for cid, course in self.pending.items():
            prereqs = set(course.get("prerequisitos", []))
            if prereqs.issubset(self.covered_skills):
                available.append(course)
        # Sort by shortest duration to start with the fastest ones
        available.sort(key=lambda c: c["duracion_semanas"])
        return available


# EVENT QUEUE (min-heap per time)

class EventQueue:
    """
    Priority queue of events ordered by occurrence time.
    Uses heapq (min-heap): the closest event is always at the top.
    """

    def __init__(self):
        self._heap = []
        self._counter = 0  

    def push(self, time: float, event_type: int, data: dict = None):
        """Insert a new event into the queue."""
        heapq.heappush(self._heap, (time, self._counter, event_type, data or {}))
        self._counter += 1

    def pop(self) -> tuple[float, int, dict]:
        """Extract the closest event. Returns (time, type, data)."""
        time, _, event_type, data = heapq.heappop(self._heap)
        return time, event_type, data

    def __len__(self):
        return len(self._heap)


# EVENT PROCESSORS

def _process_curso_fin(state: SimState, queue: EventQueue, data: dict):
    """
    The student successfully completes the active course.

    Transitions:
      1. Mark course as completed
      2. Add its skills to covered_skills
      3. Generate PREREQ_OK events for new skills
      4. If the goal is reached → META_ALCANZADA
      5. If not → attempt to start the next available course
    """
    course = state.active_course
    if course is None:
        return

    state.active_course = None
    state.completed.append(course)
    del state.pending[course["id"]]

    new_skills = set(course["habilidades_aportadas"]) - state.covered_skills
    state.covered_skills.update(course["habilidades_aportadas"])

    state.event_log.append({
        "week": round(state.clock, 1),
        "event": "CURSO_FIN",
        "course_name": course["nombre"],
        "skills_granted": list(course["habilidades_aportadas"]),
        "detail": f"Completado: {course['nombre']} → +{list(new_skills)}"
    })

    # Goal achieved?
    if state.goal_reached:
        queue.push(state.clock, META_ALCANZADA, {})
        return

    # Try starting the next course
    _try_start_next(state, queue)


def _process_curso_abandono(state: SimState, queue: EventQueue, data: dict):
    """
    The student abandons the active course.

    The course returns to the pending queue (can be resumed later).
    A time penalty is added before it can be restarted.
    """
    course = state.active_course
    if course is None:
        return

    state.active_course = None
    penalty = _uniform(course["duracion_semanas"] * 0.3, course["duracion_semanas"] * 0.8)

    state.event_log.append({
        "week": round(state.clock, 1),
        "event": "CURSO_ABANDONO",
        "course_name": course["nombre"],
        "prerequisites": list(course.get("prerequisitos", [])),
        "skills_granted": [],
        "detail": f"Abandonado: {course['nombre']}"
    })

    
    restart_time = state.clock + penalty
    queue.push(restart_time, CURSO_INICIO, {"course_id": course["id"]})


def _process_imprevisto(state: SimState, queue: EventQueue, data: dict):
    """
    An unforeseen event occurs that pauses the learning process.

    If there is an active course, it is "frozen" (its completion date is postponed).
    """
    if state.in_incident:
        return   

    state.in_incident = True
    duration = _uniform(0.5, 2.5)   # 0.5 to 2.5 weeks break

    state.event_log.append({
        "week": round(state.clock, 1),
        "event": "IMPREVISTO",
        "detail": f"Pausa de {duration:.1f} semanas por imprevisto"
    })

    # If there is an active course, postpone its end
    if state.active_course is not None:
        # Reschedule the end of the current course to the end of the unforeseen event
        course = state.active_course
        remaining = data.get("remaining_time", course["duracion_semanas"])
        queue.push(state.clock + duration + remaining, CURSO_FIN,
                   {"course_id": course["id"]})

    queue.push(state.clock + duration, IMPREVISTO_FIN, {})


def _process_imprevisto_fin(state: SimState, queue: EventQueue, data: dict):
    """An unforeseen event ends; the student resumes learning."""
    state.in_incident = False
    state.event_log.append({
        "week": round(state.clock, 1),
        "event": "IMPREVISTO_FIN",
        "detail": "Estudiante retoma el plan"
    })


def _try_start_next(state: SimState, queue: EventQueue):
    """
    Try starting the next available course.
    Only one course can be active at a time.    
    """
    if state.active_course is not None:
        return   
    if state.in_incident:
        return   

    available = state.courses_available()
    if not available:
        return   

    # Choose the first one available (already sorted by duration)
    course = available[0]
    state.active_course = course

    nominal = float(course["duracion_semanas"])
    real_duration = _normal_sample(nominal, nominal * 0.20,
                                   nominal * 0.5, nominal * 2.5)

    prereqs = course.get("prerequisitos", [])
    prereqs_str = ", ".join(prereqs) if prereqs else "ninguno"
    state.event_log.append({
        "week": round(state.clock, 1),
        "event": "CURSO_INICIO",
        "course_name": course["nombre"],
        "prerequisites": list(course.get("prerequisitos", [])),
        "detail": f"Iniciando: {course['nombre']} (~{real_duration:.1f} sem.)"
    })

    # Are you dropping out of this course?
    p_drop = _dropout_prob(course)
    if random.random() < p_drop:
        # It abandons halfway through the estimated time.
        abandon_time = state.clock + real_duration * _uniform(0.3, 0.7)
        queue.push(abandon_time, CURSO_ABANDONO, {"course_id": course["id"]})
    else:
        # Finish the course
        end_time = state.clock + real_duration
        queue.push(end_time, CURSO_FIN, {"course_id": course["id"],
                                          "remaining_time": real_duration})

    # Generate a possible unforeseen event during the course (Poisson process)
    # Time until unforeseen event ~ Exponential (λ=0.05 per week)
    time_to_incident = _exponential_sample(0.05)
    if time_to_incident < real_duration:
        incident_time = state.clock + time_to_incident
        queue.push(incident_time, IMPREVISTO,
                   {"remaining_time": real_duration - time_to_incident})


def _dropout_prob(course: dict, base: float = 0.12) -> float:
    """Simplified dropout probability"""
    p = base
    if course.get("duracion_semanas", 4) > 8:
        p += 0.06
    if course.get("horas_semana", 6) > 10:
        p += 0.05
    return max(0.03, min(0.35, p))


def run_discrete_event_simulation(
    trajectory: list[dict],
    target_skills: list[str],
    possessed_skills: list[str],
    max_weeks: float = 200.0,
    seed: int = 42
) -> dict:
    """
    Run a discrete event simulation of the entire trajectory.

    Parameters:
    trajectory: list of courses in topological order
    target_skills: skills the user wants to acquire
    possessed_skills: skills the user already possesses
    max_weeks: time limit (the simulation stops here if it hasn't finished)
    seed: random seed

    Returns a dict containing:
    - success: bool (was the goal reached?)
    - total_weeks: float (actual weeks of the simulation)
    - completed_courses: list of completed courses
    - event_log: complete timeline
    - skills_acquired: skills acquired
    - skills_missing: skills that were missed
    """
    random.seed(seed)

    state = SimState(trajectory, set(target_skills), set(possessed_skills))
    queue = EventQueue()

    # Initial event: Attempt to start the first available course 
    queue.push(0.0, CURSO_INICIO, {})
    # Security event: time limit
    queue.push(max_weeks, SIMULACION_FIN, {})

    success = False

    while len(queue) > 0:
        time, event_type, data = queue.pop()

        state.clock = time

        if event_type == CURSO_INICIO:
            _try_start_next(state, queue)

        elif event_type == CURSO_FIN:
            _process_curso_fin(state, queue, data)

        elif event_type == CURSO_ABANDONO:
            _process_curso_abandono(state, queue, data)

        elif event_type == PREREQ_OK:
            # A prerequisite has been released: check for available new courses
            state.event_log.append({
                "week": round(state.clock, 1),
                "event": "PREREQ_OK",
                "detail": f"Habilidad liberada: {data.get('skill', '?')}"
            })
            _try_start_next(state, queue)

        elif event_type == IMPREVISTO:
            _process_imprevisto(state, queue, data)

        elif event_type == IMPREVISTO_FIN:
            _process_imprevisto_fin(state, queue, data)
            _try_start_next(state, queue)

        elif event_type == META_ALCANZADA:
            state.event_log.append({
                "week": round(state.clock, 1),
                "event": "META_ALCANZADA",
                "detail": "¡Todas las habilidades objetivo adquiridas!"
            })
            success = True
            break

        elif event_type == SIMULACION_FIN:
            state.event_log.append({
                "week": round(state.clock, 1),
                "event": "SIMULACION_FIN",
                "detail": f"Tiempo agotado. Quedan {len(state.remaining_skills)} habilidades."
            })
            break

    return {
        "success": success,
        "total_weeks": round(state.clock, 1),
        "completed_courses": state.completed,
        "event_log": state.event_log,
        "skills_acquired": list(state.covered_skills & state.target_skills),
        "skills_missing": list(state.remaining_skills),
        "n_dropouts": sum(
            1 for e in state.event_log if e["event"] == "CURSO_ABANDONO"
        ),
    }
