#!/usr/bin/env python3
"""
Modern graphical interface for the Professional Trajectory Planner.
Two-column design: left form, right missing skills and courses.
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import threading
import os
from dotenv import load_dotenv

try:
    from ttkthemes import ThemedTk
    USE_THEMES = True
except ImportError:
    USE_THEMES = False
    print("Note: for better appearance install 'pip install ttkthemes'")

from src.llm.client import create_client
from src.planning_engine.graph_manager import SkillsGraph
from src.planning_engine.load_data import load_data
from src.presentation.initial_interpreter import extract_skills_from_text
from src.planning_engine.engine_router import select_and_run
from src.planning_engine.strategy_single_objective import run as run_single_objective
from src.llm.trajectory_evaluation import evaluate_trajectory
from src.llm.trajectory_comparator import compare_trajectories
from src.llm.step_justifier import justify_steps
from src.llm.relaxation_suggester import suggest_relaxation
from src.simulations.discrete_events import run_discrete_event_simulation
from src.simulations.reactive_agents import run_reactive_agents

load_dotenv()
API_KEY = os.getenv("MISTRAL_API_KEY")
if not API_KEY:
    raise ValueError("MISTRAL_API_KEY not found in .env")

print("Loading catalogs...")
skills_catalog, courses_catalog = load_data()
print("Building graph...")
graph = SkillsGraph(courses_catalog)
print("Initializing LLM client...")
client = create_client(API_KEY)


class ModernPlannerGUI:
    def __init__(self, root):
        self.root = root
        if USE_THEMES:
            root.set_theme("arc")
        else:
            style = ttk.Style()
            style.theme_use('clam')
            style.configure("TLabel", font=("Segoe UI", 10))
            style.configure("TButton", font=("Segoe UI", 10))
            style.configure("TEntry", font=("Segoe UI", 10))
            style.configure("TLabelframe.Label", font=("Segoe UI", 10, "bold"))

        self.root.title("🎓 Career Path Planner")
        self.root.geometry("1200x700")
        self.root.minsize(1000, 600)

        # Control variables
        self.goal_var = tk.StringVar()
        self.base_skills_var = tk.StringVar()
        self.only_free_var = tk.BooleanVar(value=False)
        self.priority_var = tk.StringVar(value="auto")
        self.max_cost_var = tk.StringVar()
        self.max_weeks_var = tk.StringVar()
        self.max_hpw_var = tk.StringVar()
        self.weight_cost_var = tk.StringVar(value="50")
        self.weight_time_var = tk.StringVar(value="50")
        self.courses_catalog = courses_catalog   # after load_data()

        # Result and state variables
        self.current_missing = None
        self.current_base_skills = None
        self.current_goal = None
        self.last_goal = None
        self.last_base_desc = None
        self.current_result = None
        self.current_justifications = None
        self.current_evaluation = None
        self.current_alt_result = None
        self.current_comparison = None

        self.build_ui()

    def build_ui(self):
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # ----- LEFT PANEL -----
        left_frame = ttk.Frame(main_paned, width=350)
        main_paned.add(left_frame, weight=1)

        ttk.Label(left_frame, text="Configuración", font=("Segoe UI", 14, "bold")).pack(anchor="w", pady=(0, 10))

        ttk.Label(left_frame, text="🎯 Meta profesional:", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(10, 0))
        ttk.Entry(left_frame, textvariable=self.goal_var, width=50).pack(fill="x", pady=5)

        ttk.Label(left_frame, text="🧠 Habilidades base (descripción):", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(10, 0))
        ttk.Entry(left_frame, textvariable=self.base_skills_var, width=50).pack(fill="x", pady=5)

        ttk.Separator(left_frame, orient='horizontal').pack(fill="x", pady=10)

        ttk.Label(left_frame, text="⚙️ Prioridad de búsqueda:", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(0, 5))
        priority_frame = ttk.Frame(left_frame)
        priority_frame.pack(fill="x", pady=5)
        priorities = [
            ("Automático", "auto"),
            ("Menor cantidad de cursos", "bfs"),
            ("Trayectoria más barata", "cost"),
            ("Trayectoria más corta", "time"),
            ("Límites de tiempo y presupuesto", "csp"),
            ("Balance presupuesto/tiempo", "balanced")
        ]
        for text, val in priorities:
            ttk.Radiobutton(priority_frame, text=text, variable=self.priority_var, value=val).pack(anchor="w")

        constraints_frame = ttk.LabelFrame(left_frame, text="Restricciones adicionales")
        constraints_frame.pack(fill="x", pady=10)

        row1 = ttk.Frame(constraints_frame)
        row1.pack(fill="x", padx=5, pady=5)
        ttk.Label(row1, text="💰 Presupuesto máximo($):").pack(side="left", padx=(0, 10))
        ttk.Entry(row1, textvariable=self.max_cost_var, width=12).pack(side="left")

        row2 = ttk.Frame(constraints_frame)
        row2.pack(fill="x", padx=5, pady=5)
        ttk.Label(row2, text="📅 Plazo máximo de semanas:").pack(side="left", padx=(0, 10))
        ttk.Entry(row2, textvariable=self.max_weeks_var, width=12).pack(side="left")

        row3 = ttk.Frame(constraints_frame)
        row3.pack(fill="x", padx=5, pady=5)
        ttk.Label(row3, text="⏱️ Horas/semana máximo por curso:").pack(side="left", padx=(0, 10))
        ttk.Entry(row3, textvariable=self.max_hpw_var, width=12).pack(side="left")

        row4 = ttk.Frame(constraints_frame)
        row4.pack(fill="x", padx=5, pady=5)
        ttk.Checkbutton(row4, text="Solo cursos gratuitos", variable=self.only_free_var).pack(anchor="w")

        weights_frame = ttk.LabelFrame(left_frame, text="Balance de prioridad")
        weights_frame.pack(fill="x", pady=10)
        row5 = ttk.Frame(weights_frame)
        row5.pack(fill="x", padx=5, pady=5)
        ttk.Label(row5, text="Coste (%):").pack(side="left", padx=(0, 5))
        ttk.Entry(row5, textvariable=self.weight_cost_var, width=6).pack(side="left", padx=(0, 20))
        ttk.Label(row5, text="Tiempo (%):").pack(side="left", padx=(0, 5))
        ttk.Entry(row5, textvariable=self.weight_time_var, width=6).pack(side="left")

        self.plan_btn = ttk.Button(left_frame, text="PLANIFICAR TRAYECTORIA", command=self.plan_thread)
        style = ttk.Style()
        style.configure("Black.TButton", foreground="black")
        self.plan_btn.configure(style="Black.TButton")
        self.plan_btn.pack(fill="x", pady=20)

        # ----- RIGHT PANEL -----
        right_frame = ttk.Frame(main_paned)
        main_paned.add(right_frame, weight=2)

        right_paned = ttk.PanedWindow(right_frame, orient=tk.VERTICAL)
        right_paned.pack(fill=tk.BOTH, expand=True)

        # Missing skills
        top_frame = ttk.LabelFrame(right_paned, text="Habilidades que debes adquirir")
        right_paned.add(top_frame, weight=1)

        self.skills_tree = ttk.Treeview(top_frame, columns=("skill",), show="tree headings", height=8)
        self.skills_tree.heading("#0", text="#")
        self.skills_tree.heading("skill", text="Habilidad")
        self.skills_tree.column("#0", width=40, stretch=False)
        self.skills_tree.column("skill", width=250, stretch=True)
        scroll_skills = ttk.Scrollbar(top_frame, orient="vertical", command=self.skills_tree.yview)
        self.skills_tree.configure(yscrollcommand=scroll_skills.set)
        self.skills_tree.pack(side="left", fill="both", expand=True, padx=5, pady=5)
        scroll_skills.pack(side="right", fill="y", pady=5)

        # Recommended courses
        bottom_frame = ttk.LabelFrame(right_paned, text="Ruta de aprendizaje recomendada")
        right_paned.add(bottom_frame, weight=2)

        self.courses_tree = ttk.Treeview(bottom_frame, columns=("num", "nombre", "duracion", "coste", "habilidades"), show="headings")
        self.courses_tree.heading("num", text="#")
        self.courses_tree.heading("nombre", text="Curso")
        self.courses_tree.heading("duracion", text="Duración")
        self.courses_tree.heading("coste", text="Coste")
        self.courses_tree.heading("habilidades", text="Aporta")
        self.courses_tree.column("num", width=40, anchor="center")
        self.courses_tree.column("nombre", width=250)
        self.courses_tree.column("duracion", width=80, anchor="center")
        self.courses_tree.column("coste", width=70, anchor="center")
        self.courses_tree.column("habilidades", width=300)
        scroll_courses = ttk.Scrollbar(bottom_frame, orient="vertical", command=self.courses_tree.yview)
        self.courses_tree.configure(yscrollcommand=scroll_courses.set)
        self.courses_tree.pack(side="left", fill="both", expand=True, padx=5, pady=5)
        scroll_courses.pack(side="right", fill="y", pady=5)

        # Summary
        summary_frame = ttk.LabelFrame(right_paned, text="Resumen de la trayectoria")
        right_paned.add(summary_frame, weight=0)
        self.summary_label = ttk.Label(summary_frame, text="Semanas: -- | Horas: -- | Coste: --", font=("Segoe UI", 10, "bold"))
        self.summary_label.pack(pady=10)

        # Action buttons
        actions_frame = ttk.Frame(right_frame)
        actions_frame.pack(fill="x", pady=10)

        self.justify_btn = ttk.Button(actions_frame, text="✨ Justificar cursos", command=self.show_justifications, state="disabled")
        self.justify_btn.pack(side="left", padx=5)

        self.evaluate_btn = ttk.Button(actions_frame, text="✨ Evaluar ruta", command=self.show_evaluation, state="disabled")
        self.evaluate_btn.pack(side="left", padx=5)

        self.compare_btn = ttk.Button(actions_frame, text="✨ Comparar con otro motor", command=self.compare_alternative, state="disabled")
        self.compare_btn.pack(side="left", padx=5)

        self.simulate_des_btn = ttk.Button(actions_frame, text="✨ Simular curso", command=self.simulate_discrete_events, state="disabled")
        self.simulate_des_btn.pack(side="left", padx=5)

        self.simulate_agents_btn = ttk.Button(actions_frame, text="✨ Ver competencia profesional simulada", command=self.simulate_reactive_agents)
        self.simulate_agents_btn.pack(side="left", padx=5)

        self.progress = ttk.Progressbar(right_frame, mode='indeterminate')
        self.progress.pack(fill="x", pady=5)

    # --------------------------------------------------------------
    # Planning
    # --------------------------------------------------------------
    def plan_thread(self):
        self.plan_btn.config(state="disabled")
        self.progress.start()
        self.justify_btn.config(state="disabled")
        self.evaluate_btn.config(state="disabled")
        self.compare_btn.config(state="disabled")
        self.simulate_des_btn.config(state="disabled")
        #self.simulate_agents_btn.config(state="disabled")

        for item in self.skills_tree.get_children():
            self.skills_tree.delete(item)
        for item in self.courses_tree.get_children():
            self.courses_tree.delete(item)
        self.summary_label.config(text="Semanas: -- | Horas: -- | Coste: --")

        thread = threading.Thread(target=self.do_planning)
        thread.daemon = True
        thread.start()

    def do_planning(self):
        try:
            goal = self.goal_var.get().strip()
            if not goal:
                self.show_error("La meta profesional es obligatoria.")
                return

            base_desc = self.base_skills_var.get().strip()
            goal_changed = (self.last_goal != goal)
            base_changed = (self.last_base_desc != base_desc)
            first_time = (self.current_missing is None)

            if goal_changed or base_changed or first_time:
                base_skills = []
                if base_desc:
                    base_skills = extract_skills_from_text(client, base_desc, "base", skills_catalog)

                necessary_skills = extract_skills_from_text(client, goal, "goal", skills_catalog)
                if not necessary_skills:
                    self.show_error("No se identificaron habilidades necesarias. Reformula la meta.")
                    return

                already_owned = [s for s in necessary_skills if s in base_skills]
                missing = [s for s in necessary_skills if s not in base_skills]

                self.current_missing = missing
                self.current_base_skills = base_skills
                self.current_goal = goal
                self.last_goal = goal
                self.last_base_desc = base_desc
            else:
                missing = self.current_missing
                base_skills = self.current_base_skills

            self.root.after(0, self.update_skills_tree, missing)

            if not missing:
                self.root.after(0, lambda: self.show_info("Ya posees todas las habilidades necesarias."))
                self.root.after(0, self.finish_planning)
                return

            constraints = self.build_constraints()
            result = select_and_run(
                graph=graph,
                missing_skills=missing,
                possessed_skills=base_skills,
                constraints=constraints
            )

            self.current_result = result
            self.current_justifications = None
            self.current_evaluation = None
            self.current_alt_result = None
            self.current_comparison = None

            # Check if planning failed (no courses or warnings)
            plan_failed = (not result.get("path") or result.get("warnings"))
            if plan_failed:
                self.root.after(100, lambda: self._auto_relaxation(result))

            self.root.after(0, self.update_courses_tree, result)
            self.root.after(0, self.finish_planning)

        except Exception as e:
            self.show_error(f"Error en planificación: {str(e)}")
        finally:
            self.progress.stop()
            self.root.after(0, lambda: self.plan_btn.config(state="normal"))

    def update_skills_tree(self, missing_skills):
        for item in self.skills_tree.get_children():
            self.skills_tree.delete(item)
        for i, skill in enumerate(missing_skills, start=1):
            self.skills_tree.insert("", "end", values=(skill,), text=str(i))

    def update_courses_tree(self, result):
        for item in self.courses_tree.get_children():
            self.courses_tree.delete(item)
        path = result.get("path", [])
        for i, curso in enumerate(path, 1):
            coste = f"{curso['coste']}€" if curso['coste'] > 0 else "Gratis"
            duracion = f"{curso['duracion_semanas']} sem"
            habilidades = ", ".join(curso['habilidades_aportadas'][:3]) + ("..." if len(curso['habilidades_aportadas']) > 3 else "")
            self.courses_tree.insert("", "end", values=(i, curso['nombre'], duracion, coste, habilidades))
        total_weeks = result.get('total_weeks', 0)
        total_hours = result.get('total_hours', 0)
        total_cost = result.get('total_cost', 0)
        self.summary_label.config(text=f"📅 {total_weeks} semanas   |   ⏱️ {total_hours} horas   |   💰 {total_cost} €")

    def finish_planning(self):
        self.justify_btn.config(state="normal")
        self.evaluate_btn.config(state="normal")
        self.compare_btn.config(state="normal")
        if self.current_result and self.current_result.get("path"):
            self.simulate_des_btn.config(state="normal")
            self.simulate_agents_btn.config(state="normal")
        else:
            self.simulate_des_btn.config(state="disabled")
            self.simulate_agents_btn.config(state="disabled")

    # --------------------------------------------------------------
    # Constraints
    # --------------------------------------------------------------
    def build_constraints(self):
        priority = self.priority_var.get()
        only_free = self.only_free_var.get()
        constraints = {
            "max_cost": None, "max_weeks": None, "max_hours_per_week": None,
            "only_free": only_free,
            "optimize": None,
            "weight_cost": None, "weight_time": None,
        }
        try:
            if self.max_cost_var.get().strip():
                constraints["max_cost"] = float(self.max_cost_var.get())
            if self.max_weeks_var.get().strip():
                constraints["max_weeks"] = int(self.max_weeks_var.get())
            if self.max_hpw_var.get().strip():
                constraints["max_hours_per_week"] = int(self.max_hpw_var.get())
        except:
            pass

        if priority == "cost":
            constraints["optimize"] = "cost"
        elif priority == "time":
            constraints["optimize"] = "time"
        elif priority == "balanced":
            try:
                wc = float(self.weight_cost_var.get()) / 100.0
                wt = float(self.weight_time_var.get()) / 100.0
                total = wc + wt
                constraints["weight_cost"] = wc / total if total > 0 else 0.5
                constraints["weight_time"] = wt / total if total > 0 else 0.5
            except:
                constraints["weight_cost"] = 0.5
                constraints["weight_time"] = 0.5
        return constraints

    # --------------------------------------------------------------
    # Justifications
    # --------------------------------------------------------------
    def show_justifications(self):
        if not self.current_result or not self.current_result["path"]:
            messagebox.showinfo("Info", "Primero planifica una trayectoria.")
            return
        if self.current_justifications is None:
            self.progress.start()
            self.justify_btn.config(state="disabled")
            def run():
                try:
                    justs = justify_steps(client, self.current_goal, self.current_result["path"])
                    self.root.after(0, self._on_justifications_done, justs)
                except Exception as e:
                    self.root.after(0, self._on_justifications_error, str(e))
            threading.Thread(target=run, daemon=True).start()
        else:
            self._show_window(
                title="Justificaciones de los cursos",
                content=self._format_justifications(self.current_justifications),
                width=800, height=500
            )

    def _on_justifications_done(self, justs):
        self.progress.stop()
        self.justify_btn.config(state="normal")
        self.current_justifications = justs
        if justs:
            self._show_window(
                title="Justificaciones de los cursos",
                content=self._format_justifications(justs),
                width=800, height=500
            )
        else:
            messagebox.showwarning("Advertencia", "No se obtuvieron justificaciones.")

    def _on_justifications_error(self, err):
        self.progress.stop()
        self.justify_btn.config(state="normal")
        messagebox.showerror("Error", f"Error al generar justificaciones:\n{err}")

    def _format_justifications(self, justs):
        just_map = {j["id"]: j.get("justificacion", "Sin justificación.") for j in justs}
        lines = []
        for curso in self.current_result["path"]:
            lines.append(f"🔹 {curso['nombre']}")
            just = just_map.get(curso["id"], "Sin justificación.")
            lines.append(f"   {just}\n")
        return "\n".join(lines)

    # --------------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------------
    def show_evaluation(self):
        if not self.current_result or not self.current_result["path"]:
            messagebox.showinfo("Info", "Primero planifica una trayectoria.")
            return
        if self.current_evaluation is None:
            self.progress.start()
            self.evaluate_btn.config(state="disabled")
            def run():
                try:
                    ev = evaluate_trajectory(client, self.current_goal, self.current_result["path"], self.current_missing)
                    self.root.after(0, self._on_evaluation_done, ev)
                except Exception as e:
                    self.root.after(0, self._on_evaluation_error, str(e))
            threading.Thread(target=run, daemon=True).start()
        else:
            self._show_window(
                title="Evaluación de la ruta",
                content=self._format_evaluation(self.current_evaluation),
                width=700, height=500
            )

    def _on_evaluation_done(self, ev):
        self.progress.stop()
        self.evaluate_btn.config(state="normal")
        self.current_evaluation = ev
        if ev:
            self._show_window(
                title="Evaluación de la ruta",
                content=self._format_evaluation(ev),
                width=700, height=500
            )
        else:
            messagebox.showwarning("Advertencia", "No se pudo obtener la evaluación.")

    def _on_evaluation_error(self, err):
        self.progress.stop()
        self.evaluate_btn.config(state="normal")
        messagebox.showerror("Error", f"Error al evaluar la ruta:\n{err}")

    def _format_evaluation(self, ev):
        lines = []
        lines.append(f"Puntuación: {ev.get('puntuacion', 'N/A')}/10\n")
        lines.append("Puntos fuertes:")
        for s in ev.get("puntos_fuertes", []):
            lines.append(f"  + {s}")
        lines.append("\nPuntos débiles:")
        for w in ev.get("puntos_debiles", []):
            lines.append(f"  - {w}")
        lines.append("\nHabilidades no cubiertas:")
        for u in ev.get("habilidades_no_cubiertas", []):
            lines.append(f"  ! {u}")
        lines.append(f"\nRecomendación:\n{ev.get('recomendacion_general', '')}")
        return "\n".join(lines)

    # --------------------------------------------------------------
    # Comparison with opposite engine
    # --------------------------------------------------------------
    def compare_alternative(self):
        if not self.current_missing:
            messagebox.showinfo("Info", "No hay habilidades para comparar.")
            return

        # Determine opposite criterion to current engine
        current_criterion = self.current_result.get("criterion", "")
        if "min_time" in current_criterion or "time" in current_criterion:
            opposite_criterion = "cost"
            opposite_name = "mínimo coste"
        else:
            opposite_criterion = "time"
            opposite_name = "mínimo tiempo"

        if self.current_alt_result is None:
            self.progress.start()
            self.compare_btn.config(state="disabled")
            def run():
                try:
                    alt = run_single_objective(
                        graph=graph,
                        missing_skills=self.current_missing,
                        possessed_skills=self.current_base_skills,
                        criterion=opposite_criterion
                    )
                    comp = compare_trajectories(client, self.current_goal, self.current_result, alt)
                    self.root.after(0, self._on_compare_done, alt, comp, opposite_name)
                except Exception as e:
                    self.root.after(0, self._on_compare_error, str(e))
            threading.Thread(target=run, daemon=True).start()
        else:
            self._show_window(
                title=f"Comparación con motor de {self._get_opposite_name()}",
                content=self._format_comparison(self.current_alt_result, self.current_comparison),
                width=800, height=600
            )

    def _get_opposite_name(self):
        crit = self.current_result.get("criterion", "")
        return "mínimo tiempo" if "min_time" in crit or "time" in crit else "mínimo coste"

    def _on_compare_done(self, alt, comp, opposite_name):
        self.progress.stop()
        self.compare_btn.config(state="normal")
        self.current_alt_result = alt
        self.current_comparison = comp
        self._show_window(
            title=f"Comparación con motor de {opposite_name}",
            content=self._format_comparison(alt, comp),
            width=800, height=600
        )

    def _on_compare_error(self, err):
        self.progress.stop()
        self.compare_btn.config(state="normal")
        messagebox.showerror("Error", f"Error al comparar trayectorias:\n{err}")

    def _format_comparison(self, alt, comp):
        lines = []
        lines.append(f"📊 Comparación entre la ruta actual y la alternativa ({self._get_opposite_name()})\n")
        lines.append(f"Actual: {len(self.current_result['path'])} cursos, {self.current_result['total_weeks']} semanas, {self.current_result['total_cost']}€")
        lines.append(f"Alternativa: {len(alt['path'])} cursos, {alt['total_weeks']} semanas, {alt['total_cost']}€\n")
        if comp:
            lines.append(f"🤖 LLM recomienda: {comp.get('recomendada', '')}")
            lines.append(f"Razón: {comp.get('razon', '')}\n")
            for point in comp.get("tabla_comparativa", []):
                lines.append(f"• {point}")
        else:
            lines.append("No se pudo obtener análisis comparativo.")
        return "\n".join(lines)
    
    def _auto_relaxation(self, result):
        """Called automatically when planning fails."""
        warnings = result.get("warnings", [])
        if not warnings and result.get("path"):
            return  # not a real failure
        
        def run():
            try:
                advice = suggest_relaxation(
                    client,
                    self.current_goal,
                    self.build_constraints(),
                    warnings
                )
                self.root.after(0, self._on_auto_relaxation_done, advice)
            except Exception as e:
                self.root.after(0, self._on_auto_relaxation_error, str(e))
        
        threading.Thread(target=run, daemon=True).start()

    def _on_auto_relaxation_done(self, advice):
        self.progress.stop()
        self.plan_btn.config(state="normal")
        
        # Format text for better display
        formatted = self._format_suggestion(advice)
        
        self._show_window(
            title="Sugerencias para relajar restricciones",
            content=formatted,
            width=750, height=400
        )

    def _format_suggestion(self, text: str) -> str:
        """Improves suggestion formatting: double line breaks, bullet indentation."""
        lines = text.split('\n')
        formatted_lines = []
        for line in lines:
            line = line.strip()
            if not line:
                formatted_lines.append('')  # empty line
            elif line.startswith('-'):
                # Add indentation and bullet
                formatted_lines.append('  • ' + line[1:].strip())
            else:
                formatted_lines.append(line)
        # Join with double line breaks to separate paragraphs
        result = '\n\n'.join(formatted_lines)
        return result

    def _on_auto_relaxation_error(self, err):
        self.progress.stop()
        self.plan_btn.config(state="normal")
        messagebox.showerror("Error", f"No se pudo obtener sugerencia: {err}")

    #Simulations
    def simulate_discrete_events(self):
        if not self.current_result or not self.current_result["path"]:
            messagebox.showinfo("Info", "Primero planifica una trayectoria.")
            return

        # Extract required data
        trajectory = self.current_result["path"]
        target_skills = self.current_missing if self.current_missing else []
        possessed_skills = self.current_base_skills if self.current_base_skills else []

        # Show progress window
        self.progress.start()
        self.simulate_des_btn.config(state="disabled")

        def run():
            try:
                # Run discrete event simulation
                result = run_discrete_event_simulation(
                    trajectory=trajectory,
                    target_skills=target_skills,
                    possessed_skills=possessed_skills,
                    max_weeks=200.0,     # default value, could be taken from constraints
                    seed=42
                )
                self.root.after(0, self._show_discrete_event_result, result)
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", f"Simulación fallida:\n{e}"))
            finally:
                self.progress.stop()
                self.root.after(0, lambda: self.simulate_des_btn.config(state="normal"))

        threading.Thread(target=run, daemon=True).start()

    def _show_discrete_event_result(self, result):
        """Displays the simulation result in a table window."""
        from tkinter import ttk

        win = tk.Toplevel(self.root)
        win.title("Simulación de Eventos Discretos - Timeline")
        win.geometry("1100x700")
        win.transient(self.root)
        win.grab_set()
        win.lift()

        # Summary frame (top)
        summary_frame = ttk.LabelFrame(win, text="Resumen de la simulación")
        summary_frame.pack(fill="x", padx=10, pady=5)
        status = "✓ META ALCANZADA" if result["success"] else "✗ TIEMPO AGOTADO"
        ttk.Label(summary_frame, text=f"Estado: {status}").pack(anchor="w", padx=5, pady=2)
        ttk.Label(summary_frame, text=f"Semanas totales: {result['total_weeks']}").pack(anchor="w", padx=5)
        ttk.Label(summary_frame, text=f"Cursos completados: {len(result['completed_courses'])}").pack(anchor="w", padx=5)
        ttk.Label(summary_frame, text=f"Abandonos: {result['n_dropouts']}").pack(anchor="w", padx=5)

        # Event table
        table_frame = ttk.LabelFrame(win, text="Timeline de eventos")
        table_frame.pack(fill="both", expand=True, padx=10, pady=5)

        # Treeview with columns
        columns = ("semana", "evento", "curso", "prerrequisitos", "habilidades_aportadas")
        tree = ttk.Treeview(table_frame, columns=columns, show="headings", height=20)
        tree.heading("semana", text="Semana")
        tree.heading("evento", text="Evento")
        tree.heading("curso", text="Curso")
        tree.heading("prerrequisitos", text="Prerrequisitos")
        tree.heading("habilidades_aportadas", text="Habilidades aportadas")
        tree.column("semana", width=80, anchor="center")
        tree.column("evento", width=120)
        tree.column("curso", width=250)
        tree.column("prerrequisitos", width=250)
        tree.column("habilidades_aportadas", width=300)

        scrollbar = ttk.Scrollbar(table_frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Insert events
        for ev in result["event_log"]:
            week = ev.get("week", "")
            event = ev.get("event", "")
            course = ev.get("course_name", "")
            prereqs = ", ".join(ev.get("prerequisites", [])) if ev.get("prerequisites") else ""
            skills = ", ".join(ev.get("skills_granted", [])) if ev.get("skills_granted") else ""
            tree.insert("", "end", values=(week, event, course, prereqs, skills))

        # Close button
        btn = ttk.Button(win, text="Cerrar", command=win.destroy)
        btn.pack(pady=10)


    def simulate_reactive_agents(self):
        

        # For agent simulation you need the full course catalog
        # (already available as courses_catalog, global in module)
        # But careful: courses_catalog is outside the class; you can pass it as argument or access globally.
        # Recommendation: save courses_catalog as class attribute in __init__
        # For simplicity, assume self.courses_catalog is already set.

        self.progress.start()
        self.simulate_agents_btn.config(state="disabled")

        def run():
            try:
                # Run reactive agents simulation
                result = run_reactive_agents(
                    available_courses=courses_catalog,   # global variable
                    n_agents=100,
                    n_weeks=52,
                    seed=42
                )
                self.root.after(0, self._show_reactive_agents_result, result)
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", f"Simulación de agentes fallida:\n{e}"))
            finally:
                self.progress.stop()
                self.root.after(0, lambda: self.simulate_agents_btn.config(state="normal"))

        threading.Thread(target=run, daemon=True).start()

    def _show_reactive_agents_result(self, result):
        lines = []
        lines.append(f"Tasa de éxito (meta completa): {result['success_rate']*100:.1f}%")
        lines.append("\nEstadísticas por perfil:")
        for pf, stats in result["profile_stats"].items():
            lines.append(f" Agente {pf}: {stats['success_rate']*100:.0f}% éxito ({stats['success']}/{stats['total']})")
        lines.append("\nTop 5 habilidades más demandadas:")
        for skill, frac in result["most_competitive_skills"]:
            lines.append(f"  {frac*100:.0f}%  {skill}")
        lines.append("\nTop 5 habilidades menos demandadas:")
        for skill, frac in result["least_competitive_skills"]:
            lines.append(f"  {frac*100:.0f}%  {skill}")
        if result["market_insights"]:
            lines.append("\nPerspectivas del mercado:")
            for ins in result["market_insights"]:
                lines.append(f"  → {ins}")
        content = "\n".join(lines)
        self._show_window("Simulación de Agentes Reactivos", content, width=800, height=600)

    # --------------------------------------------------------------
    # Generic popup window
    # --------------------------------------------------------------
    def _show_window(self, title, content, width=700, height=500):
        """Creates a popup window with formatted text (supports **bold** and bullet lists)."""
        win = tk.Toplevel(self.root)
        win.title(title)
        win.geometry(f"{width}x{height}")
        win.transient(self.root)
        win.grab_set()
        win.lift()

        text_widget = tk.Text(win, wrap=tk.WORD, font=("Segoe UI", 10), relief=tk.FLAT, borderwidth=0)
        text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        scrollbar = ttk.Scrollbar(win, orient="vertical", command=text_widget.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        text_widget.configure(yscrollcommand=scrollbar.set)

        # Configure style tags
        text_widget.tag_configure("bold", font=("Segoe UI", 10, "bold"))
        text_widget.tag_configure("bullet", lmargin1=20, lmargin2=35)
        text_widget.tag_configure("heading", font=("Segoe UI", 11, "bold"), spacing3=5)

        # Insert formatted content
        self._insert_formatted_text(text_widget, content)
        text_widget.configure(state="disabled")

        # Close button
        btn = ttk.Button(win, text="Cerrar", command=win.destroy)
        btn.pack(pady=10)

    def _insert_formatted_text(self, text_widget, raw_text):
        """
        Inserts text into the Text widget, interpreting:
        - **bold** for text between double asterisks
        - Lists starting with '- ' or '• ' as bullets
        - Preserves line breaks
        """
        # First, replace markdown bullets '•' with '- ' for processing
        raw_text = raw_text.replace('•', '-')
        lines = raw_text.split('\n')
        in_bold = False
        for line in lines:
            # Process bullets
            if line.strip().startswith('-'):
                # Remove the leading dash
                content = line.strip()[1:].lstrip()
                text_widget.insert(tk.END, "  • ", "bullet")
                # Process bold inside the content
                self._insert_with_bold(text_widget, content)
                text_widget.insert(tk.END, "\n")
            else:
                # Normal line
                self._insert_with_bold(text_widget, line)
                text_widget.insert(tk.END, "\n")
            # Add extra space after empty paragraphs (double newline)
            if line.strip() == '':
                text_widget.insert(tk.END, "\n")

    def _insert_with_bold(self, text_widget, text):
        """Inserts text interpreting **bold** within the line."""
        parts = text.split('**')
        for i, part in enumerate(parts):
            if i % 2 == 1:  # odd parts are between ** ** → bold
                text_widget.insert(tk.END, part, "bold")
            else:
                text_widget.insert(tk.END, part)

    # --------------------------------------------------------------
    # Utilities
    # --------------------------------------------------------------
    def show_error(self, msg):
        self.progress.stop()
        self.plan_btn.config(state="normal")
        messagebox.showerror("Error", msg)

    def show_info(self, msg):
        messagebox.showinfo("Info", msg)


if __name__ == "__main__":
    if USE_THEMES:
        root = ThemedTk(theme="arc")
    else:
        root = tk.Tk()
    app = ModernPlannerGUI(root)
    root.mainloop()