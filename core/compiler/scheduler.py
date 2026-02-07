"""Constraint-Aware Topological Scheduler for WorkflowIR.

This module provides DAG scheduling capabilities that respect task dependencies,
resource constraints, and execution priorities. It transforms a WorkflowIR into
an optimized execution schedule.

Example:
    >>> from core.compiler.ir import WorkflowIR, Task
    >>> from core.compiler.scheduler import ConstraintAwareScheduler, Schedule
    >>>
    >>> ir = WorkflowIR(
    ...     intent="Process data pipeline",
    ...     tasks=[
    ...         Task(id="extract", description="Extract", agent_type="data_fetcher"),
    ...         Task(id="transform", description="Transform", agent_type="transform_agent",
    ...              depends_on=["extract"]),
    ...         Task(id="load", description="Load", agent_type="data_fetcher",
    ...              depends_on=["transform"]),
    ...     ],
    ...     failure_policy={"retry": True, "max_attempts": 3},
    ...     metadata={"version": "v1"}
    ... )
    >>>
    >>> scheduler = ConstraintAwareScheduler(max_parallelism=2)
    >>> schedule = scheduler.schedule(ir)
    >>> print(schedule.critical_path)
    ['extract', 'transform', 'load']
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from core.compiler.ir import Task, WorkflowIR


@dataclass
class ScheduledTask:
    """A task with scheduling metadata.

    Attributes:
        task: The original Task from the IR.
        start_time: Estimated start time (relative units).
        end_time: Estimated end time (relative units).
        parallelism_slot: Which parallel slot this task occupies.
        depth: Distance from entry nodes (0 = entry).
    """

    task: Task
    start_time: int = 0
    end_time: int = 1
    parallelism_slot: int = 0
    depth: int = 0


@dataclass
class Schedule:
    """An optimized execution schedule for a WorkflowIR.

    Attributes:
        order: Topologically sorted task execution order.
        scheduled_tasks: Tasks with timing and slot information.
        parallelism: Maximum concurrent tasks allowed.
        critical_path: Longest dependency chain (bottleneck).
        estimated_duration: Estimated total execution time.
        stages: Tasks grouped by execution wave (parallelizable).
    """

    order: list[str]
    scheduled_tasks: dict[str, ScheduledTask]
    parallelism: int
    critical_path: list[str]
    estimated_duration: int
    stages: list[list[str]] = field(default_factory=list)

    def get_task_schedule(self, task_id: str) -> ScheduledTask | None:
        """Get scheduling info for a specific task."""
        return self.scheduled_tasks.get(task_id)

    def get_ready_tasks(self, completed: set[str]) -> list[str]:
        """Get tasks that are ready to execute given completed tasks."""
        ready = []
        for task_id in self.order:
            if task_id in completed:
                continue
            scheduled = self.scheduled_tasks[task_id]
            if all(dep in completed for dep in scheduled.task.depends_on):
                ready.append(task_id)
        return ready

    def get_parallel_batch(self, completed: set[str]) -> list[str]:
        """Get the next batch of tasks that can run in parallel.

        Returns tasks that:
        1. Are ready (all dependencies completed)
        2. Fit within parallelism limit
        """
        ready = self.get_ready_tasks(completed)
        running = [
            tid for tid in self.order
            if tid not in completed and tid not in ready
        ]
        slots_available = self.parallelism - len(running)
        return ready[:slots_available]


class ConstraintAwareScheduler:
    """Schedules WorkflowIR tasks respecting dependencies and constraints.

    This scheduler:
    1. Validates the DAG (no cycles)
    2. Computes topological order
    3. Calculates critical path (longest dependency chain)
    4. Groups tasks into parallelizable stages
    5. Assigns timing estimates

    Example:
        >>> scheduler = ConstraintAwareScheduler(max_parallelism=4)
        >>> schedule = scheduler.schedule(ir)
        >>>
        >>> # Execute wave by wave
        >>> completed = set()
        >>> while len(completed) < len(schedule.order):
        ...     batch = schedule.get_parallel_batch(completed)
        ...     # Execute batch in parallel
        ...     results = await execute_parallel(batch)
        ...     completed.update(batch)
    """

    def __init__(
        self,
        max_parallelism: int = 4,
        default_task_duration: int = 1,
    ):
        """Initialize the scheduler.

        Args:
            max_parallelism: Maximum concurrent tasks allowed.
            default_task_duration: Default duration estimate for tasks.
        """
        self.max_parallelism = max_parallelism
        self.default_duration = default_task_duration

    def schedule(self, ir: WorkflowIR) -> Schedule:
        """Create an optimized schedule for the given WorkflowIR.

        Args:
            ir: The workflow intermediate representation.

        Returns:
            Schedule with ordering, timing, and parallelism info.

        Raises:
            ValueError: If the task dependencies contain a cycle.
        """
        # Build dependency graph
        graph = self._build_graph(ir)

        # Validate DAG
        if self._has_cycle(graph):
            raise ValueError(
                f"Cycle detected in task dependencies for intent: {ir.intent}"
            )

        # Compute topological order (Kahn's algorithm)
        order = self._topological_sort(graph)

        # Compute depths (distance from entry nodes)
        depths = self._compute_depths(graph, order)

        # Compute critical path
        critical_path = self._compute_critical_path(graph, order)

        # Group into parallelizable stages
        stages = self._compute_stages(graph, order, depths)

        # Assign parallelism slots and timing
        scheduled_tasks = self._assign_schedule(ir, order, depths, stages)

        # Estimate total duration
        estimated_duration = max(
            st.end_time for st in scheduled_tasks.values()
        ) if scheduled_tasks else 0

        return Schedule(
            order=order,
            scheduled_tasks=scheduled_tasks,
            parallelism=self.max_parallelism,
            critical_path=critical_path,
            estimated_duration=estimated_duration,
            stages=stages,
        )

    def _build_graph(self, ir: WorkflowIR) -> dict[str, list[str]]:
        """Build adjacency list from tasks.

        Returns dict mapping task_id -> list of dependent task_ids.
        """
        graph: dict[str, list[str]] = defaultdict(list)
        task_ids = {task.id for task in ir.tasks}

        for task in ir.tasks:
            # Ensure task is in graph
            if task.id not in graph:
                graph[task.id] = []
            # Add dependencies
            for dep in task.depends_on:
                if dep in task_ids:
                    graph[dep].append(task.id)

        return dict(graph)

    def _has_cycle(self, graph: dict[str, list[str]]) -> bool:
        """Detect cycles using DFS."""
        WHITE, GRAY, BLACK = 0, 1, 2
        color = {node: WHITE for node in graph}

        def dfs(node: str) -> bool:
            color[node] = GRAY
            for neighbor in graph.get(node, []):
                if color[neighbor] == GRAY:
                    return True  # Back edge = cycle
                if color[neighbor] == WHITE and dfs(neighbor):
                    return True
            color[node] = BLACK
            return False

        for node in graph:
            if color[node] == WHITE:
                if dfs(node):
                    return True
        return False

    def _topological_sort(self, graph: dict[str, list[str]]) -> list[str]:
        """Kahn's algorithm for topological sort."""
        # Compute in-degrees
        in_degree = {node: 0 for node in graph}
        for node in graph:
            for neighbor in graph[node]:
                in_degree[neighbor] += 1

        # Start with nodes that have no dependencies
        queue = [node for node in graph if in_degree[node] == 0]
        result = []

        while queue:
            # Sort for deterministic output
            queue.sort()
            node = queue.pop(0)
            result.append(node)

            for neighbor in graph.get(node, []):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        return result

    def _compute_depths(
        self, graph: dict[str, list[str]], order: list[str]
    ) -> dict[str, int]:
        """Compute the depth (distance from entry) for each task.

        Entry nodes (no dependencies) have depth 0.
        """
        depths: dict[str, int] = {}

        for task_id in order:
            # Find max depth of dependencies
            max_dep_depth = -1
            for node, dependents in graph.items():
                if task_id in dependents:
                    max_dep_depth = max(max_dep_depth, depths.get(node, 0))

            depths[task_id] = max_dep_depth + 1 if max_dep_depth >= 0 else 0

        return depths

    def _compute_critical_path(self, graph: dict[str, list[str]], order: list[str]) -> list[str]:
        """Find the longest path through the DAG.

        This represents the minimum execution time (bottleneck).
        """
        # Compute longest path to each node
        longest_path: dict[str, list[str]] = {node: [node] for node in graph}

        for task_id in order:
            # Find the longest path ending at this task
            for node, dependents in graph.items():
                if task_id in dependents:
                    candidate = longest_path[node] + [task_id]
                    if len(candidate) > len(longest_path[task_id]):
                        longest_path[task_id] = candidate

        # Return the overall longest path
        return max(longest_path.values(), key=len) if longest_path else []

    def _compute_stages(
        self,
        graph: dict[str, list[str]],
        order: list[str],
        depths: dict[str, int],
    ) -> list[list[str]]:
        """Group tasks into stages where each stage can run in parallel.

        A stage consists of tasks at the same depth level.
        """
        if not order:
            return []

        max_depth = max(depths.values()) if depths else 0
        stages: list[list[str]] = [[] for _ in range(max_depth + 1)]

        for task_id in order:
            depth = depths[task_id]
            stages[depth].append(task_id)

        return stages

    def _assign_schedule(
        self,
        ir: WorkflowIR,
        order: list[str],
        depths: dict[str, int],
        stages: list[list[str]],
    ) -> dict[str, ScheduledTask]:
        """Create ScheduledTask objects with timing and slot assignments."""
        task_map = {task.id: task for task in ir.tasks}
        scheduled: dict[str, ScheduledTask] = {}

        for task_id in order:
            task = task_map[task_id]
            depth = depths[task_id]

            # Calculate start time based on dependencies
            start_time = 0
            for dep in task.depends_on:
                if dep in scheduled:
                    start_time = max(start_time, scheduled[dep].end_time)

            # If no dependencies at this depth level started yet, use depth
            if start_time == 0:
                start_time = depth * self.default_duration

            # Assign parallelism slot
            stage = stages[depth]
            slot = stage.index(task_id) % self.max_parallelism

            scheduled[task_id] = ScheduledTask(
                task=task,
                start_time=start_time,
                end_time=start_time + self.default_duration,
                parallelism_slot=slot,
                depth=depth,
            )

        return scheduled


@dataclass
class ResourceConstraint:
    """Resource constraint for scheduling.

    Future enhancement: constrain by CPU, memory, API rate limits, etc.

    Attributes:
        resource_type: Type of resource (e.g., "api_calls", "memory_mb").
        limit: Maximum amount available.
        task_requirements: Map of task_id -> required amount.
    """

    resource_type: str
    limit: int
    task_requirements: dict[str, int] = field(default_factory=dict)


class ResourceConstrainedScheduler(ConstraintAwareScheduler):
    """Scheduler that respects resource constraints (future work).

    This is a placeholder for future enhancement where tasks have
    resource requirements and the scheduler must ensure constraints
    are not violated.
    """

    def __init__(
        self,
        max_parallelism: int = 4,
        constraints: list[ResourceConstraint] | None = None,
    ):
        super().__init__(max_parallelism)
        self.constraints = constraints or []

    def _can_schedule(
        self, task_id: str, scheduled: dict[str, ScheduledTask]
    ) -> bool:
        """Check if scheduling this task would violate constraints."""
        # TODO: Implement resource constraint checking
        return True
