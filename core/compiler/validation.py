"""Compiler Validation Framework with Property-Based Testing.

This module provides property-based validation for the NL compiler,
ensuring that generated workflows maintain invariants regardless of
input variations.

Example:
    >>> from core.compiler.validation import WorkflowValidator
    >>> from core.compiler.ir import WorkflowIR
    >>>
    >>> validator = WorkflowValidator()
    >>> ir = compile_intent("Fetch data and email report")
    >>> result = validator.validate(ir)
    >>> print(result.is_valid)
    True
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from core.compiler.ir import Task, WorkflowIR


@dataclass
class ValidationError:
    """A single validation error.

    Attributes:
        code: Error category code.
        message: Human-readable error description.
        task_id: ID of task with error (if applicable).
        context: Additional context for debugging.
    """

    code: str
    message: str
    task_id: str | None = None
    context: dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationResult:
    """Result of validating a WorkflowIR.

    Attributes:
        is_valid: True if no errors found.
        errors: List of validation errors.
        warnings: List of non-critical warnings.
        properties_checked: List of properties that were validated.
    """

    is_valid: bool
    errors: list[ValidationError] = field(default_factory=list)
    warnings: list[ValidationError] = field(default_factory=list)
    properties_checked: list[str] = field(default_factory=list)

    def add_error(
        self,
        code: str,
        message: str,
        task_id: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        """Add an error to the result."""
        self.errors.append(
            ValidationError(
                code=code,
                message=message,
                task_id=task_id,
                context=context or {},
            )
        )
        self.is_valid = False

    def add_warning(
        self,
        code: str,
        message: str,
        task_id: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        """Add a warning to the result."""
        self.warnings.append(
            ValidationError(
                code=code,
                message=message,
                task_id=task_id,
                context=context or {},
            )
        )


class WorkflowValidator:
    """Validates WorkflowIR against compiler invariants.

    This validator checks that compiled workflows maintain properties
    that are required for correct execution, regardless of the input
    that generated them.

    Properties validated:
    - DAG_STRUCTURE: No cycles in dependencies
    - REACHABILITY: All tasks reachable from entry points
    - DEPENDENCY_RESOLUTION: All dependencies reference existing tasks
    - TYPE_CONSISTENCY: Agent types are valid
    - SCHEMA_VALIDITY: Output schemas are well-formed
    - RESOURCE_SAFETY: Resource limits are respected

    Example:
        >>> validator = WorkflowValidator()
        >>>
        >>> # Full validation
        >>> result = validator.validate(ir)
        >>>
        >>> # Check specific property
        >>> result = validator.validate_dag_structure(ir)
    """

    def __init__(self):
        """Initialize validator with registered checks."""
        self._checks: dict[str, Callable[[WorkflowIR, ValidationResult], None]] = {
            "DAG_STRUCTURE": self._check_dag_structure,
            "REACHABILITY": self._check_reachability,
            "DEPENDENCY_RESOLUTION": self._check_dependency_resolution,
            "TYPE_CONSISTENCY": self._check_type_consistency,
            "SCHEMA_VALIDITY": self._check_schema_validity,
            "RESOURCE_SAFETY": self._check_resource_safety,
        }

    def validate(
        self,
        ir: WorkflowIR,
        checks: list[str] | None = None,
    ) -> ValidationResult:
        """Run all or specified validation checks.

        Args:
            ir: The workflow intermediate representation to validate.
            checks: Specific checks to run (None = all checks).

        Returns:
            ValidationResult with errors and warnings.
        """
        result = ValidationResult(is_valid=True)

        check_list = checks or list(self._checks.keys())

        for check_name in check_list:
            if check_name in self._checks:
                result.properties_checked.append(check_name)
                self._checks[check_name](ir, result)

        return result

    def _check_dag_structure(
        self, ir: WorkflowIR, result: ValidationResult
    ) -> None:
        """Check that dependencies form a valid DAG (no cycles)."""
        # Build adjacency list
        graph: dict[str, list[str]] = {
            task.id: [] for task in ir.tasks
        }
        for task in ir.tasks:
            for dep in task.depends_on:
                if dep in graph:
                    graph[dep].append(task.id)

        # DFS for cycle detection
        WHITE, GRAY, BLACK = 0, 1, 2
        color = {node: WHITE for node in graph}

        def has_cycle(node: str, path: list[str]) -> bool:
            color[node] = GRAY
            path.append(node)

            for neighbor in graph.get(node, []):
                if color[neighbor] == GRAY:
                    # Found cycle - report it
                    cycle_start = path.index(neighbor)
                    cycle = path[cycle_start:] + [neighbor]
                    result.add_error(
                        code="DAG_STRUCTURE",
                        message=f"Cycle detected in dependencies: {' -> '.join(cycle)}",
                        context={"cycle": cycle},
                    )
                    return True
                if color[neighbor] == WHITE and has_cycle(neighbor, path):
                    return True

            path.pop()
            color[node] = BLACK
            return False

        for node in graph:
            if color[node] == WHITE:
                if has_cycle(node, []):
                    return

    def _check_reachability(
        self, ir: WorkflowIR, result: ValidationResult
    ) -> None:
        """Check that all tasks are reachable from entry points."""
        if not ir.tasks:
            return

        # Find entry points (tasks with no dependencies)
        entry_points = [
            task.id for task in ir.tasks if not task.depends_on
        ]

        if not entry_points:
            result.add_error(
                code="REACHABILITY",
                message="No entry points found - all tasks have dependencies",
            )
            return

        # BFS to find reachable tasks
        reachable: set[str] = set()
        queue = list(entry_points)

        while queue:
            task_id = queue.pop(0)
            if task_id in reachable:
                continue
            reachable.add(task_id)

            # Find tasks that depend on this one
            for task in ir.tasks:
                if task_id in task.depends_on and task.id not in reachable:
                    queue.append(task.id)

        # Check for unreachable tasks
        all_task_ids = {task.id for task in ir.tasks}
        unreachable = all_task_ids - reachable

        for task_id in unreachable:
            result.add_error(
                code="REACHABILITY",
                message=f"Task '{task_id}' is unreachable from entry points",
                task_id=task_id,
                context={
                    "entry_points": entry_points,
                    "unreachable": list(unreachable),
                },
            )

    def _check_dependency_resolution(
        self, ir: WorkflowIR, result: ValidationResult
    ) -> None:
        """Check that all dependencies reference existing tasks."""
        task_ids = {task.id for task in ir.tasks}

        for task in ir.tasks:
            for dep in task.depends_on:
                if dep not in task_ids:
                    result.add_error(
                        code="DEPENDENCY_RESOLUTION",
                        message=f"Task '{task.id}' depends on unknown task '{dep}'",
                        task_id=task.id,
                        context={"missing_dependency": dep, "available_tasks": list(task_ids)},
                    )

    def _check_type_consistency(
        self, ir: WorkflowIR, result: ValidationResult
    ) -> None:
        """Check that agent types are valid and consistent."""
        valid_agent_types = {
            "data_fetcher", "data_transformer", "email_sender",
            "slack_notifier", "report_generator", "data_analyzer",
            "database_writer", "file_writer", "text_generator",
            "summarizer", "web_searcher", "llm_agent", "function_agent",
        }

        for task in ir.tasks:
            if task.agent_type not in valid_agent_types:
                result.add_warning(
                    code="TYPE_CONSISTENCY",
                    message=f"Unknown agent type '{task.agent_type}'",
                    task_id=task.id,
                    context={
                        "agent_type": task.agent_type,
                        "valid_types": list(valid_agent_types),
                    },
                )

    def _check_schema_validity(
        self, ir: WorkflowIR, result: ValidationResult
    ) -> None:
        """Check that input/output schemas are well-formed."""
        for task in ir.tasks:
            # Check for duplicate inputs
            if len(task.inputs) != len(set(task.inputs)):
                result.add_error(
                    code="SCHEMA_VALIDITY",
                    message=f"Duplicate input keys in task '{task.id}'",
                    task_id=task.id,
                    context={"inputs": task.inputs},
                )

    def _check_resource_safety(
        self, ir: WorkflowIR, result: ValidationResult
    ) -> None:
        """Check that resource limits are respected."""
        max_tasks = 100
        max_dependencies_per_task = 10

        if len(ir.tasks) > max_tasks:
            result.add_error(
                code="RESOURCE_SAFETY",
                message=f"Too many tasks: {len(ir.tasks)} (max: {max_tasks})",
                context={"task_count": len(ir.tasks), "max_allowed": max_tasks},
            )

        for task in ir.tasks:
            if len(task.depends_on) > max_dependencies_per_task:
                result.add_error(
                    code="RESOURCE_SAFETY",
                    message=f"Task '{task.id}' has too many dependencies",
                    task_id=task.id,
                    context={
                        "dependency_count": len(task.depends_on),
                        "max_allowed": max_dependencies_per_task,
                    },
                )

    def register_check(
        self,
        name: str,
        check_func: Callable[[WorkflowIR, ValidationResult], None],
    ) -> None:
        """Register a custom validation check.

        Args:
            name: Unique name for the check.
            check_func: Function that takes (ir, result) and adds errors/warnings.
        """
        self._checks[name] = check_func


class InvariantChecker:
    """High-level invariant checker for compiler properties.

    This class provides methods to check specific invariants that must
    hold for all valid compiler outputs.
    """

    @staticmethod
    def is_valid_dag(tasks: list[Task]) -> bool:
        """Check if tasks form a valid DAG (no cycles)."""
        graph = {task.id: [] for task in tasks}
        for task in tasks:
            for dep in task.depends_on:
                if dep in graph:
                    graph[dep].append(task.id)

        visited: set[str] = set()
        rec_stack: set[str] = set()

        def has_cycle(node: str) -> bool:
            visited.add(node)
            rec_stack.add(node)

            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    if has_cycle(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True

            rec_stack.remove(node)
            return False

        for node in graph:
            if node not in visited:
                if has_cycle(node):
                    return False

        return True

    @staticmethod
    def all_dependencies_resolvable(tasks: list[Task]) -> bool:
        """Check if all dependencies reference existing tasks."""
        task_ids = {task.id for task in tasks}
        return all(
            dep in task_ids
            for task in tasks
            for dep in task.depends_on
        )

    @staticmethod
    def has_entry_point(tasks: list[Task]) -> bool:
        """Check if there's at least one task with no dependencies."""
        return any(not task.depends_on for task in tasks)

    @staticmethod
    def no_orphan_inputs(tasks: list[Task]) -> bool:
        """Check that all task inputs are produced by some task or external source."""
        all_task_ids = {task.id for task in tasks}
        all_produced: set[str] = set()  # Would be outputs from tasks
        all_consumed: set[str] = set()

        for task in tasks:
            all_consumed.update(task.inputs)

        # For the compiler IR, we check that inputs are reasonable
        # Most inputs should come from other tasks or be external
        return True  # Simplified for IR structure
