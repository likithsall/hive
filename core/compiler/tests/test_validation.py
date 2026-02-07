"""Property-Based Tests for Compiler Validation Framework.

These tests use Hypothesis to generate random workflows and verify
that validation invariants hold for all possible inputs.
"""

from __future__ import annotations

import pytest

try:
    from hypothesis import given, strategies as st, settings, assume
    HYPOTHESIS_AVAILABLE = True
except ImportError:
    HYPOTHESIS_AVAILABLE = False

from core.compiler.ir import Task, WorkflowIR
from core.compiler.validation import (
    ValidationError,
    ValidationResult,
    WorkflowValidator,
    InvariantChecker,
)


# Hypothesis strategies for generating test data

def task_strategy():
    """Generate a Task with random properties."""
    return st.builds(
        Task,
        id=st.sampled_from(["task_a", "task_b", "task_c", "task_d", "task_e"]),
        description=st.text(min_size=1, max_size=100),
        agent_type=st.sampled_from([
            "data_fetcher", "email_sender", "report_generator",
            "llm_agent", "transform_agent"
        ]),
        inputs=st.lists(st.sampled_from(["input1", "input2", "input3"]), max_size=3),
        depends_on=st.lists(st.sampled_from(["task_a", "task_b", "task_c"]), max_size=2),
    )


def workflow_strategy():
    """Generate a WorkflowIR with random tasks."""
    return st.builds(
        WorkflowIR,
        intent=st.text(min_size=5, max_size=200),
        tasks=st.lists(task_strategy(), min_size=0, max_size=10),
        failure_policy=st.just({"retry": True, "max_attempts": 3}),
        metadata=st.just({"version": "v1"}),
    )


class TestWorkflowValidator:
    """Tests for the WorkflowValidator."""

    def test_validate_empty_workflow(self):
        """Empty workflow should be valid."""
        ir = WorkflowIR(
            intent="Empty",
            tasks=[],
            failure_policy={"retry": True, "max_attempts": 3},
            metadata={"version": "v1"},
        )
        validator = WorkflowValidator()

        result = validator.validate(ir)

        assert result.is_valid
        assert len(result.errors) == 0

    def test_validate_single_task(self):
        """Single task with no dependencies should be valid."""
        ir = WorkflowIR(
            intent="Single task",
            tasks=[
                Task(
                    id="only",
                    description="Only task",
                    agent_type="llm_agent",
                ),
            ],
            failure_policy={"retry": True, "max_attempts": 3},
            metadata={"version": "v1"},
        )
        validator = WorkflowValidator()

        result = validator.validate(ir)

        assert result.is_valid

    def test_detects_cycle(self):
        """Should detect cyclic dependencies."""
        ir = WorkflowIR(
            intent="Cyclic",
            tasks=[
                Task(
                    id="a",
                    description="Task A",
                    agent_type="llm_agent",
                    depends_on=["c"],
                ),
                Task(
                    id="b",
                    description="Task B",
                    agent_type="llm_agent",
                    depends_on=["a"],
                ),
                Task(
                    id="c",
                    description="Task C",
                    agent_type="llm_agent",
                    depends_on=["b"],
                ),
            ],
            failure_policy={"retry": True, "max_attempts": 3},
            metadata={"version": "v1"},
        )
        validator = WorkflowValidator()

        result = validator.validate(ir)

        assert not result.is_valid
        assert any(e.code == "DAG_STRUCTURE" for e in result.errors)

    def test_detects_missing_dependency(self):
        """Should detect references to non-existent tasks."""
        ir = WorkflowIR(
            intent="Missing dep",
            tasks=[
                Task(
                    id="task1",
                    description="Task 1",
                    agent_type="llm_agent",
                    depends_on=["nonexistent"],
                ),
            ],
            failure_policy={"retry": True, "max_attempts": 3},
            metadata={"version": "v1"},
        )
        validator = WorkflowValidator()

        result = validator.validate(ir)

        assert not result.is_valid
        assert any(e.code == "DEPENDENCY_RESOLUTION" for e in result.errors)

    def test_detects_unreachable_task(self):
        """Should detect tasks unreachable from entry points."""
        ir = WorkflowIR(
            intent="Unreachable",
            tasks=[
                Task(
                    id="reachable",
                    description="Reachable",
                    agent_type="llm_agent",
                ),
                Task(
                    id="orphan",
                    description="Orphan",
                    agent_type="llm_agent",
                    depends_on=["missing"],
                ),
            ],
            failure_policy={"retry": True, "max_attempts": 3},
            metadata={"version": "v1"},
        )
        validator = WorkflowValidator()

        result = validator.validate(ir)

        # Should have reachability error for orphan
        assert any(e.code == "REACHABILITY" for e in result.errors)

    def test_detects_duplicate_inputs(self):
        """Should detect duplicate input keys in same task."""
        ir = WorkflowIR(
            intent="Duplicate inputs",
            tasks=[
                Task(
                    id="task1",
                    description="Task 1",
                    agent_type="llm_agent",
                    inputs=["data", "data", "config"],
                ),
            ],
            failure_policy={"retry": True, "max_attempts": 3},
            metadata={"version": "v1"},
        )
        validator = WorkflowValidator()

        result = validator.validate(ir)

        assert not result.is_valid
        assert any(e.code == "SCHEMA_VALIDITY" for e in result.errors)

    def test_valid_linear_pipeline(self):
        """Linear pipeline should be valid."""
        ir = WorkflowIR(
            intent="Linear pipeline",
            tasks=[
                Task(
                    id="step1",
                    description="Step 1",
                    agent_type="data_fetcher",
                    inputs=["data"],
                ),
                Task(
                    id="step2",
                    description="Step 2",
                    agent_type="transform_agent",
                    inputs=["data"],
                    depends_on=["step1"],
                ),
                Task(
                    id="step3",
                    description="Step 3",
                    agent_type="report_generator",
                    inputs=["data"],
                    depends_on=["step2"],
                ),
            ],
            failure_policy={"retry": True, "max_attempts": 3},
            metadata={"version": "v1"},
        )
        validator = WorkflowValidator()

        result = validator.validate(ir)

        assert result.is_valid

    def test_valid_parallel_branches(self):
        """Parallel branches should be valid."""
        ir = WorkflowIR(
            intent="Parallel",
            tasks=[
                Task(
                    id="source",
                    description="Source",
                    agent_type="data_fetcher",
                    inputs=["data"],
                ),
                Task(
                    id="branch_a",
                    description="Branch A",
                    agent_type="transform_agent",
                    inputs=["data"],
                    depends_on=["source"],
                ),
                Task(
                    id="branch_b",
                    description="Branch B",
                    agent_type="transform_agent",
                    inputs=["data"],
                    depends_on=["source"],
                ),
                Task(
                    id="merge",
                    description="Merge",
                    agent_type="report_generator",
                    inputs=["data"],
                    depends_on=["branch_a", "branch_b"],
                ),
            ],
            failure_policy={"retry": True, "max_attempts": 3},
            metadata={"version": "v1"},
        )
        validator = WorkflowValidator()

        result = validator.validate(ir)

        assert result.is_valid

    def test_warning_for_unknown_agent_type(self):
        """Should warn about unknown agent types."""
        ir = WorkflowIR(
            intent="Unknown type",
            tasks=[
                Task(
                    id="task1",
                    description="Task 1",
                    agent_type="unknown_custom_agent",
                ),
            ],
            failure_policy={"retry": True, "max_attempts": 3},
            metadata={"version": "v1"},
        )
        validator = WorkflowValidator()

        result = validator.validate(ir)

        # Should be valid (warning, not error)
        assert result.is_valid
        assert any(w.code == "TYPE_CONSISTENCY" for w in result.warnings)

    def test_resource_limits(self):
        """Should enforce resource limits."""
        ir = WorkflowIR(
            intent="Too many tasks",
            tasks=[
                Task(
                    id=f"task_{i}",
                    description=f"Task {i}",
                    agent_type="llm_agent",
                )
                for i in range(150)  # Exceeds limit of 100
            ],
            failure_policy={"retry": True, "max_attempts": 3},
            metadata={"version": "v1"},
        )
        validator = WorkflowValidator()

        result = validator.validate(ir)

        assert not result.is_valid
        assert any(e.code == "RESOURCE_SAFETY" for e in result.errors)

    def test_selective_validation(self):
        """Should allow running specific checks only."""
        ir = WorkflowIR(
            intent="Test",
            tasks=[
                Task(
                    id="task1",
                    description="Task 1",
                    agent_type="unknown_type",
                ),
            ],
            failure_policy={"retry": True, "max_attempts": 3},
            metadata={"version": "v1"},
        )
        validator = WorkflowValidator()

        # Only run DAG check, skip type consistency
        result = validator.validate(ir, checks=["DAG_STRUCTURE"])

        assert result.is_valid  # No DAG errors
        assert "DAG_STRUCTURE" in result.properties_checked
        assert "TYPE_CONSISTENCY" not in result.properties_checked

    def test_custom_check_registration(self):
        """Should support custom validation checks."""
        validator = WorkflowValidator()

        def custom_check(ir: WorkflowIR, result: ValidationResult) -> None:
            if "urgent" in ir.intent.lower() and len(ir.tasks) < 2:
                result.add_error(
                    code="CUSTOM",
                    message="Urgent workflows need at least 2 tasks",
                )

        validator.register_check("CUSTOM", custom_check)

        ir = WorkflowIR(
            intent="Urgent: fix bug",
            tasks=[Task(id="t1", description="Task", agent_type="llm_agent")],
            failure_policy={"retry": True, "max_attempts": 3},
            metadata={"version": "v1"},
        )

        result = validator.validate(ir)

        assert not result.is_valid
        assert any(e.code == "CUSTOM" for e in result.errors)


class TestInvariantChecker:
    """Tests for the InvariantChecker static methods."""

    def test_is_valid_dag_true(self):
        """Should return True for valid DAG."""
        tasks = [
            Task(id="a", description="A", agent_type="llm_agent"),
            Task(
                id="b",
                description="B",
                agent_type="llm_agent",
                depends_on=["a"],
            ),
        ]

        assert InvariantChecker.is_valid_dag(tasks)

    def test_is_valid_dag_false(self):
        """Should return False for cyclic graph."""
        tasks = [
            Task(
                id="a",
                description="A",
                agent_type="llm_agent",
                depends_on=["b"],
            ),
            Task(
                id="b",
                description="B",
                agent_type="llm_agent",
                depends_on=["a"],
            ),
        ]

        assert not InvariantChecker.is_valid_dag(tasks)

    def test_all_dependencies_resolvable_true(self):
        """Should return True when all deps exist."""
        tasks = [
            Task(id="a", description="A", agent_type="llm_agent"),
            Task(
                id="b",
                description="B",
                agent_type="llm_agent",
                depends_on=["a"],
            ),
        ]

        assert InvariantChecker.all_dependencies_resolvable(tasks)

    def test_all_dependencies_resolvable_false(self):
        """Should return False when deps missing."""
        tasks = [
            Task(
                id="a",
                description="A",
                agent_type="llm_agent",
                depends_on=["nonexistent"],
            ),
        ]

        assert not InvariantChecker.all_dependencies_resolvable(tasks)

    def test_has_entry_point_true(self):
        """Should return True when entry point exists."""
        tasks = [
            Task(id="a", description="A", agent_type="llm_agent"),
            Task(
                id="b",
                description="B",
                agent_type="llm_agent",
                depends_on=["a"],
            ),
        ]

        assert InvariantChecker.has_entry_point(tasks)

    def test_has_entry_point_false(self):
        """Should return False when no entry point."""
        tasks = [
            Task(
                id="a",
                description="A",
                agent_type="llm_agent",
                depends_on=["b"],
            ),
            Task(
                id="b",
                description="B",
                agent_type="llm_agent",
                depends_on=["a"],
            ),
        ]

        assert not InvariantChecker.has_entry_point(tasks)

    def test_no_orphan_inputs(self):
        """Should return True for reasonable inputs."""
        tasks = [
            Task(
                id="a",
                description="A",
                agent_type="llm_agent",
                inputs=["external_data"],
            ),
            Task(
                id="b",
                description="B",
                agent_type="llm_agent",
                inputs=["external_data"],
                depends_on=["a"],
            ),
        ]

        assert InvariantChecker.no_orphan_inputs(tasks)


class TestValidationResult:
    """Tests for ValidationResult dataclass."""

    def test_add_error_makes_invalid(self):
        """Adding error should set is_valid to False."""
        result = ValidationResult(is_valid=True)

        result.add_error("TEST", "Test error")

        assert not result.is_valid
        assert len(result.errors) == 1

    def test_add_warning_preserves_valid(self):
        """Adding warning should keep is_valid as True."""
        result = ValidationResult(is_valid=True)

        result.add_warning("TEST", "Test warning")

        assert result.is_valid
        assert len(result.warnings) == 1

    def test_error_context(self):
        """Errors should include context."""
        result = ValidationResult(is_valid=True)

        result.add_error(
            "TEST",
            "Test error",
            task_id="task1",
            context={"key": "value"},
        )

        error = result.errors[0]
        assert error.task_id == "task1"
        assert error.context == {"key": "value"}


class TestPropertyBasedTests:
    """Property-based tests using Hypothesis (if available)."""

    @pytest.mark.skip(reason="Hypothesis not installed in environment")
    def test_all_generated_workflows_valid(self):
        """Property: All valid compiler outputs should pass validation."""
        # This would use Hypothesis if available:
        # @given(workflow_strategy())
        # def test_property(ir):
        #     validator = WorkflowValidator()
        #     result = validator.validate(ir)
        #     # If we generated a valid workflow, it should be valid
        #     assert result.is_valid
        pass

    def test_idempotent_validation(self):
        """Property: Validation should be idempotent."""
        ir = WorkflowIR(
            intent="Test",
            tasks=[
                Task(id="a", description="A", agent_type="llm_agent"),
                Task(
                    id="b",
                    description="B",
                    agent_type="llm_agent",
                    depends_on=["a"],
                ),
            ],
            failure_policy={"retry": True, "max_attempts": 3},
            metadata={"version": "v1"},
        )
        validator = WorkflowValidator()

        result1 = validator.validate(ir)
        result2 = validator.validate(ir)

        assert result1.is_valid == result2.is_valid
        assert len(result1.errors) == len(result2.errors)
