"""Tests for the Constraint-Aware Topological Scheduler.

These tests validate DAG scheduling, topological ordering, critical path
computation, and parallel execution wave grouping.
"""

from __future__ import annotations

import pytest

from core.compiler.ir import Task, WorkflowIR
from core.compiler.scheduler import (
    ConstraintAwareScheduler,
    ResourceConstraint,
    ResourceConstrainedScheduler,
    Schedule,
    ScheduledTask,
)


class TestConstraintAwareScheduler:
    """Tests for the ConstraintAwareScheduler."""

    def test_schedule_linear_pipeline(self):
        """Should handle simple linear dependency chain."""
        ir = WorkflowIR(
            intent="Linear pipeline",
            tasks=[
                Task(id="a", description="Task A", agent_type="llm_agent"),
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
        scheduler = ConstraintAwareScheduler()

        schedule = scheduler.schedule(ir)

        assert schedule.order == ["a", "b", "c"]
        assert schedule.critical_path == ["a", "b", "c"]
        assert len(schedule.stages) == 3  # Each at different depth

    def test_schedule_parallel_branches(self):
        """Should handle tasks that can run in parallel."""
        ir = WorkflowIR(
            intent="Parallel branches",
            tasks=[
                Task(id="start", description="Start", agent_type="llm_agent"),
                Task(
                    id="branch_a",
                    description="Branch A",
                    agent_type="llm_agent",
                    depends_on=["start"],
                ),
                Task(
                    id="branch_b",
                    description="Branch B",
                    agent_type="llm_agent",
                    depends_on=["start"],
                ),
                Task(
                    id="merge",
                    description="Merge",
                    agent_type="llm_agent",
                    depends_on=["branch_a", "branch_b"],
                ),
            ],
            failure_policy={"retry": True, "max_attempts": 3},
            metadata={"version": "v1"},
        )
        scheduler = ConstraintAwareScheduler()

        schedule = scheduler.schedule(ir)

        # Order should respect dependencies
        assert schedule.order.index("start") < schedule.order.index("branch_a")
        assert schedule.order.index("start") < schedule.order.index("branch_b")
        assert schedule.order.index("branch_a") < schedule.order.index("merge")
        assert schedule.order.index("branch_b") < schedule.order.index("merge")

        # Branch A and B should be at same depth
        assert (
            schedule.scheduled_tasks["branch_a"].depth
            == schedule.scheduled_tasks["branch_b"].depth
        )

    def test_schedule_diamond_pattern(self):
        """Should handle diamond dependency pattern."""
        ir = WorkflowIR(
            intent="Diamond workflow",
            tasks=[
                Task(id="source", description="Source", agent_type="llm_agent"),
                Task(
                    id="left",
                    description="Left branch",
                    agent_type="llm_agent",
                    depends_on=["source"],
                ),
                Task(
                    id="right",
                    description="Right branch",
                    agent_type="llm_agent",
                    depends_on=["source"],
                ),
                Task(
                    id="sink",
                    description="Sink",
                    agent_type="llm_agent",
                    depends_on=["left", "right"],
                ),
            ],
            failure_policy={"retry": True, "max_attempts": 3},
            metadata={"version": "v1"},
        )
        scheduler = ConstraintAwareScheduler()

        schedule = scheduler.schedule(ir)

        assert schedule.order[0] == "source"
        assert schedule.order[-1] == "sink"

        # Critical path goes through one branch
        assert "source" in schedule.critical_path
        assert "sink" in schedule.critical_path
        assert len(schedule.critical_path) == 3

    def test_schedule_empty_workflow(self):
        """Should handle empty task list."""
        ir = WorkflowIR(
            intent="Empty workflow",
            tasks=[],
            failure_policy={"retry": True, "max_attempts": 3},
            metadata={"version": "v1"},
        )
        scheduler = ConstraintAwareScheduler()

        schedule = scheduler.schedule(ir)

        assert schedule.order == []
        assert schedule.critical_path == []
        assert schedule.estimated_duration == 0

    def test_schedule_single_task(self):
        """Should handle single task."""
        ir = WorkflowIR(
            intent="Single task",
            tasks=[Task(id="only", description="Only task", agent_type="llm_agent")],
            failure_policy={"retry": True, "max_attempts": 3},
            metadata={"version": "v1"},
        )
        scheduler = ConstraintAwareScheduler()

        schedule = scheduler.schedule(ir)

        assert schedule.order == ["only"]
        assert schedule.critical_path == ["only"]
        assert schedule.scheduled_tasks["only"].depth == 0

    def test_schedule_detects_cycle(self):
        """Should raise ValueError for cyclic dependencies."""
        ir = WorkflowIR(
            intent="Cyclic workflow",
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
        scheduler = ConstraintAwareScheduler()

        with pytest.raises(ValueError, match="Cycle detected"):
            scheduler.schedule(ir)

    def test_schedule_self_dependency(self):
        """Should detect task depending on itself."""
        ir = WorkflowIR(
            intent="Self-referential",
            tasks=[
                Task(
                    id="self",
                    description="Self task",
                    agent_type="llm_agent",
                    depends_on=["self"],
                ),
            ],
            failure_policy={"retry": True, "max_attempts": 3},
            metadata={"version": "v1"},
        )
        scheduler = ConstraintAwareScheduler()

        with pytest.raises(ValueError, match="Cycle detected"):
            scheduler.schedule(ir)

    def test_schedule_parallelism_limit(self):
        """Should respect max_parallelism constraint."""
        ir = WorkflowIR(
            intent="Many parallel tasks",
            tasks=[
                Task(id="base", description="Base", agent_type="llm_agent"),
                Task(
                    id="p1",
                    description="Parallel 1",
                    agent_type="llm_agent",
                    depends_on=["base"],
                ),
                Task(
                    id="p2",
                    description="Parallel 2",
                    agent_type="llm_agent",
                    depends_on=["base"],
                ),
                Task(
                    id="p3",
                    description="Parallel 3",
                    agent_type="llm_agent",
                    depends_on=["base"],
                ),
                Task(
                    id="p4",
                    description="Parallel 4",
                    agent_type="llm_agent",
                    depends_on=["base"],
                ),
            ],
            failure_policy={"retry": True, "max_attempts": 3},
            metadata={"version": "v1"},
        )
        scheduler = ConstraintAwareScheduler(max_parallelism=2)

        schedule = scheduler.schedule(ir)

        assert schedule.parallelism == 2
        # All parallel tasks should be at depth 1
        parallel_tasks = ["p1", "p2", "p3", "p4"]
        for tid in parallel_tasks:
            assert schedule.scheduled_tasks[tid].depth == 1


class TestScheduleExecutionHelpers:
    """Tests for Schedule execution helper methods."""

    def test_get_ready_tasks(self):
        """Should identify tasks ready to execute."""
        ir = WorkflowIR(
            intent="Test workflow",
            tasks=[
                Task(id="a", description="Task A", agent_type="llm_agent"),
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
                    depends_on=["a"],
                ),
            ],
            failure_policy={"retry": True, "max_attempts": 3},
            metadata={"version": "v1"},
        )
        scheduler = ConstraintAwareScheduler()
        schedule = scheduler.schedule(ir)

        # Initially, only 'a' is ready
        ready = schedule.get_ready_tasks(set())
        assert ready == ["a"]

        # After 'a' completes, 'b' and 'c' are ready
        ready = schedule.get_ready_tasks({"a"})
        assert set(ready) == {"b", "c"}

    def test_get_parallel_batch(self):
        """Should return batch respecting parallelism limit."""
        ir = WorkflowIR(
            intent="Test workflow",
            tasks=[
                Task(id="base", description="Base", agent_type="llm_agent"),
                Task(
                    id="p1",
                    description="Parallel 1",
                    agent_type="llm_agent",
                    depends_on=["base"],
                ),
                Task(
                    id="p2",
                    description="Parallel 2",
                    agent_type="llm_agent",
                    depends_on=["base"],
                ),
                Task(
                    id="p3",
                    description="Parallel 3",
                    agent_type="llm_agent",
                    depends_on=["base"],
                ),
            ],
            failure_policy={"retry": True, "max_attempts": 3},
            metadata={"version": "v1"},
        )
        scheduler = ConstraintAwareScheduler(max_parallelism=2)
        schedule = scheduler.schedule(ir)

        # After base completes, should get only 2 of the 3 parallel tasks
        batch = schedule.get_parallel_batch({"base"})
        assert len(batch) <= 2
        assert set(batch).issubset({"p1", "p2", "p3"})


class TestCriticalPath:
    """Tests for critical path computation."""

    def test_critical_path_linear(self):
        """Critical path in linear chain is the whole chain."""
        ir = WorkflowIR(
            intent="Linear",
            tasks=[
                Task(id="a", description="A", agent_type="llm_agent"),
                Task(
                    id="b", description="B", agent_type="llm_agent", depends_on=["a"]
                ),
                Task(
                    id="c", description="C", agent_type="llm_agent", depends_on=["b"]
                ),
            ],
            failure_policy={"retry": True, "max_attempts": 3},
            metadata={"version": "v1"},
        )
        scheduler = ConstraintAwareScheduler()
        schedule = scheduler.schedule(ir)

        assert schedule.critical_path == ["a", "b", "c"]

    def test_critical_path_with_parallel(self):
        """Critical path should be longest dependency chain."""
        ir = WorkflowIR(
            intent="With parallel",
            tasks=[
                Task(id="start", description="Start", agent_type="llm_agent"),
                Task(
                    id="long_a",
                    description="Long A",
                    agent_type="llm_agent",
                    depends_on=["start"],
                ),
                Task(
                    id="long_b",
                    description="Long B",
                    agent_type="llm_agent",
                    depends_on=["long_a"],
                ),
                Task(
                    id="short",
                    description="Short",
                    agent_type="llm_agent",
                    depends_on=["start"],
                ),
                Task(
                    id="end",
                    description="End",
                    agent_type="llm_agent",
                    depends_on=["long_b", "short"],
                ),
            ],
            failure_policy={"retry": True, "max_attempts": 3},
            metadata={"version": "v1"},
        )
        scheduler = ConstraintAwareScheduler()
        schedule = scheduler.schedule(ir)

        # Critical path goes through the longer branch
        assert "start" in schedule.critical_path
        assert "long_a" in schedule.critical_path
        assert "long_b" in schedule.critical_path
        assert "end" in schedule.critical_path
        assert len(schedule.critical_path) == 4


class TestStages:
    """Tests for parallel stage computation."""

    def test_stages_linear_pipeline(self):
        """Linear pipeline has each task in its own stage."""
        ir = WorkflowIR(
            intent="Linear",
            tasks=[
                Task(id="a", description="A", agent_type="llm_agent"),
                Task(
                    id="b", description="B", agent_type="llm_agent", depends_on=["a"]
                ),
                Task(
                    id="c", description="C", agent_type="llm_agent", depends_on=["b"]
                ),
            ],
            failure_policy={"retry": True, "max_attempts": 3},
            metadata={"version": "v1"},
        )
        scheduler = ConstraintAwareScheduler()
        schedule = scheduler.schedule(ir)

        # Each task at different depth = different stage
        assert len(schedule.stages) == 3
        assert schedule.stages[0] == ["a"]
        assert schedule.stages[1] == ["b"]
        assert schedule.stages[2] == ["c"]

    def test_stages_with_parallelism(self):
        """Parallel tasks should be in same stage."""
        ir = WorkflowIR(
            intent="Parallel",
            tasks=[
                Task(id="base", description="Base", agent_type="llm_agent"),
                Task(
                    id="p1",
                    description="P1",
                    agent_type="llm_agent",
                    depends_on=["base"],
                ),
                Task(
                    id="p2",
                    description="P2",
                    agent_type="llm_agent",
                    depends_on=["base"],
                ),
            ],
            failure_policy={"retry": True, "max_attempts": 3},
            metadata={"version": "v1"},
        )
        scheduler = ConstraintAwareScheduler()
        schedule = scheduler.schedule(ir)

        # p1 and p2 should be in same stage
        assert len(schedule.stages) == 2
        assert set(schedule.stages[1]) == {"p1", "p2"}


class TestScheduledTask:
    """Tests for ScheduledTask dataclass."""

    def test_scheduled_task_creation(self):
        """Should create ScheduledTask with correct fields."""
        task = Task(id="test", description="Test", agent_type="llm_agent")
        scheduled = ScheduledTask(
            task=task,
            start_time=0,
            end_time=1,
            parallelism_slot=0,
            depth=0,
        )

        assert scheduled.task.id == "test"
        assert scheduled.start_time == 0
        assert scheduled.end_time == 1
        assert scheduled.depth == 0


class TestResourceConstrainedScheduler:
    """Tests for ResourceConstrainedScheduler (future work)."""

    def test_resource_scheduler_creation(self):
        """Should create resource-constrained scheduler."""
        constraints = [
            ResourceConstraint(
                resource_type="api_calls",
                limit=10,
                task_requirements={"task1": 5},
            )
        ]
        scheduler = ResourceConstrainedScheduler(
            max_parallelism=2,
            constraints=constraints,
        )

        assert scheduler.max_parallelism == 2
        assert len(scheduler.constraints) == 1

    def test_resource_scheduler_basic_schedule(self):
        """Resource scheduler should work like base scheduler."""
        ir = WorkflowIR(
            intent="Test",
            tasks=[
                Task(id="a", description="A", agent_type="llm_agent"),
                Task(
                    id="b", description="B", agent_type="llm_agent", depends_on=["a"]
                ),
            ],
            failure_policy={"retry": True, "max_attempts": 3},
            metadata={"version": "v1"},
        )
        scheduler = ResourceConstrainedScheduler()
        schedule = scheduler.schedule(ir)

        assert schedule.order == ["a", "b"]
