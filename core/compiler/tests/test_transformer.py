"""Tests for the IR-to-Plan transformer.

These tests validate that WorkflowIR is correctly transformed into
executable Hive Plans, including dependency validation and agent type
resolution.
"""

from __future__ import annotations

import pytest

from framework.graph.plan import ActionType, Plan

from core.compiler.ir import Task, WorkflowIR
from core.compiler.transformer import (
    AgentTemplate,
    AgentTypeResolver,
    IRToPlanTransformer,
    UnknownAgentTypeError,
    compile_and_transform,
)


class TestAgentTypeResolver:
    """Tests for the AgentTypeResolver."""

    def test_resolve_known_data_fetcher(self):
        """Should resolve data_fetcher to TOOL_USE action."""
        resolver = AgentTypeResolver()

        action = resolver.resolve("data_fetcher")

        assert action.action_type == ActionType.TOOL_USE
        assert action.tool_name == "http_request"

    def test_resolve_known_reporting_agent(self):
        """Should resolve reporting_agent to LLM_CALL action."""
        resolver = AgentTypeResolver()

        action = resolver.resolve("reporting_agent")

        assert action.action_type == ActionType.LLM_CALL
        assert "reporting agent" in (action.system_prompt or "").lower()

    def test_resolve_unknown_agent_defaults_to_llm(self):
        """Unknown agent types should default to LLM_CALL."""
        resolver = AgentTypeResolver()

        action = resolver.resolve("unknown_custom_agent")

        assert action.action_type == ActionType.LLM_CALL
        assert "unknown_custom_agent" in (action.system_prompt or "")

    def test_register_custom_template(self):
        """Should allow registering custom agent templates."""
        resolver = AgentTypeResolver()
        custom_template = AgentTemplate(
            agent_type="custom_api_agent",
            action_type=ActionType.TOOL_USE,
            tool_name="custom_api",
            description="Custom API agent",
        )

        resolver.register_template(custom_template)
        action = resolver.resolve("custom_api_agent")

        assert action.action_type == ActionType.TOOL_USE
        assert action.tool_name == "custom_api"

    def test_list_supported_types(self):
        """Should return sorted list of supported agent types."""
        resolver = AgentTypeResolver()

        types = resolver.list_supported_types()

        assert "data_fetcher" in types
        assert "email_agent" in types
        assert "reporting_agent" in types
        assert types == sorted(types)  # Should be sorted


class TestIRToPlanTransformer:
    """Tests for the IRToPlanTransformer."""

    def test_transform_basic_ir(self):
        """Should transform a simple IR to a valid Plan."""
        ir = WorkflowIR(
            intent="Fetch data and generate report",
            tasks=[
                Task(
                    id="fetch",
                    description="Fetch sales data",
                    agent_type="data_fetcher",
                ),
                Task(
                    id="report",
                    description="Generate report",
                    agent_type="reporting_agent",
                    depends_on=["fetch"],
                ),
            ],
            failure_policy={"retry": True, "max_attempts": 3},
            metadata={"version": "v1"},
        )
        transformer = IRToPlanTransformer()

        plan = transformer.transform(ir)

        assert isinstance(plan, Plan)
        assert plan.goal_id == "compiled_goal"
        assert len(plan.steps) == 2

    def test_transform_preserves_dependencies(self):
        """Should preserve task dependencies in plan steps."""
        ir = WorkflowIR(
            intent="Test workflow",
            tasks=[
                Task(id="step1", description="Step 1", agent_type="llm_agent"),
                Task(
                    id="step2",
                    description="Step 2",
                    agent_type="llm_agent",
                    depends_on=["step1"],
                ),
                Task(
                    id="step3",
                    description="Step 3",
                    agent_type="llm_agent",
                    depends_on=["step1", "step2"],
                ),
            ],
            failure_policy={"retry": True, "max_attempts": 3},
            metadata={"version": "v1"},
        )
        transformer = IRToPlanTransformer()

        plan = transformer.transform(ir)

        step1 = next(s for s in plan.steps if s.id == "step1")
        step2 = next(s for s in plan.steps if s.id == "step2")
        step3 = next(s for s in plan.steps if s.id == "step3")

        assert step1.dependencies == []
        assert step2.dependencies == ["step1"]
        assert set(step3.dependencies) == {"step1", "step2"}

    def test_transform_generates_expected_outputs(self):
        """Should generate expected output keys for each step."""
        ir = WorkflowIR(
            intent="Test workflow",
            tasks=[
                Task(id="task_a", description="Task A", agent_type="llm_agent"),
            ],
            failure_policy={"retry": True, "max_attempts": 3},
            metadata={"version": "v1"},
        )
        transformer = IRToPlanTransformer()

        plan = transformer.transform(ir)

        step = plan.steps[0]
        assert f"{step.id}_result" in step.expected_outputs

    def test_transform_detects_cycle(self):
        """Should raise ValueError if dependencies contain a cycle."""
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
        transformer = IRToPlanTransformer()

        with pytest.raises(ValueError, match="Cycle detected"):
            transformer.transform(ir)

    def test_transform_detects_unknown_dependency(self):
        """Should raise ValueError if task depends on unknown task."""
        ir = WorkflowIR(
            intent="Invalid workflow",
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
        transformer = IRToPlanTransformer()

        with pytest.raises(ValueError, match="depends on unknown task"):
            transformer.transform(ir)

    def test_transform_preserves_intent_as_description(self):
        """Should use the IR intent as the plan description."""
        ir = WorkflowIR(
            intent="Generate quarterly sales report from CRM",
            tasks=[],
            failure_policy={"retry": True, "max_attempts": 3},
            metadata={"version": "v1"},
        )
        transformer = IRToPlanTransformer()

        plan = transformer.transform(ir)

        assert plan.description == "Generate quarterly sales report from CRM"

    def test_transform_with_custom_goal_id(self):
        """Should use provided goal_id in the plan."""
        ir = WorkflowIR(
            intent="Test workflow",
            tasks=[Task(id="t1", description="Task 1", agent_type="llm_agent")],
            failure_policy={"retry": True, "max_attempts": 3},
            metadata={"version": "v1"},
        )
        transformer = IRToPlanTransformer()

        plan = transformer.transform(ir, goal_id="my_custom_goal")

        assert plan.goal_id == "my_custom_goal"

    def test_transform_empty_tasks(self):
        """Should handle IR with no tasks gracefully."""
        ir = WorkflowIR(
            intent="Empty workflow",
            tasks=[],
            failure_policy={"retry": True, "max_attempts": 3},
            metadata={"version": "v1"},
        )
        transformer = IRToPlanTransformer()

        plan = transformer.transform(ir)

        assert len(plan.steps) == 0
        assert plan.description == "Empty workflow"


class TestGetRequiredTools:
    """Tests for the get_required_tools method."""

    def test_gets_tools_for_data_fetcher(self):
        """Should identify http_request tool for data_fetcher."""
        ir = WorkflowIR(
            intent="Fetch data",
            tasks=[
                Task(id="fetch", description="Fetch", agent_type="data_fetcher"),
            ],
            failure_policy={"retry": True, "max_attempts": 3},
            metadata={"version": "v1"},
        )
        transformer = IRToPlanTransformer()

        tools = transformer.get_required_tools(ir)

        assert "http_request" in tools

    def test_gets_tools_for_email_agent(self):
        """Should identify send_email tool for email_agent."""
        ir = WorkflowIR(
            intent="Send email",
            tasks=[
                Task(id="email", description="Email", agent_type="email_agent"),
            ],
            failure_policy={"retry": True, "max_attempts": 3},
            metadata={"version": "v1"},
        )
        transformer = IRToPlanTransformer()

        tools = transformer.get_required_tools(ir)

        assert "send_email" in tools

    def test_combines_multiple_tools(self):
        """Should combine tools from multiple tasks."""
        ir = WorkflowIR(
            intent="Fetch and email",
            tasks=[
                Task(id="fetch", description="Fetch", agent_type="data_fetcher"),
                Task(id="email", description="Email", agent_type="email_agent"),
            ],
            failure_policy={"retry": True, "max_attempts": 3},
            metadata={"version": "v1"},
        )
        transformer = IRToPlanTransformer()

        tools = transformer.get_required_tools(ir)

        assert "http_request" in tools
        assert "send_email" in tools

    def test_no_tools_for_llm_agent(self):
        """LLM agents should not require specific tools."""
        ir = WorkflowIR(
            intent="Generate text",
            tasks=[
                Task(id="gen", description="Generate", agent_type="llm_agent"),
            ],
            failure_policy={"retry": True, "max_attempts": 3},
            metadata={"version": "v1"},
        )
        transformer = IRToPlanTransformer()

        tools = transformer.get_required_tools(ir)

        assert tools == []


class TestCompileAndTransform:
    """Tests for the compile_and_transform convenience function."""

    def test_compile_and_transform_integration(self):
        """Should compile intent and return executable Plan."""
        # Note: This test relies on the actual compiler rules
        plan = compile_and_transform(
            "Fetch report data and email it to the team",
            goal_id="email_report_goal",
        )

        assert isinstance(plan, Plan)
        assert plan.goal_id == "email_report_goal"
        # The compiler should detect "report" and "email" keywords
        assert len(plan.steps) >= 1

    def test_compile_and_transform_with_custom_templates(self):
        """Should accept custom templates for agent resolution."""
        custom_templates = {
            "my_custom_agent": AgentTemplate(
            agent_type="my_custom_agent",
            action_type=ActionType.FUNCTION,
            )
        }

        # This would need the compiler to output "my_custom_agent" type
        # For now, just verify the function accepts the parameter
        plan = compile_and_transform(
            "Generate a simple report",
            custom_templates=custom_templates,
        )

        assert isinstance(plan, Plan)


class TestEdgeCases:
    """Edge case tests for the transformer."""

    def test_self_dependency_detected_as_cycle(self):
        """Task depending on itself should be detected as cycle."""
        ir = WorkflowIR(
            intent="Self-referential",
            tasks=[
                Task(
                    id="self",
                    description="Self-referential task",
                    agent_type="llm_agent",
                    depends_on=["self"],
                ),
            ],
            failure_policy={"retry": True, "max_attempts": 3},
            metadata={"version": "v1"},
        )
        transformer = IRToPlanTransformer()

        with pytest.raises(ValueError, match="Cycle detected"):
            transformer.transform(ir)

    def test_diamond_dependency_structure(self):
        """Should handle diamond-shaped dependency graphs."""
        #     A
        #    / \
        #   B   C
        #    \ /
        #     D
        ir = WorkflowIR(
            intent="Diamond workflow",
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
                Task(
                    id="d",
                    description="Task D",
                    agent_type="llm_agent",
                    depends_on=["b", "c"],
                ),
            ],
            failure_policy={"retry": True, "max_attempts": 3},
            metadata={"version": "v1"},
        )
        transformer = IRToPlanTransformer()

        plan = transformer.transform(ir)

        assert len(plan.steps) == 4
        step_d = next(s for s in plan.steps if s.id == "d")
        assert set(step_d.dependencies) == {"b", "c"}

    def test_task_with_inputs(self):
        """Should handle tasks with input specifications."""
        ir = WorkflowIR(
            intent="Parameterized workflow",
            tasks=[
                Task(
                    id="process",
                    description="Process data",
                    agent_type="transform_agent",
                    inputs=["data_source", "$previous_output"],
                ),
            ],
            failure_policy={"retry": True, "max_attempts": 3},
            metadata={"version": "v1"},
        )
        transformer = IRToPlanTransformer()

        plan = transformer.transform(ir)

        step = plan.steps[0]
        assert "data_source" in step.inputs
        assert "previous_output" in step.inputs
