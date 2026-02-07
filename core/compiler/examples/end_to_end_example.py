"""End-to-End Example: Natural Language → IR → Plan → Execution

This example demonstrates the complete workflow from natural language intent
to an executable Hive Plan. It shows how the compiler, transformer, and
Hive's execution engine work together.

To run this example:
    cd core
    uv run python -m compiler.examples.end_to_end_example

The example intentionally uses mock execution so it can run without
external dependencies (API keys, databases, etc.).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

# Import the compiler components
from core.compiler.compiler import compile_intent
from core.compiler.ir import Task, WorkflowIR
from core.compiler.transformer import (
    AgentTemplate,
    AgentTypeResolver,
    IRToPlanTransformer,
    compile_and_transform,
)

# Import Hive's Plan structures
from framework.graph.plan import (
    ActionSpec,
    ActionType,
    Judgment,
    JudgmentAction,
    Plan,
    PlanExecutionResult,
    PlanStep,
    StepStatus,
)


@dataclass
class MockExecutionResult:
    """Mock result for demonstration purposes."""

    success: bool
    output: dict[str, Any] | None = None
    error: str | None = None


class MockPlanExecutor:
    """Mock executor that simulates plan execution without external dependencies.

    In a real scenario, you would use FlexibleGraphExecutor or AgentRunner.
    This mock demonstrates the interface and shows how Plans are executed.
    """

    def __init__(self):
        self.execution_log: list[dict] = []

    async def execute_step(self, step: PlanStep, context: dict) -> MockExecutionResult:
        """Execute a single plan step (mock implementation)."""
        print(f"  Executing: {step.id} - {step.description}")
        print(f"    Action: {step.action.action_type.value}")

        # Simulate execution based on action type
        if step.action.action_type == ActionType.TOOL_USE:
            tool = step.action.tool_name or "unknown_tool"
            print(f"    Tool: {tool}")
            # Mock successful tool execution
            return MockExecutionResult(
                success=True,
                output={f"{step.id}_result": f"Mock output from {tool}"},
            )

        elif step.action.action_type == ActionType.LLM_CALL:
            print(f"    LLM prompt would be used here")
            # Mock successful LLM generation
            return MockExecutionResult(
                success=True,
                output={f"{step.id}_result": f"Generated content for {step.id}"},
            )

        elif step.action.action_type == ActionType.FUNCTION:
            print(f"    Function execution")
            return MockExecutionResult(
                success=True,
                output={f"{step.id}_result": f"Function result for {step.id}"},
            )

        else:
            return MockExecutionResult(
                success=True,
                output={f"{step.id}_result": f"Generic result for {step.id}"},
            )

    async def execute_plan(
        self,
        plan: Plan,
        initial_context: dict[str, Any] | None = None,
    ) -> PlanExecutionResult:
        """Execute a plan step by step, respecting dependencies.

        This demonstrates how a real executor would work:
        1. Find ready steps (all dependencies completed)
        2. Execute them
        3. Store results in context
        4. Repeat until all steps complete
        """
        print(f"\n📋 Executing Plan: {plan.description}")
        print(f"   Goal ID: {plan.goal_id}")
        print(f"   Total Steps: {len(plan.steps)}")

        context = initial_context or {}
        completed_steps: set[str] = set()
        failed_steps: set[str] = set()
        total_tokens = 0
        total_latency_ms = 0

        iteration = 0
        max_iterations = len(plan.steps) * 2  # Safety limit

        while len(completed_steps) + len(failed_steps) < len(plan.steps):
            iteration += 1
            if iteration > max_iterations:
                print("⚠️  Max iterations reached, stopping")
                break

            # Find steps that are ready to execute
            ready_steps = [
                step
                for step in plan.steps
                if step.id not in completed_steps
                and step.id not in failed_steps
                and all(dep in completed_steps for dep in step.dependencies)
            ]

            if not ready_steps:
                # Check for dependency cycles or unmet dependencies
                remaining = [
                    step.id
                    for step in plan.steps
                    if step.id not in completed_steps and step.id not in failed_steps
                ]
                if remaining:
                    print(f"⚠️  Deadlock detected: {remaining}")
                break

            for step in ready_steps:
                print(f"\n▶️  Step {step.id} is ready")
                result = await self.execute_step(step, context)

                if result.success:
                    completed_steps.add(step.id)
                    if result.output:
                        context.update(result.output)
                    print(f"   ✅ Completed")
                else:
                    failed_steps.add(step.id)
                    print(f"   ❌ Failed: {result.error}")

        # Determine final status
        if failed_steps:
            status = "failed"
        elif len(completed_steps) == len(plan.steps):
            status = "completed"
        else:
            status = "partial"

        print(f"\n📊 Execution Summary:")
        print(f"   Completed: {len(completed_steps)}/{len(plan.steps)}")
        print(f"   Failed: {len(failed_steps)}/{len(plan.steps)}")
        print(f"   Status: {status}")

        return PlanExecutionResult(
            status=status,  # type: ignore
            results=context,
            completed_steps=list(completed_steps),
            steps_executed=len(completed_steps),
            total_tokens=total_tokens,
            total_latency_ms=total_latency_ms,
        )


async def example_1_basic_workflow():
    """Example 1: Basic workflow - fetch data and generate report."""
    print("=" * 60)
    print("EXAMPLE 1: Basic Report Generation Workflow")
    print("=" * 60)

    intent = "Fetch sales data and generate a weekly report"

    print(f"\n📝 Input Intent:\n   '{intent}'")

    # Step 1: Compile intent to IR
    print("\n🔨 Step 1: Compiling to Intermediate Representation (IR)")
    ir = compile_intent(intent)

    print(f"   Intent: {ir.intent}")
    print(f"   Tasks: {len(ir.tasks)}")
    for task in ir.tasks:
        deps = f" (depends on: {task.depends_on})" if task.depends_on else ""
        print(f"   - {task.id}: {task.description} [{task.agent_type}]{deps}")

    # Step 2: Transform IR to Plan
    print("\n🔄 Step 2: Transforming IR to Hive Plan")
    transformer = IRToPlanTransformer()
    plan = transformer.transform(ir, goal_id="weekly_report_goal")

    print(f"   Plan ID: {plan.id}")
    print(f"   Goal ID: {plan.goal_id}")
    print(f"   Steps: {len(plan.steps)}")

    # Show required tools
    tools = transformer.get_required_tools(ir)
    if tools:
        print(f"   Required Tools: {', '.join(tools)}")

    # Step 3: Execute the plan
    print("\n⚡ Step 3: Executing Plan")
    executor = MockPlanExecutor()
    result = await executor.execute_plan(plan)

    print(f"\n✅ Example 1 Complete!\n")
    return result


async def example_2_custom_agent_types():
    """Example 2: Using custom agent type templates."""
    print("=" * 60)
    print("EXAMPLE 2: Custom Agent Types")
    print("=" * 60)

    # Define custom templates for a specific domain
    custom_templates = {
        "slack_notifier": AgentTemplate(
            agent_type="slack_notifier",
            action_type=ActionType.TOOL_USE,
            tool_name="send_slack_message",
            description="Sends notifications to Slack channels",
            required_tools=["send_slack_message"],
        ),
        "database_query": AgentTemplate(
            agent_type="database_query",
            action_type=ActionType.TOOL_USE,
            tool_name="execute_sql",
            description="Executes SQL queries against the database",
            required_tools=["execute_sql"],
        ),
    }

    intent = "Query database and notify team on Slack"

    print(f"\n📝 Input Intent:\n   '{intent}'")
    print(f"\n🔧 Custom Templates:")
    for name, template in custom_templates.items():
        print(f"   - {name}: {template.description}")

    # Use the convenience function
    print("\n🔨 Compiling with custom templates...")
    plan = compile_and_transform(
        intent,
        goal_id="db_notification_goal",
        custom_templates=custom_templates,
    )

    print(f"\n📋 Generated Plan:")
    print(f"   Description: {plan.description}")
    print(f"   Steps: {len(plan.steps)}")
    for step in plan.steps:
        print(f"   - {step.id}: {step.action.action_type.value}")

    print(f"\n✅ Example 2 Complete!\n")


async def example_3_dependency_validation():
    """Example 3: Demonstrating dependency validation."""
    print("=" * 60)
    print("EXAMPLE 3: Dependency Validation")
    print("=" * 60)

    # Create an IR with a valid DAG structure
    valid_ir = WorkflowIR(
        intent="Process data pipeline",
        tasks=[
            Task(id="extract", description="Extract data", agent_type="data_fetcher"),
            Task(
                id="transform",
                description="Transform data",
                agent_type="transform_agent",
                depends_on=["extract"],
            ),
            Task(
                id="load",
                description="Load to warehouse",
                agent_type="data_fetcher",
                depends_on=["transform"],
            ),
        ],
        failure_policy={"retry": True, "max_attempts": 3},
        metadata={"version": "v1"},
    )

    print("\n✅ Valid DAG:")
    print("   extract → transform → load")

    transformer = IRToPlanTransformer()
    plan = transformer.transform(valid_ir)
    print(f"   Successfully created plan with {len(plan.steps)} steps")

    # Try creating an IR with a cycle (this should fail)
    print("\n❌ Invalid DAG (cycle):")
    print("   A → B → C → A")

    cyclic_ir = WorkflowIR(
        intent="Cyclic workflow",
        tasks=[
            Task(
                id="a", description="Task A", agent_type="llm_agent", depends_on=["c"]
            ),
            Task(
                id="b", description="Task B", agent_type="llm_agent", depends_on=["a"]
            ),
            Task(
                id="c", description="Task C", agent_type="llm_agent", depends_on=["b"]
            ),
        ],
        failure_policy={"retry": True, "max_attempts": 3},
        metadata={"version": "v1"},
    )

    try:
        transformer.transform(cyclic_ir)
        print("   ERROR: Should have raised ValueError!")
    except ValueError as e:
        print(f"   Correctly detected: {e}")

    print(f"\n✅ Example 3 Complete!\n")


async def example_4_manual_ir_construction():
    """Example 4: Manually constructing an IR and transforming it."""
    print("=" * 60)
    print("EXAMPLE 4: Manual IR Construction")
    print("=" * 60)

    # Manually build an IR for a complex workflow
    ir = WorkflowIR(
        intent="Multi-channel marketing campaign",
        tasks=[
            Task(
                id="generate_content",
                description="Generate marketing copy with LLM",
                agent_type="llm_agent",
                inputs=["campaign_topic"],
            ),
            Task(
                id="create_email",
                description="Create email from content",
                agent_type="transform_agent",
                depends_on=["generate_content"],
                inputs=["$generate_content_result"],
            ),
            Task(
                id="create_social",
                description="Create social media posts",
                agent_type="transform_agent",
                depends_on=["generate_content"],
                inputs=["$generate_content_result"],
            ),
            Task(
                id="send_email",
                description="Send email to subscribers",
                agent_type="email_agent",
                depends_on=["create_email"],
            ),
            Task(
                id="post_social",
                description="Post to social media",
                agent_type="data_fetcher",  # Represents API call
                depends_on=["create_social"],
            ),
        ],
        failure_policy={"retry": True, "max_attempts": 3},
        metadata={"campaign_id": "summer_2024"},
    )

    print("\n📝 Manually Constructed IR:")
    print(f"   Intent: {ir.intent}")

    # Visualize the dependency graph
    print("\n   Dependency Graph:")
    print("          generate_content")
    print("                 /\\")
    print("                /  \\")
    print("       create_email  create_social")
    print("              |           |")
    print("       send_email     post_social")

    # Transform and execute
    print("\n🔄 Transforming to Plan...")
    transformer = IRToPlanTransformer()
    plan = transformer.transform(ir, goal_id="marketing_campaign")

    print(f"\n⚡ Executing...")
    executor = MockPlanExecutor()
    result = await executor.execute_plan(plan, initial_context={"campaign_topic": "Summer Sale"})

    print(f"\n📊 Final Context:")
    for key, value in (result.results or {}).items():
        print(f"   {key}: {str(value)[:50]}...")

    print(f"\n✅ Example 4 Complete!\n")


async def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("HIVE COMPILER: End-to-End Examples")
    print("Natural Language → IR → Plan → Execution")
    print("=" * 60 + "\n")

    await example_1_basic_workflow()
    await example_2_custom_agent_types()
    await example_3_dependency_validation()
    await example_4_manual_ir_construction()

    print("=" * 60)
    print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("\nKey Takeaways:")
    print("  1. Natural language intents compile to structured IR")
    print("  2. IR transforms to Hive Plans with validated dependencies")
    print("  3. Plans are executable by Hive's FlexibleGraphExecutor")
    print("  4. Custom agent types extend the system for any domain")
    print()


if __name__ == "__main__":
    asyncio.run(main())
