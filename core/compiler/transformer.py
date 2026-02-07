"""Transform WorkflowIR into executable Hive Plans.

This module bridges the gap between the natural language compiler's
intermediate representation (IR) and Hive's native Plan/PlanStep structures.

Example:
    >>> from core.compiler.ir import WorkflowIR, Task
    >>> from core.compiler.transformer import IRToPlanTransformer
    >>>
    >>> ir = WorkflowIR(
    ...     intent="Fetch sales data and email report",
    ...     tasks=[
    ...         Task(id="fetch", description="Get data", agent_type="data_fetcher"),
    ...         Task(id="email", description="Send report", agent_type="emailer", depends_on=["fetch"])
    ...     ],
    ...     failure_policy={"retry": True, "max_attempts": 3},
    ...     metadata={"version": "v1"}
    ... )
    >>>
    >>> transformer = IRToPlanTransformer()
    >>> plan = transformer.transform(ir)
    >>> print(plan.steps[0].action.action_type)
    ActionType.TOOL_USE
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from framework.graph.plan import ActionSpec, ActionType, Plan, PlanStep
from framework.llm.provider import Tool

from core.compiler.ir import Task, WorkflowIR


@dataclass
class AgentTemplate:
    """Template for resolving agent types to executable actions.

    Attributes:
        agent_type: The logical agent type name from the IR.
        action_type: The Hive ActionType to use.
        tool_name: Optional tool name for TOOL_USE actions.
        system_prompt: Optional system prompt for LLM_CALL actions.
        description: Human-readable description of this template.
        required_tools: List of tool names this agent requires.
    """

    agent_type: str
    action_type: ActionType
    tool_name: str | None = None
    system_prompt: str | None = None
    description: str = ""
    required_tools: list[str] | None = None


class AgentTypeResolver:
    """Resolves agent type strings from IR to Hive ActionSpecs.

    This is a simple rule-based resolver that maps common agent types
to their corresponding Hive actions. In a production system, this could
    be extended with a registry or semantic matching.
    """

    DEFAULT_TEMPLATES: dict[str, AgentTemplate] = {
        "data_fetch_agent": AgentTemplate(
            agent_type="data_fetch_agent",
            action_type=ActionType.TOOL_USE,
            tool_name="http_request",
            description="Fetches data from external APIs",
            required_tools=["http_request"],
        ),
        "data_fetcher": AgentTemplate(
            agent_type="data_fetcher",
            action_type=ActionType.TOOL_USE,
            tool_name="http_request",
            description="Fetches data from external sources",
            required_tools=["http_request"],
        ),
        "reporting_agent": AgentTemplate(
            agent_type="reporting_agent",
            action_type=ActionType.LLM_CALL,
            system_prompt="You are a reporting agent. Generate clear, structured reports from data.",
            description="Generates reports from data using LLM",
        ),
        "email_agent": AgentTemplate(
            agent_type="email_agent",
            action_type=ActionType.TOOL_USE,
            tool_name="send_email",
            description="Sends emails with generated content",
            required_tools=["send_email"],
        ),
        "transform_agent": AgentTemplate(
            agent_type="transform_agent",
            action_type=ActionType.FUNCTION,
            description="Transforms data using Python functions",
        ),
        "llm_agent": AgentTemplate(
            agent_type="llm_agent",
            action_type=ActionType.LLM_CALL,
            system_prompt="You are a helpful AI assistant.",
            description="General-purpose LLM agent",
        ),
    }

    def __init__(self, custom_templates: dict[str, AgentTemplate] | None = None):
        """Initialize resolver with optional custom templates.

        Args:
            custom_templates: Additional agent type templates to register.
        """
        self.templates = {**self.DEFAULT_TEMPLATES}
        if custom_templates:
            self.templates.update(custom_templates)

    def resolve(self, agent_type: str) -> ActionSpec:
        """Resolve an agent type string to an ActionSpec.

        Args:
            agent_type: The agent type from the IR (e.g., "data_fetch_agent").

        Returns:
            ActionSpec configured for this agent type.

        Raises:
            UnknownAgentTypeError: If the agent type is not recognized.
        """
        template = self.templates.get(agent_type)

        if template is None:
            # Default to LLM_CALL for unknown agent types
            return ActionSpec(
                action_type=ActionType.LLM_CALL,
                system_prompt=f"You are a {agent_type}. Perform your task.",
            )

        return ActionSpec(
            action_type=template.action_type,
            tool_name=template.tool_name,
            system_prompt=template.system_prompt,
        )

    def register_template(self, template: AgentTemplate) -> None:
        """Register a new agent type template.

        Args:
            template: The template to register.
        """
        self.templates[template.agent_type] = template

    def list_supported_types(self) -> list[str]:
        """Get list of supported agent type names.

        Returns:
            Sorted list of agent type strings.
        """
        return sorted(self.templates.keys())


class UnknownAgentTypeError(Exception):
    """Raised when an agent type cannot be resolved."""

    def __init__(self, agent_type: str, available: list[str]):
        self.agent_type = agent_type
        self.available = available
        super().__init__(
            f"Unknown agent type: '{agent_type}'. "
            f"Available types: {', '.join(available)}"
        )


class IRToPlanTransformer:
    """Transforms WorkflowIR into executable Hive Plans.

    This transformer converts the compiler's intermediate representation
    into Hive's native Plan structure, making the compiler output executable
    by the FlexibleGraphExecutor.

    Example:
        >>> transformer = IRToPlanTransformer()
        >>> plan = transformer.transform(ir)
        >>> # Plan is now ready for execution
        >>> result = await executor.execute_plan(plan, goal, context)
    """

    def __init__(self, resolver: AgentTypeResolver | None = None):
        """Initialize the transformer.

        Args:
            resolver: Optional custom agent type resolver.
        """
        self.resolver = resolver or AgentTypeResolver()

    def transform(
        self,
        ir: WorkflowIR,
        goal_id: str = "compiled_goal",
        plan_id: str | None = None,
    ) -> Plan:
        """Transform a WorkflowIR into a Hive Plan.

        Args:
            ir: The workflow intermediate representation from the compiler.
            goal_id: The goal ID to associate with this plan.
            plan_id: Optional plan ID (defaults to generated ID).

        Returns:
            A Plan ready for execution by FlexibleGraphExecutor.

        Raises:
            ValueError: If the IR contains invalid task references.
        """
        # Validate task dependencies
        self._validate_dependencies(ir)

        # Transform each task to a PlanStep
        steps: list[PlanStep] = []
        task_ids = {task.id for task in ir.tasks}

        for task in ir.tasks:
            step = self._transform_task(task, task_ids)
            steps.append(step)

        return Plan(
            id=plan_id or f"plan_{hash(ir.intent) & 0xFFFF:04x}",
            goal_id=goal_id,
            description=ir.intent,
            steps=steps,
            context={},
            revision=1,
        )

    def _transform_task(self, task: Task, all_task_ids: set[str]) -> PlanStep:
        """Transform a single Task to a PlanStep.

        Args:
            task: The task from the IR.
            all_task_ids: Set of all task IDs for validation.

        Returns:
            A configured PlanStep.

        Raises:
            ValueError: If task references unknown dependencies.
        """
        # Validate dependencies exist
        for dep in task.depends_on:
            if dep not in all_task_ids:
                raise ValueError(
                    f"Task '{task.id}' depends on unknown task '{dep}'"
                )

        # Resolve agent type to action
        action = self.resolver.resolve(task.agent_type)

        # Build inputs from task inputs
        inputs: dict[str, Any] = {}
        for input_key in task.inputs:
            # Reference previous task outputs using $variable syntax
            if input_key.startswith("$"):
                inputs[input_key[1:]] = input_key  # Keep as reference
            else:
                inputs[input_key] = None  # Will be provided at runtime

        # Generate expected outputs
        expected_outputs = [f"{task.id}_result"]

        return PlanStep(
            id=task.id,
            description=task.description,
            action=action,
            inputs=inputs,
            expected_outputs=expected_outputs,
            dependencies=task.depends_on,
            max_retries=3,  # Default retry policy
        )

    def _validate_dependencies(self, ir: WorkflowIR) -> None:
        """Validate that task dependencies form a valid DAG.

        Args:
            ir: The workflow IR to validate.

        Raises:
            ValueError: If a cycle is detected in dependencies.
        """
        # Build adjacency list
        graph: dict[str, list[str]] = {
            task.id: task.depends_on for task in ir.tasks
        }

        # Check for cycles using DFS
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

        for task_id in graph:
            if task_id not in visited:
                if has_cycle(task_id):
                    raise ValueError(
                        f"Cycle detected in task dependencies for intent: {ir.intent}"
                    )

    def get_required_tools(self, ir: WorkflowIR) -> list[str]:
        """Get list of tools required to execute this plan.

        Args:
            ir: The workflow IR.

        Returns:
            List of tool names needed for execution.
        """
        tools: set[str] = set()

        for task in ir.tasks:
            template = self.resolver.templates.get(task.agent_type)
            if template and template.required_tools:
                tools.update(template.required_tools)

        return sorted(tools)


def compile_and_transform(
    intent: str,
    goal_id: str = "compiled_goal",
    custom_templates: dict[str, AgentTemplate] | None = None,
) -> Plan:
    """Convenience function to compile intent and transform to Plan.

    This is a one-shot function that:
    1. Compiles natural language intent to WorkflowIR
    2. Transforms the IR to an executable Plan
    3. Returns the Plan ready for execution

    Args:
        intent: Natural language description of the workflow.
        goal_id: The goal ID to associate with the plan.
        custom_templates: Optional custom agent type templates.

    Returns:
        Executable Plan.

    Example:
        >>> plan = compile_and_transform(
        ...     "Fetch sales data and email weekly report"
        ... )
        >>> # Plan is ready for execution
    """
    from core.compiler.compiler import compile_intent

    # Compile to IR
    ir = compile_intent(intent)

    # Transform to Plan
    resolver = AgentTypeResolver(custom_templates)
    transformer = IRToPlanTransformer(resolver)

    return transformer.transform(ir, goal_id=goal_id)
