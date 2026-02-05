from core.compiler.ir import Task, WorkflowIR
from core.compiler.rules import extract_tasks

def compile_intent(intent: str) -> WorkflowIR:
    raw_tasks = extract_tasks(intent)

    tasks = [Task(**task) for task in raw_tasks]

    return WorkflowIR(
        intent=intent,
        tasks=tasks,
        failure_policy={
            "retry": True,
            "max_attempts": 3
        },
        metadata={
            "generated_by": "nl_to_workflow_compiler",
            "version": "v1"
        }
    )
