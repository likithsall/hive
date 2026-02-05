from dataclasses import dataclass, field
from typing import List, Dict

@dataclass
class Task:
    id: str
    description: str
    agent_type: str
    inputs: List[str] = field(default_factory=list)
    depends_on: List[str] = field(default_factory=list)

@dataclass
class WorkflowIR:
    intent: str
    tasks: List[Task]
    failure_policy: Dict
    metadata: Dict
