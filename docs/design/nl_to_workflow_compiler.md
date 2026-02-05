## Workflow Intermediate Representation (IR)

The compiler outputs a deterministic workflow intermediate representation (IR).
This IR acts as a bridge between natural language intent and executable Hive workflows.

### IR Structure

```yaml
intent: <string>

tasks:
  - id: <string>
    description: <string>
    agent_type: <string>
    inputs: [<string>]
    depends_on: [<string>]

failure_policy:
  retry: <boolean>
  max_attempts: <integer>

metadata:
  generated_by: nl_to_workflow_compiler
  version: v1

Field Semantics

intent: Original user intent in natural language.

tasks: Ordered list of atomic tasks derived from the intent.

id: Unique task identifier.

description: Human-readable explanation of the task.

agent_type: Logical agent category (not implementation-specific).

inputs: Required inputs or parameters.

depends_on: Task dependencies forming a DAG.

failure_policy: Global retry behavior.

metadata: Compiler and versioning information.

Design Notes

The IR is declarative, not executable.

Execution engines consume the IR but do not modify it.

Agent resolution happens downstream, not in the compiler.


---

### What this achieves
- Locks the **contract**
- Makes review easy
- Gives maintainers something concrete to discuss
- Prevents scope creep later

---
