def extract_tasks(intent: str):
    tasks = []

    if "report" in intent.lower():
        tasks.append({
            "id": "fetch_data",
            "description": "Fetch required data",
            "agent_type": "data_fetch_agent",
            "inputs": ["source"]
        })
        tasks.append({
            "id": "generate_report",
            "description": "Generate report from data",
            "agent_type": "reporting_agent",
            "depends_on": ["fetch_data"]
        })

    if "email" in intent.lower():
        tasks.append({
            "id": "email_result",
            "description": "Email the generated report",
            "agent_type": "email_agent",
            "depends_on": ["generate_report"]
        })

    return tasks
