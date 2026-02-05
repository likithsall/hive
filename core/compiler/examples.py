from core.compiler.compiler import compile_intent

def demo():
    intent = "Generate weekly sales report and email it"
    workflow = compile_intent(intent)
    return workflow
