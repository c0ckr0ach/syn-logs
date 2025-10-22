import dspy

local_llama = dspy.LM(
    model='ollama_chat/llama3.1',
    api_base='http://localhost:11434',
    api_key='none',
    max_tokens=4096,
    temperature=0.7
)
dspy.settings.configure(lm=local_llama)

print("DSPy configured successfully to use local Llama 3.1 via Ollama and LiteLLM.")

generate_questions_signature = "topic -> questions: list[str]"