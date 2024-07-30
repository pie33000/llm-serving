from app import app
from llm_model import LLM


@app.local_entrypoint()
def main(input: str, top_p: float = 0.95, temperature: float = 0.9):
    model = LLM()
    for val in model.generate.remote_gen(input, top_p, temperature):
        print(val, end="", flush=True)
