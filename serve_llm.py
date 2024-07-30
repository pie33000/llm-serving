from modal import Image, web_endpoint
from pydantic import BaseModel

from app import app
from llm_model import LLM


class GenerateRequest(BaseModel):
    prompt: str
    top_p: float = 0.95
    temperature: float = 1.0
    settings: dict | None


web_image = Image.debian_slim(python_version="3.11")


# For local testing, run `modal run -q serve_llm --input "Where is the best sushi in New York?"`
# @app.local_entrypoint()
@app.function(image=web_image)
@web_endpoint(method="POST", label="generate-web", docs=True)
def main(data: GenerateRequest):
    model = LLM()
    res = []
    for val in model.generate.remote_gen(data.prompt, data.top_p, data.temperature):
        res.append(val)
    return "".join(res)



