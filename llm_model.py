import os
import time

from modal import build, enter, method

from app import app
from base_image import llm_base_image

MODEL_NAME = os.getenv("MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.2")
GPU_NAME = os.getenv("GPU_NAME", "T4")
CONTAINER_IDLE_TIMEOUT = os.getenv("CONTAINER_IDLE_TIMEOUT", 300)
IS_QUANTIZED = bool(os.getenv("IS_QUANTIZED", False))
HF_TOKEN = os.getenv("HF_TOKEN")


with llm_base_image.imports():
    from threading import Thread

    from awq import AutoAWQForCausalLM
    from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer


@app.cls(
    image=llm_base_image, gpu=GPU_NAME, container_idle_timeout=CONTAINER_IDLE_TIMEOUT
)
class LLM:
    @build()
    def download_model(self):
        from huggingface_hub import snapshot_download

        snapshot_download(
            MODEL_NAME,
            headers={"Authorization": HF_TOKEN},
        )

    @enter()
    def load_model(self):
        t0 = time.time()
        print(f"Model name: {MODEL_NAME}")
        if IS_QUANTIZED:
            print("Loading AWQ quantized model...")
            self.model = AutoAWQForCausalLM.from_quantized(
                MODEL_NAME, fuse_layers=False, version="GEMV"
            )
        else:
            print("Loading model...")
            self.model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).cuda()

        print(f"Model loaded in {time.time() - t0:.2f}s")

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True
        )

    @method()
    async def generate(self, input, top_p, temperature, history=[]):
        if input == "":
            return

        t0 = time.time()

        assert len(history) % 2 == 0, "History must be an even number of messages"

        messages = [{"role": "system", "content": ""}]
        for i in range(0, len(history), 2):
            messages.append({"role": "user", "content": history[i]})
            messages.append({"role": "user", "content": history[i + 1]})

        messages.append({"role": "user", "content": input})
        tokenized_chat = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).cuda()

        generation_kwargs = dict(
            inputs=tokenized_chat,
            streamer=self.streamer,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=1.2,
            max_new_tokens=1024,
        )

        # Run generation on separate thread to enable response streaming.
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()
        for new_text in self.streamer:
            yield new_text
        thread.join()

        print(f"Output generated in {time.time() - t0:.2f}s")
