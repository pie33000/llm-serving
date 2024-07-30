import os

from modal import Image

llm_base_image = (
    Image.debian_slim(python_version="3.11")
    .pip_install("autoawq==0.1.8", "torch==2.1.2", "huggingface_hub[hf_transfer]")
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "GPU_NAME": os.getenv("GPU_NAME", "T4"),
            "MODEL_NAME": os.getenv("MODEL_NAME", "TheBloke/OpenOrca-Zephyr-7B-AWQ"),
            "IS_QUANTIZED": os.getenv("IS_QUANTIZED", "False"),
            "APP_NAME": os.getenv("APP_NAME", "test_e2e"),
            "HF_TOKEN": os.getenv("HF_TOKEN"),
        }
    )
)
