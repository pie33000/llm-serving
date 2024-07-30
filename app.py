import os

from modal import App

APP_NAME = os.getenv("APP_NAME", "llm_serve")

app = App(name=APP_NAME)
