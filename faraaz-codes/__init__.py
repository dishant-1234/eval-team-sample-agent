# __init__.py  ── OpenAI version
# Loads env vars and exposes the model name. No Google/Vertex auth needed.

import os
from dotenv import load_dotenv

load_dotenv()

MODEL = os.getenv("OPENAI_MODEL", "gpt-5.2")
