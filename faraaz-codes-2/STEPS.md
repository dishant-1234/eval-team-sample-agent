# Local setup

Needs Python 3.12 on the machine.

```powershell
py -3.12 -m venv .venv
.venv\Scripts\activate

pip install -U pip
pip install -e .
pip install pytest pytest-asyncio "google-adk[eval]>=1.18.0,<2.0.0"
copy example.env .env
```

Populate `.env` with `OPENAI_API_KEY`, `OPENAI_MODEL`, and `OPENAI_BASE_URL`.
Add `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` if you want traces.

```powershell
adk web --port=8000
```