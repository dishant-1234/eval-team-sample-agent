needs python 3.12 on the machine

py -3.12 -m venv .venv
.venv\Scripts\activate

pip install -U pip
pip install -e .

populate env variables 

adk web --port=8000