# Researcher-GPT
This repo is forked from <https://github.com/JayZeeDesign/researcher-gpt>. Made changes to search
using Google's API and using `requests` directly for webscraping. I may try using bedrock instead of
OpenAI in the future.

## Setup:
1. Get Google search api keys: <https://python.langchain.com/docs/integrations/tools/google_search>
2. Get OpenAI api key: <https://platform.openai.com/api-keys>
3. Setup `.env` file:
```sh
GOOGLE_API_KEY=<GOOGLE_API_KEY>
GOOGLE_CSE_ID=<GOOGLE_SEARCH_KEY>
OPENAI_API_KEY=<OPENAI_API_KEY>
```
4. `pip install` the requirements

## Run
You have two options: Streamlit (browser-based) or FastAPI

### Streamlit
1. Uncomment streamlit stuff, then run
```sh
streamlit run app.py
```

### FastAPI
1. Uncomment FastAPI stuff, then run (you may need to `pip install uvicorn` before)
```sh
uvicorn app:app --reload
```
2. To interact, run something like
```py
import requests

data = {"query": "How to become a better software developer in 2023?"}
url = "http://127.0.0.1:8000"
print(requests.post(url, json=data).json())
```