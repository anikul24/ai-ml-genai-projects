## 🚀 retirement-genai-advisor


```
retirement-genai-advisor/
├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt
├── pyproject.toml (optional)
├── data/                    # raw documents, excel files, PDFs (NOT checked in)
├── notebooks/
│   └── research_agent_langgraph.ipynb   # copy or move your existing notebook here
├── src/
│   ├── ingest.py            # ingest + chunking + embeddings + upsert
│   ├── index_utils.py       # small helpers for vector DB (batching, size checks)
│   ├── qa_service.py        # conversational retrieval chain + wrappers
│   ├── tools.py             # calculators, policy extractors, small tools
│   ├── graph_agent.py       # optional LangGraph orchestrator glue
│   └── app_streamlit.py     # streamlit UI (or app_gradio.py)
├── tests/
│   ├── test_ingest.py
│   ├── test_tools.py
│   └── test_qa_service.py
└── infra/
    └── deploy.md           # notes for deployment (Streamlit Cloud / HF / Docker)


```