#!/bin/sh
# Start the API and the web page in one container.
# The API listens on 8000 and the Streamlit page on 8501.
set -e
uvicorn codelist_rag.api:app --host 0.0.0.0 --port 8000 &
exec streamlit run app/streamlit_app.py --server.port 8501 --server.address 0.0.0.0 \
  --server.headless true --browser.gatherUsageStats false
