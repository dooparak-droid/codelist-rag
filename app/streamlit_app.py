"""streamlit_app.py

Browser front end for the codelist-rag service. It sends requests to the
FastAPI service (default http://localhost:8000) and shows the codelist with
fabricated codes highlighted. Start the service first, then run:

    streamlit run app/streamlit_app.py
"""

import io
import os

import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("CODELIST_API_URL", "http://localhost:8000").rstrip("/")
REQUEST_TIMEOUT_SECONDS = 600  # reasoning models can take over a minute

DEFAULT_MODELS = {
    "openai": "gpt-5.5",
    "google": "gemini-3.1-flash-lite",
    "ollama": "medgemma:4b",
}
STRATEGIES = {
    "zero_shot": "Zero-shot (default)",
    "few_shot": "Few-shot",
    "chain_of_thought": "Chain of thought",
}

st.set_page_config(page_title="codelist-rag", layout="wide")
st.title("codelist-rag")
st.warning(
    "Research tool, not for clinical use. A qualified clinician must review every "
    "codelist before it is used in a study."
)


def api_get(path):
    return requests.get(f"{API_URL}{path}", timeout=10)


try:
    health = api_get("/health").json()
except requests.RequestException:
    st.error(f"Cannot reach the service at {API_URL}. Start it first, then reload this page.")
    st.stop()

with st.sidebar:
    st.subheader("Service")
    st.write(f"Version {health['version']}")
    st.write(f"Terminology: {health['terminology_release']}")
    st.write(f"{health['terminology_concepts']:,} concepts loaded")
    if not health["retriever_loaded"]:
        st.error("No index is loaded, so retrieval is unavailable.")

condition = st.text_input("Condition", placeholder="Atrial fibrillation")

col1, col2, col3 = st.columns(3)
with col1:
    provider = st.selectbox("Provider", list(DEFAULT_MODELS))
    model = st.text_input("Model", value=DEFAULT_MODELS[provider], key=f"model_{provider}")
with col2:
    strategy = st.radio("Prompt", list(STRATEGIES), format_func=STRATEGIES.get)
with col3:
    use_rag = st.checkbox("Use retrieval", value=True, disabled=not health["retriever_loaded"])
    n_results = st.number_input("Candidate concepts", 10, 1000, 400, disabled=not use_rag)

if provider == "openai" and model.startswith("gpt-5.5"):
    st.caption("GPT-5.5 is a paid model. In testing a request cost about $0.20 to $0.25.")

if st.button("Generate codelist", type="primary", disabled=not condition.strip()):
    payload = {
        "condition": condition.strip(),
        "strategy": strategy,
        "use_rag": use_rag,
        "n_results": int(n_results),
        "provider": provider,
        "model": model.strip(),
    }
    with st.spinner("Waiting for the model. This can take a minute or more."):
        try:
            resp = requests.post(f"{API_URL}/codelist", json=payload, timeout=REQUEST_TIMEOUT_SECONDS)
        except requests.RequestException as e:
            st.session_state.pop("result", None)
            st.error(f"Request failed: {e}")
        else:
            if resp.status_code == 200:
                st.session_state["result"] = resp.json()
                st.session_state.pop("score", None)
            else:
                st.session_state.pop("result", None)
                st.error(f"Service returned {resp.status_code}: {resp.json().get('detail', resp.text)}")

result = st.session_state.get("result")
if result:
    codes = pd.DataFrame(result["codes"])
    n_flagged = int((codes["code_is_real"] == False).sum())  # noqa: E712 (None means unchecked)
    st.subheader(f"{result['condition']}: {result['total_codes']} codes")
    st.write(
        f"{n_flagged} not found in the index"
        + (f" ({result['fabrication_rate']:.1%})" if result["fabrication_rate"] is not None else "")
        + f" | model {result['model']} | strategy {result['strategy']}"
        + f" | retrieval {'on' if result['use_rag'] else 'off'}"
    )

    view = st.radio("Show", ["All codes", "Flagged only"], horizontal=True)
    shown = codes[codes["code_is_real"] == False] if view == "Flagged only" else codes  # noqa: E712
    shown = shown.rename(columns={"code": "Code", "term": "Term", "code_is_real": "In index"})

    def highlight(row):
        colour = "background-color: #f8d7da" if row["In index"] is False else ""
        return [colour] * len(row)

    st.dataframe(shown.style.apply(highlight, axis=1), width="stretch", hide_index=True)

    d1, d2, _ = st.columns([1, 1, 4])
    d1.download_button("Download CSV", codes.to_csv(index=False), "codelist.csv", "text/csv")
    d2.download_button("Download JSON", pd.Series(result).to_json(indent=2), "codelist.json", "application/json")

    st.divider()
    st.subheader("Score against a reference codelist (optional)")
    uploaded = st.file_uploader("CSV with a column named 'code'", type="csv")
    if uploaded is not None:
        gold = pd.read_csv(io.BytesIO(uploaded.getvalue()), dtype=str)
        if "code" not in gold.columns:
            st.error("The file needs a column named 'code'.")
        else:
            body = {
                "generated_codes": [{"code": c["code"], "term": c["term"]} for c in result["codes"]],
                "gold_codes": gold["code"].dropna().tolist(),
                "retrieval_type": "non_rag",
            }
            score = requests.post(f"{API_URL}/evaluate", json=body, timeout=30).json()
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Precision", f"{score['precision']:.2f}")
            m2.metric("Recall", f"{score['recall']:.2f}")
            m3.metric("F1", f"{score['f1']:.2f}")
            m4.metric("Generated / reference", f"{score['generated_count']} / {score['gold_count']}")
            st.caption(
                "Codes are matched exactly and not deduplicated before scoring, as in the thesis."
            )
