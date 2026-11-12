
from pathlib import Path
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Model Performance | CareInsight",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Model Performance")

st.write("""
Explore how four machine learning models performed
on the held-out NSS 76th Round survey test data.
""")

ROOT = Path(__file__).resolve().parents[2]
csv_path = ROOT / "reports" / "model_comparison.csv"

if not csv_path.exists():
    st.error("Model comparison CSV is unavailable.")
    st.stop()

results = pd.read_csv(csv_path)

st.subheader("Performance Overview")

col1, col2 = st.columns(2)

with col1:
    st.metric("Best Test ROC-AUC", "0.7694")
    st.caption("Stacking")

with col2:
    st.metric("Deployed Model ROC-AUC", "0.7683")
    st.caption("Gradient Boosting")

st.divider()

metrics = {
    "ROC-AUC": "Test ROC-AUC",
    "F1 Score": "F1",
    "Precision": "Precision",
    "Recall": "Recall",
    "Train-Test AUC Gap": "Train-Test Gap"
}

selected = st.selectbox(
    "Select a model evaluation metric",
    list(metrics.keys())
)

metric_column = metrics[selected]

st.bar_chart(
    results.set_index("Model")[[metric_column]]
)

with st.expander("What does this metric mean?"):
    descriptions = {
        "ROC-AUC": (
            "Measures how well a model ranks positive "
            "cases above negative cases."
        ),
        "F1 Score": (
            "Balances precision and recall."
        ),
        "Precision": (
            "Of the cases predicted positive, "
            "what proportion were actually positive?"
        ),
        "Recall": (
            "Of the actual positive cases, "
            "what proportion did the model identify?"
        ),
        "Train-Test AUC Gap": (
            "The difference between training and test "
            "AUC. A larger positive gap may suggest "
            "overfitting."
        )
    }

    st.write(descriptions[selected])

st.subheader("Detailed Comparison")

st.dataframe(
    results,
    hide_index=True,
    use_container_width=True
)

st.subheader("Why Gradient Boosting?")

st.write("""
Although Stacking achieved the highest test ROC-AUC
(0.7694), Gradient Boosting achieved nearly identical
performance (0.7683).

Gradient Boosting was selected for deployment because
it offers a simpler inference pipeline.

The model choice also involves precision, recall,
complexity, and deployment considerations.
""")

st.info(
    "These are historical evaluation results from "
    "the original research notebook, not live "
    "production performance metrics."
)
