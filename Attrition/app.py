import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import os

st.set_page_config(page_title="Employee Attrition Risk Dashboard", layout="wide")

# Folder where app.py is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_artifacts():
    model = joblib.load(os.path.join(BASE_DIR, "attrition_model.pkl"))
    scaler = joblib.load(os.path.join(BASE_DIR, "attrition_scaler.pkl"))
    encoders = joblib.load(os.path.join(BASE_DIR, "attrition_encoders.pkl"))
    feature_names = joblib.load(os.path.join(BASE_DIR, "attrition_features.pkl"))
    explainer = joblib.load(os.path.join(BASE_DIR, "attrition_explainer.pkl"))
    reference_df = pd.read_csv(os.path.join(BASE_DIR, "attrition_reference.csv"))

    return model, scaler, encoders, feature_names, explainer, reference_df

model, scaler, encoders, feature_names, explainer, reference_df = load_artifacts()

st.title("Employee Attrition Risk Dashboard")
st.caption("XGBoost model + SHAP explainability, trained on the IBM HR Analytics dataset")

tab1, tab2 = st.tabs([
    "Single Employee Check",
    "Department Overview"
])

# ---------------------------------------------------------------
# TAB 1: Single employee — form input, risk score, SHAP waterfall
# ---------------------------------------------------------------
with tab1:
    st.subheader("Score a single employee")
# TAB 1: Single employee — form input, risk score, SHAP waterfall
# ---------------------------------------------------------------
with tab1:
    st.subheader("Score a single employee")

    # Columns that were label-encoded — show their original category labels in the form
    categorical_cols = [c for c in encoders.keys() if c != "Attrition" and c in feature_names]
    numeric_cols = [c for c in feature_names if c not in categorical_cols]

    with st.form("employee_form"):
        col1, col2, col3 = st.columns(3)
        user_input = {}

        for i, col in enumerate(categorical_cols):
            target_col = [col1, col2, col3][i % 3]
            options = list(encoders[col].classes_)
            choice = target_col.selectbox(col, options)
            user_input[col] = encoders[col].transform([choice])[0]

        for i, col in enumerate(numeric_cols):
            target_col = [col1, col2, col3][i % 3]
            default = int(reference_df[col].median())
            lo, hi = int(reference_df[col].min()), int(reference_df[col].max())
            user_input[col] = target_col.number_input(col, min_value=lo, max_value=hi, value=default)

        submitted = st.form_submit_button("Predict Attrition Risk")

    if submitted:
        row = pd.DataFrame([user_input])[feature_names]
        row_scaled = scaler.transform(row)

        risk = model.predict_proba(row_scaled)[0, 1]
        pred = model.predict(row_scaled)[0]

        c1, c2 = st.columns([1, 2])
        with c1:
            st.metric("Attrition Risk", f"{risk:.1%}")
            st.write("**Prediction:**", "Likely to leave" if pred == 1 else "Likely to stay")
            if risk > 0.5:
                st.warning("High risk — consider a retention conversation.")
            else:
                st.success("Low risk.")

        with c2:
            st.write("**Why this score? (SHAP)**")
            sv = explainer.shap_values(row_scaled)
            explanation = shap.Explanation(
                values=sv[0],
                base_values=explainer.expected_value,
                data=row_scaled[0],
                feature_names=feature_names,
            )
            fig, ax = plt.subplots(figsize=(8, 5))
            shap.plots.waterfall(explanation, show=False)
            st.pyplot(fig, bbox_inches="tight")

# ---------------------------------------------------------------
# TAB 2: Department-level view — batch score + heatmap
# ---------------------------------------------------------------
with tab2:
    st.subheader("Department-level attrition risk")

    uploaded = st.file_uploader(
        "Upload an employee CSV (same columns as training data) — or leave empty to use the reference dataset",
        type="csv",
    )
    data = pd.read_csv(uploaded) if uploaded is not None else reference_df.copy()

    X_batch = data[feature_names]
    X_batch_scaled = scaler.transform(X_batch)
    data["Predicted Risk"] = model.predict_proba(X_batch_scaled)[:, 1]

    # Decode Department + JobRole back to readable labels for the heatmap, if encoded
    dept_col = data.copy()
    if "Department" in encoders:
        dept_col["Department"] = encoders["Department"].inverse_transform(dept_col["Department"])
    if "JobRole" in encoders:
        dept_col["JobRole"] = encoders["JobRole"].inverse_transform(dept_col["JobRole"])

    pivot = dept_col.pivot_table(
        index="JobRole", columns="Department", values="Predicted Risk", aggfunc="mean"
    )

    st.write("**Average predicted attrition risk by Department / Job Role**")
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(pivot.values, cmap="Reds", aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    for r in range(pivot.shape[0]):
        for c in range(pivot.shape[1]):
            val = pivot.values[r, c]
            if not np.isnan(val):
                ax.text(c, r, f"{val:.0%}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="Avg. predicted risk")
    st.pyplot(fig)

    st.write("**Highest-risk employees**")
    st.dataframe(
        data.sort_values("Predicted Risk", ascending=False).head(15),
        use_container_width=True,
    )
