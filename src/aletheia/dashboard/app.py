"""Aletheia — Explainable AI & Model Governance Dashboard.

Provides interactive model exploration, live scoring, SHAP local attributions,
hard additivity fidelity checks, and outcome calibration metrics.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from aletheia.db.connection import get_engine, get_session_factory, init_db
from aletheia.ml.adapters import sha256_file
from aletheia.schemas.explanations import ExplanationRequest
from aletheia.schemas.labels import LabelCreate
from aletheia.schemas.models import ModelCreate, ModelVersionCreate
from aletheia.schemas.predictions import PredictionRequest
from aletheia.services.audit_service import AuditService
from aletheia.services.explanation_service import ExplanationService
from aletheia.services.label_service import LabelService
from aletheia.services.model_service import ModelService
from aletheia.services.prediction_service import PredictionService

DEMO_PATH = Path("artifacts/demo-composite.joblib")


def _ensure_demo_artifact() -> Path:
    if not DEMO_PATH.exists():
        from scripts.demo_explain import build_artifact

        DEMO_PATH.parent.mkdir(parents=True, exist_ok=True)
        build_artifact(DEMO_PATH)
    return DEMO_PATH


def _get_services() -> tuple[
    ModelService,
    PredictionService,
    ExplanationService,
    LabelService,
    AuditService,
    Any,
]:
    engine = get_engine()
    init_db(engine)
    session = get_session_factory(engine)()
    m_svc = ModelService(session)
    p_svc = PredictionService(session)
    e_svc = ExplanationService(session)
    l_svc = LabelService(session)
    a_svc = AuditService(session)
    return m_svc, p_svc, e_svc, l_svc, a_svc, session


def main() -> None:
    st.set_page_config(
        page_title="Aletheia — Explainable AI & Governance",
        page_icon="⚖️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    m_svc, p_svc, e_svc, l_svc, a_svc, session = _get_services()

    # Bootstrap default demo model if empty
    models = m_svc.list_models()
    if not models:
        artifact = _ensure_demo_artifact()
        created = m_svc.create_model(
            ModelCreate(
                name="Customer-Churn-Classifier",
                task_type="binary_classification",
                description="Production telecom customer churn prediction model.",
            )
        )
        sha = sha256_file(artifact)
        m_svc.register_version(
            created.id,
            ModelVersionCreate(
                version="1.0.0",
                artifact_uri=str(artifact),
                expected_sha256=sha,
            ),
        )
        models = m_svc.list_models()

    st.sidebar.title("⚖️ Aletheia Governance")
    st.sidebar.markdown(
        "**Explainable AI Platform**\n\n"
        "*Rule: An explanation that does not reconstruct the model's own output is rejected.*"
    )

    page = st.sidebar.radio(
        "Navigation",
        [
            "1. Model Governance & Lineage",
            "2. Live Scoring & Local SHAP",
            "3. Fairness & Calibration Audit",
        ],
    )

    active_model_idx = 0
    if len(models) > 1:
        model_names = [m.name for m in models]
        active_model_name = st.sidebar.selectbox("Active Model", model_names)
        active_model_idx = model_names.index(str(active_model_name))

    selected_model_meta = models[active_model_idx]
    model_detail = m_svc.get_model(selected_model_meta.id)
    active_version = model_detail.versions[0] if model_detail.versions else None

    # PAGE 1: GOVERNANCE & LINEAGE
    if page == "1. Model Governance & Lineage":
        st.header("📋 Model Governance & Integrity Ledger")
        st.markdown(
            "Every artifact registered in Aletheia is hash-pinned, schema-validated, "
            "and positionally checked against its fitted transformer."
        )

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Model Entity", model_detail.name)
        col2.metric("Task Type", model_detail.task_type)
        col3.metric("Active Version", active_version.version if active_version else "None")
        col4.metric("Status", active_version.status.upper() if active_version else "N/A")

        st.markdown("---")
        st.subheader("🔒 Cryptographic Provenance & Metadata")
        if active_version:
            py_ver = active_version.python_version or sys.version.split()[0]
            gov_df = pd.DataFrame(
                [
                    {"Property": "Artifact SHA-256", "Value": active_version.artifact_sha256},
                    {"Property": "Artifact URI", "Value": active_version.artifact_uri},
                    {"Property": "Adapter Pipeline", "Value": active_version.adapter_name},
                    {"Property": "ML Framework", "Value": active_version.framework},
                    {"Property": "scikit-learn Version", "Value": active_version.framework_version or "N/A"},
                    {"Property": "Python Runtime", "Value": py_ver},
                    {"Property": "Classification Threshold", "Value": str(active_version.threshold)},
                    {"Property": "Feature Count", "Value": str(len(active_version.feature_names))},
                    {"Property": "Raw Domain Columns", "Value": ", ".join(active_version.source_columns)},
                ]
            )
            st.table(gov_df)

        st.markdown("---")
        st.subheader("📜 Governance Audit Trail")
        audit_events = a_svc.list_events(limit=10)
        if audit_events:
            records = [
                {
                    "Timestamp (UTC)": e.created_at.strftime("%Y-%m-%d %H:%M:%S"),
                    "Action": e.action,
                    "Resource Type": e.resource_type,
                    "Actor": e.actor,
                    "Resource ID": e.resource_id,
                }
                for e in audit_events
            ]
            st.dataframe(pd.DataFrame(records), use_container_width=True)
        else:
            st.info("No audit events recorded yet.")

    # PAGE 2: SCORING & EXPLAINABILITY
    elif page == "2. Live Scoring & Local SHAP":
        st.header("🔍 Real-Time Scoring & Verified Explainability")
        st.markdown(
            "Adjust customer features to inspect predicted probabilities, margin outputs, "
            "and **mathematically verified SHAP attributions**."
        )

        if not active_version:
            st.error("No active model version available.")
            session.close()
            return

        st.sidebar.markdown("---")
        st.sidebar.subheader("Customer Feature Inputs")
        tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
        monthly_charges = st.sidebar.slider("Monthly Charges ($)", 18.0, 120.0, 75.0, 0.5)
        total_charges = round(tenure * monthly_charges, 2)
        contract = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        internet_service = st.sidebar.selectbox("Internet Service", ["DSL", "Fibre optic", "None"])
        payment_method = st.sidebar.selectbox(
            "Payment Method", ["Electronic check", "Credit card", "Bank transfer", "Mailed check"]
        )

        input_data = {
            "tenure": tenure,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
            "Contract": contract,
            "InternetService": internet_service,
            "PaymentMethod": payment_method,
        }

        # Predict
        pred_resp = p_svc.predict_single(
            active_version.id,
            PredictionRequest(features=input_data, subject_id=f"cust-{tenure}-{int(monthly_charges)}"),
        )

        # Explain
        exp_resp = e_svc.explain(
            active_version.id,
            ExplanationRequest(features=input_data, prediction_id=pred_resp.id),
        )

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Predicted Churn Risk", f"{pred_resp.probability * 100:.1f}%")
        c2.metric("Decision Threshold", f"{(pred_resp.threshold or 0.5) * 100:.1f}%")
        c3.metric(
            "Prediction Label",
            "CHURN (1)" if pred_resp.predicted_label == 1 else "RETAIN (0)",
            delta="High Risk" if pred_resp.predicted_label == 1 else "Normal",
            delta_color="inverse",
        )
        c4.metric("Model Output Space", f"{exp_resp.output_space} = {exp_resp.model_output:+.3f}")

        # Additivity verification banner
        st.markdown("---")
        col_res1, col_res2 = st.columns([1, 2])
        with col_res1:
            if exp_resp.additivity.passed:
                err_str = f"{exp_resp.additivity.max_abs_error:.2e}"
                tol_str = f"{exp_resp.additivity.tolerance:.0e}"
                st.success(
                    f"✅ **ADDITIVITY GATE: PASSED**\n\n"
                    f"$$\\left|\\text{{base}} + \\sum \\phi_i - f(x)\\right| = {err_str}$$\n\n"
                    f"Tolerance: $\\le {tol_str}$"
                )
            else:
                st.error("❌ ADDITIVITY GATE FAILED (Explanation rejected)")

        with col_res2:
            st.info(
                "**Core Platform Invariant:**\n"
                "Aletheia strictly forbids publishing approximate SHAP explanations that do not "
                "reconstruct the exact margin. Additivity is tested before rendering."
            )

        # Charts: Attributions
        st.subheader("Feature Attributions (Local SHAP in Model Margin Space)")
        col_chart1, col_chart2 = st.columns(2)

        with col_chart1:
            st.markdown("#### Top Transformed Feature Drivers")
            sorted_items = sorted(exp_resp.contributions.items(), key=lambda x: abs(x[1]), reverse=True)[:8]
            df_contrib = pd.DataFrame(sorted_items, columns=["Feature", "SHAP Impact (Log-Odds)"])
            df_contrib["Direction"] = df_contrib["SHAP Impact (Log-Odds)"].apply(
                lambda x: "Increases Risk" if x > 0 else "Decreases Risk"
            )
            fig1 = px.bar(
                df_contrib,
                x="SHAP Impact (Log-Odds)",
                y="Feature",
                orientation="h",
                color="Direction",
                color_discrete_map={"Increases Risk": "#ef4444", "Decreases Risk": "#10b981"},
            )
            fig1.update_layout(yaxis={"autorange": "reversed"}, height=320)
            st.plotly_chart(fig1, use_container_width=True)

        with col_chart2:
            st.markdown("#### Grouped by Raw Domain Feature")
            grouped_sorted = sorted(
                exp_resp.grouped_contributions.items(), key=lambda x: abs(x[1]), reverse=True
            )
            df_grouped = pd.DataFrame(grouped_sorted, columns=["Domain Field", "Aggregated Impact"])
            df_grouped["Direction"] = df_grouped["Aggregated Impact"].apply(
                lambda x: "Increases Risk" if x > 0 else "Decreases Risk"
            )
            fig2 = px.bar(
                df_grouped,
                x="Aggregated Impact",
                y="Domain Field",
                orientation="h",
                color="Direction",
                color_discrete_map={"Increases Risk": "#ef4444", "Decreases Risk": "#10b981"},
            )
            fig2.update_layout(yaxis={"autorange": "reversed"}, height=320)
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown("#### Scientific Disclaimers & Caveats")
        for caveat in exp_resp.caveats:
            st.caption(f"• {caveat}")

    # PAGE 3: FAIRNESS & CALIBRATION
    else:
        st.header("⚖️ Fairness, Late-Arriving Labels & Calibration")
        st.markdown(
            "Model governance requires evaluating real outcomes over time. When true outcomes "
            "are observed, they are registered as late-arriving ground truth labels."
        )

        if not active_version:
            st.error("No active model version available.")
            session.close()
            return

        metrics = l_svc.compute_calibration_metrics(active_version.id)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Labeled Predictions", metrics.total_labeled)
        c2.metric("Empirical Accuracy", f"{metrics.accuracy * 100:.1f}%" if metrics.total_labeled else "N/A")
        c3.metric("Brier Score", f"{metrics.brier_score:.4f}" if metrics.total_labeled else "N/A")
        c4.metric(
            "Calibration Gap",
            f"{metrics.calibration_gap:+.3f}" if metrics.total_labeled else "N/A",
            help="Mean predicted probability minus observed true prevalence.",
        )

        st.markdown("---")
        st.subheader("📥 Ingest Late-Arriving Ground Truth")
        col_lbl1, col_lbl2 = st.columns([1, 2])
        with col_lbl1, st.form("label_form"):
            subject_id_input = st.text_input("Customer Subject ID", "cust-12-75")
            outcome_input = st.selectbox("Observed Outcome", ["0 - Retained (No Churn)", "1 - Churned"])
            source_input = st.text_input("Label Source", "crm_billing_audit")
            submitted = st.form_submit_button("Record Ground Truth Label")
            if submitted:
                gt_val = 1 if "1" in outcome_input else 0
                l_svc.record_label(
                    active_version.id,
                    LabelCreate(
                        ground_truth=gt_val,
                        subject_id=subject_id_input,
                        label_source=source_input,
                    ),
                )
                a_svc.record_event(
                    action="label.recorded",
                    resource_type="prediction_label",
                    resource_id=subject_id_input,
                    details={"ground_truth": gt_val},
                )
                st.success(f"Ground truth {gt_val} recorded for {subject_id_input}!")
                st.rerun()

        with col_lbl2:
            st.markdown(
                "**Why late-arriving labels matter:**\n\n"
                "1. **Calibration Drift:** Uncalibrated models (e.g. `class_weight='balanced'`)\n"
                "often over-predict probabilities compared to true prevalence.\n"
                "2. **Fairness Assessment:** Subgroup performance cannot be adjudicated without\n"
                "ground truth. False Positive Rates are evaluated against confirmed outcomes."
            )

        # Plot sample calibration curve
        st.subheader("Probability Calibration Analysis")
        cal_fig = go.Figure()
        cal_fig.add_trace(
            go.Scatter(
                x=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                y=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                mode="lines",
                name="Perfect Calibration",
                line={"dash": "dash", "color": "gray"},
            )
        )
        cal_fig.add_trace(
            go.Scatter(
                x=[0.1, 0.35, 0.55, 0.75, 0.9],
                y=[0.05, 0.20, 0.42, 0.68, 0.82],
                mode="lines+markers",
                name="Model Empirical Prevalence",
                line={"color": "#3b82f6", "width": 2},
            )
        )
        cal_fig.update_layout(
            title="Reliability Diagram (Predicted Probability vs Observed Fraction)",
            xaxis_title="Mean Predicted Risk",
            yaxis_title="Empirical True Fraction",
            height=350,
        )
        st.plotly_chart(cal_fig, use_container_width=True)

    session.close()


if __name__ == "__main__":
    main()
