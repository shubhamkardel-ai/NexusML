import requests
import streamlit as st


API_BASE_URL = "http://127.0.0.1:8000"


st.set_page_config(
    page_title="NexusML Monitoring",
    page_icon="🤖",
    layout="wide",
)


st.title("NexusML — ML Monitoring Dashboard")
st.caption("Production ML lifecycle and model reliability platform")


def get_api_data(endpoint: str):
    """Fetch data from the NexusML API."""

    response = requests.get(
        f"{API_BASE_URL}{endpoint}",
        timeout=10,
    )

    response.raise_for_status()

    return response.json()


# ---------------------------------------------------------
# API Health
# ---------------------------------------------------------

try:
    health = get_api_data("/health")

    st.success(
        f"API Online — {health['model']} "
        f"(Version {health['model_version']})"
    )

except requests.RequestException:
    st.error(
        "NexusML API is unavailable. "
        "Start the FastAPI server first."
    )

    st.stop()


# ---------------------------------------------------------
# Monitoring Data
# ---------------------------------------------------------

try:
    metrics = get_api_data("/monitoring/metrics")
    models = get_api_data("/monitoring/models")
    drift = get_api_data("/monitoring/drift")

except requests.RequestException as error:
    st.error(f"Unable to retrieve monitoring data: {error}")
    st.stop()


# ---------------------------------------------------------
# Overview
# ---------------------------------------------------------

st.header("Production Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total Predictions",
        metrics["total_predictions"],
    )

with col2:
    st.metric(
        "Churn Predictions",
        metrics["churn_predictions"],
    )

with col3:
    st.metric(
        "Non-Churn Predictions",
        metrics["non_churn_predictions"],
    )

with col4:
    st.metric(
        "Avg Churn Probability",
        f"{metrics['average_churn_probability']:.2%}",
    )


st.divider()


# ---------------------------------------------------------
# Model Information
# ---------------------------------------------------------

st.header("Active Model")

model_col1, model_col2 = st.columns(2)

with model_col1:

    st.info(
        f"**Model:** {health['model']}\n\n"
        f"**Version:** {health['model_version']}"
    )

with model_col2:

    if models["models"]:

        active_model = models["models"][0]

        st.info(
            f"**Production Predictions:** "
            f"{active_model['prediction_count']}\n\n"
            f"**Average Probability:** "
            f"{active_model['average_churn_probability']:.2%}"
        )


st.divider()


# ---------------------------------------------------------
# Drift Monitoring
# ---------------------------------------------------------

st.header("Feature Drift Detection")


if drift["status"] == "insufficient_data":

    st.warning(
        "Not enough production data for reliable drift detection."
    )

    st.write(
        f"Production samples: "
        f"{drift['production_samples']} / "
        f"{drift['minimum_required_samples']}"
    )

elif drift["drift_detected"]:

    st.error("⚠️ Feature drift detected")

    drifted_features = drift["drifted_features"]

    st.write(
        f"**Drifted features:** "
        f"{', '.join(drifted_features)}"
    )

else:

    st.success("✅ No significant feature drift detected")


# ---------------------------------------------------------
# Feature Analysis
# ---------------------------------------------------------

if drift.get("feature_results"):

    st.subheader("Feature Drift Analysis")

    feature_rows = []

    for feature, result in drift["feature_results"].items():

        feature_rows.append(
            {
                "Feature": feature,
                "Reference Mean": result["reference_mean"],
                "Production Mean": result["production_mean"],
                "Mean Change (%)": result["percentage_change"],
                "KS Statistic": result["ks_statistic"],
                "P-Value": result["p_value"],
                "Drift Detected": result["drift_detected"],
            }
        )

    st.dataframe(
        feature_rows,
        use_container_width=True,
        hide_index=True,
    )


st.divider()


# ---------------------------------------------------------
# Refresh
# ---------------------------------------------------------

st.caption(
    "Data is retrieved from the NexusML FastAPI monitoring endpoints."
)

if st.button("Refresh Monitoring Data"):

    st.rerun()