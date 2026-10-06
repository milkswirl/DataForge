import streamlit as st
import pandas as pd
from scipy.stats import ks_2samp

st.set_page_config(
    page_title="DataForge Drift Detection",
    layout="wide"
)

st.title("Dataset Drift Detection")
st.caption("Compare two dataset versions and detect important changes.")

col1, col2 = st.columns(2)

with col1:
    old_file = st.file_uploader(
        "Upload Dataset V1",
        type=["csv"],
        key="old"
    )

with col2:
    new_file = st.file_uploader(
        "Upload Dataset V2",
        type=["csv"],
        key="new"
    )


def numeric_drift(old_df, new_df):
    results = []

    common_columns = sorted(
        set(old_df.columns) & set(new_df.columns)
    )

    for column in common_columns:

        if (
            pd.api.types.is_numeric_dtype(old_df[column])
            and pd.api.types.is_numeric_dtype(new_df[column])
        ):

            old_values = old_df[column].dropna()
            new_values = new_df[column].dropna()

            if len(old_values) < 2 or len(new_values) < 2:
                continue

            statistic, p_value = ks_2samp(
                old_values,
                new_values
            )

            results.append({
                "Column": column,
                "KS Statistic": round(statistic, 4),
                "P-Value": round(p_value, 4),
                "Drift Detected": p_value < 0.05
            })

    return pd.DataFrame(results)


if old_file is not None and new_file is not None:

    old_df = pd.read_csv(old_file)
    new_df = pd.read_csv(new_file)

    st.success("Both datasets loaded successfully.")

    st.header("Dataset Comparison")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "V1 Rows",
        f"{len(old_df):,}"
    )

    c2.metric(
        "V2 Rows",
        f"{len(new_df):,}",
        delta=len(new_df) - len(old_df)
    )

    c3.metric(
        "V1 Columns",
        len(old_df.columns)
    )

    c4.metric(
        "V2 Columns",
        len(new_df.columns),
        delta=len(new_df.columns) - len(old_df.columns)
    )

    st.divider()

    st.header("Schema Changes")

    old_columns = set(old_df.columns)
    new_columns = set(new_df.columns)

    added_columns = sorted(
        list(new_columns - old_columns)
    )

    removed_columns = sorted(
        list(old_columns - new_columns)
    )

    if added_columns:
        st.warning(
            "Added columns: " + ", ".join(added_columns)
        )

    if removed_columns:
        st.warning(
            "Removed columns: " + ", ".join(removed_columns)
        )

    if not added_columns and not removed_columns:
        st.success(
            "No columns were added or removed."
        )

    st.divider()

    st.header("Numerical Distribution Drift")

    drift_results = numeric_drift(
        old_df,
        new_df
    )

    if not drift_results.empty:

        st.dataframe(
            drift_results,
            use_container_width=True
        )

        drifted = drift_results[
            drift_results["Drift Detected"]
        ]

        if not drifted.empty:
            st.warning(
                f"{len(drifted)} numerical columns "
                "show significant drift."
            )
        else:
            st.success(
                "No significant numerical drift detected."
            )

    else:
        st.info(
            "No comparable numerical columns were found."
        )

    st.divider()

    st.header("Missing Value Changes")

    common_columns = sorted(
        old_columns & new_columns
    )

    missing_results = []

    for column in common_columns:

        old_missing = (
            old_df[column].isnull().mean() * 100
        )

        new_missing = (
            new_df[column].isnull().mean() * 100
        )

        missing_results.append({
            "Column": column,
            "V1 Missing %": round(old_missing, 2),
            "V2 Missing %": round(new_missing, 2),
            "Change": round(
                new_missing - old_missing,
                2
            )
        })

    missing_df = pd.DataFrame(
        missing_results
    )

    st.dataframe(
        missing_df,
        use_container_width=True
    )

else:

    st.info(
        "Upload both Dataset V1 and Dataset V2 "
        "to begin drift analysis."
    )