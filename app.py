import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(
    page_title="DataForge",
    layout="wide"
)

st.title("DataForge")
st.caption("Automated Data Quality, Cleaning & Dataset Analysis")

uploaded_file = st.file_uploader(
    "Upload a CSV dataset",
    type=["csv"]
)


def get_outlier_details(df):
    numeric_columns = df.select_dtypes(include=np.number).columns

    details = []

    for column in numeric_columns:
        series = df[column].dropna()

        if len(series) == 0:
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outliers = series[
            (series < lower_bound) |
            (series > upper_bound)
        ]

        details.append({
            "Column": column,
            "Outliers": len(outliers),
            "Lower Bound": round(lower_bound, 2),
            "Upper Bound": round(upper_bound, 2)
        })

    return pd.DataFrame(details)


def count_outliers(df):
    details = get_outlier_details(df)

    if details.empty:
        return 0

    return int(details["Outliers"].sum())


def quality_score(df):
    total_cells = df.shape[0] * df.shape[1]

    missing_values = int(df.isnull().sum().sum())
    duplicate_rows = int(df.duplicated().sum())
    outlier_count = count_outliers(df)

    if total_cells > 0:
        missing_percent = (missing_values / total_cells) * 100
    else:
        missing_percent = 0

    if len(df) > 0:
        duplicate_percent = (
            duplicate_rows / len(df)
        ) * 100
    else:
        duplicate_percent = 0

    numeric_columns = df.select_dtypes(include=np.number).columns

    if len(df) > 0 and len(numeric_columns) > 0:
        outlier_percent = (
            outlier_count /
            (len(df) * len(numeric_columns))
        ) * 100
    else:
        outlier_percent = 0

    score = 100

    score -= min(missing_percent * 2, 30)
    score -= min(duplicate_percent * 2, 20)
    score -= min(outlier_percent, 20)

    return max(round(score), 0)


def clean_dataset(df):
    cleaned_df = df.copy()

    # Remove duplicate rows
    cleaned_df = cleaned_df.drop_duplicates()

    # Fill missing numerical values with median
    numeric_columns = cleaned_df.select_dtypes(include=np.number).columns

    for column in numeric_columns:
        if cleaned_df[column].isnull().any():
            cleaned_df[column] = cleaned_df[column].fillna(
                cleaned_df[column].median()
            )

    # Fill missing text values with mode
    text_columns = cleaned_df.select_dtypes(
        include=["object", "string"]
    ).columns

    for column in text_columns:
        if cleaned_df[column].isnull().any():

            mode = cleaned_df[column].mode()

            if not mode.empty:
                cleaned_df[column] = cleaned_df[column].fillna(
                    mode.iloc[0]
                )

    # Normalize column names
    cleaned_df.columns = [
        column.strip().lower().replace(" ", "_")
        for column in cleaned_df.columns
    ]

    return cleaned_df


if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    score = quality_score(df)

    missing_values = int(df.isnull().sum().sum())
    duplicates = int(df.duplicated().sum())
    outlier_count = count_outliers(df)

    st.success("Dataset uploaded successfully.")

    # ----------------------------
    # Dataset overview
    # ----------------------------

    st.header("Dataset Overview")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Rows",
        f"{len(df):,}"
    )

    c2.metric(
        "Columns",
        len(df.columns)
    )

    c3.metric(
        "Missing Values",
        missing_values
    )

    c4.metric(
        "Quality Score",
        f"{score}/100"
    )

    st.divider()

    # ----------------------------
    # Data quality
    # ----------------------------

    st.header("Data Quality")

    st.progress(score / 100)

    if score >= 90:
        st.success("Excellent data quality")

    elif score >= 75:
        st.info("Good data quality with some issues")

    elif score >= 50:
        st.warning("Dataset requires cleaning")

    else:
        st.error("Poor data quality detected")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Missing Values",
        missing_values
    )

    c2.metric(
        "Duplicate Rows",
        duplicates
    )

    c3.metric(
        "Potential Outliers",
        outlier_count
    )

    st.divider()

    # ----------------------------
    # Missing value analysis
    # ----------------------------

    st.header("Missing Value Analysis")

    missing_by_column = (
        df.isnull()
        .sum()
        .reset_index()
    )

    missing_by_column.columns = [
        "Column",
        "Missing Values"
    ]

    missing_by_column = missing_by_column[
        missing_by_column["Missing Values"] > 0
    ]

    if not missing_by_column.empty:

        fig_missing = px.bar(
            missing_by_column,
            x="Column",
            y="Missing Values",
            title="Missing Values by Column"
        )

        st.plotly_chart(
            fig_missing,
            use_container_width=True
        )

    else:
        st.success("No missing values detected.")

    # ----------------------------
    # Outlier analysis
    # ----------------------------

    st.header("Outlier Analysis")

    outlier_details = get_outlier_details(df)

    if not outlier_details.empty:

        st.dataframe(
            outlier_details,
            use_container_width=True
        )

        outlier_chart_data = outlier_details[
            outlier_details["Outliers"] > 0
        ]

        if not outlier_chart_data.empty:

            fig_outliers = px.bar(
                outlier_chart_data,
                x="Column",
                y="Outliers",
                title="Potential Outliers by Column"
            )

            st.plotly_chart(
                fig_outliers,
                use_container_width=True
            )

        else:
            st.success("No numerical outliers detected.")

    # ----------------------------
    # Distribution explorer
    # ----------------------------

    st.header("Distribution Explorer")

    numeric_columns = list(
        df.select_dtypes(include=np.number).columns
    )

    if numeric_columns:

        selected_column = st.selectbox(
            "Select a numerical column",
            numeric_columns
        )

        fig_hist = px.histogram(
            df,
            x=selected_column,
            title=f"Distribution of {selected_column}"
        )

        st.plotly_chart(
            fig_hist,
            use_container_width=True
        )

        fig_box = px.box(
            df,
            y=selected_column,
            title=f"Box Plot of {selected_column}"
        )

        st.plotly_chart(
            fig_box,
            use_container_width=True
        )

    # ----------------------------
    # Column profile
    # ----------------------------

    st.header("Column Profile")

    profile = pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str).values,
        "Missing": df.isnull().sum().values,
        "Missing %": (
            df.isnull().mean() * 100
        ).round(2).values,
        "Unique Values": df.nunique().values
    })

    st.dataframe(
        profile,
        use_container_width=True
    )

    st.divider()

    # ----------------------------
    # Automatic cleaning
    # ----------------------------

    st.header("Automatic Data Cleaning")

    st.write(
        "DataForge can remove duplicate rows, "
        "fill missing values, and standardize column names."
    )

    if st.button(
        "Clean Dataset",
        type="primary"
    ):

        cleaned_df = clean_dataset(df)

        st.session_state["cleaned_df"] = cleaned_df

    if "cleaned_df" in st.session_state:

        cleaned_df = st.session_state["cleaned_df"]

        cleaned_score = quality_score(cleaned_df)

        st.success("Dataset cleaned successfully.")

        st.subheader("Before vs After")

        b1, b2, b3, b4 = st.columns(4)

        b1.metric(
            "Quality Score",
            f"{cleaned_score}/100",
            delta=cleaned_score - score
        )

        b2.metric(
            "Missing Values",
            int(cleaned_df.isnull().sum().sum()),
            delta=(
                int(cleaned_df.isnull().sum().sum())
                - missing_values
            )
        )

        b3.metric(
            "Duplicate Rows",
            int(cleaned_df.duplicated().sum()),
            delta=(
                int(cleaned_df.duplicated().sum())
                - duplicates
            )
        )

        b4.metric(
            "Rows",
            len(cleaned_df),
            delta=(
                len(cleaned_df)
                - len(df)
            )
        )

        st.subheader("Cleaned Dataset Preview")

        st.dataframe(
            cleaned_df.head(25),
            use_container_width=True
        )

        csv = cleaned_df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "Download Cleaned CSV",
            data=csv,
            file_name="dataforge_cleaned.csv",
            mime="text/csv"
        )

else:

    st.info(
        "Upload a CSV file to begin analyzing your dataset."
    )