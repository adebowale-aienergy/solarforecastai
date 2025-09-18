import streamlit as st
import pandas as pd
import os
import numpy as np
import plotly.express as px

# --- Imports from your src package ---
from src.data_utils import (
    filter_data,
    prepare_data_for_model,
    make_prophet_frame,
    get_unique_values,
)
from src.visualization import (
    preview_table,
    plot_parameter_distribution_boxplot,
    plot_time_series_by_country,
)
from src.model_utils import (
    train_random_forest_model, train_prophet_model, train_lstm_model,
    predict_random_forest, forecast_prophet, predict_lstm, create_sequences
)
from src.eval_utils import calculate_regression_metrics
from src.constants import (
    TARGET_COL, COUNTRY_COL, PARAMETER_COL, VALUE_COL,
    RF_FEATURES, LSTM_FEATURES, DEFAULT_HORIZON, MIN_HORIZON,
    MAX_HORIZON, SOLAR_FORECAST_PARAMETERS
)
from src.geo import get_country_regions, get_country_coordinates

# --- Streamlit App ---
st.set_page_config(layout="wide", page_title="Global Solar Forecasting Dashboard")
st.title("🌍 Global Solar Forecasting and Monitoring Dashboard")

st.markdown("""
Welcome to the **Solar Forecasting and Monitoring Dashboard**.  
Explore climate data and solar forecasting insights across different regions and countries.
""")

# --- Data Loader ---
@st.cache_data
def load_data(model_type=None):
    """Loads the correct processed dataset depending on model type."""
    try:
        if model_type == "Random Forest":
            path = os.path.join("data", "features_data.csv")
            df = pd.read_csv(path)

        elif model_type in ["Prophet", "LSTM"]:
            path = os.path.join("data", "clean_data.csv")
            df = pd.read_csv(path)

            # Standardize date column
            if "observation_date" in df.columns:
                df.rename(columns={"observation_date": "ds"}, inplace=True)
            elif "date" in df.columns:
                df.rename(columns={"date": "ds"}, inplace=True)

            df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
        else:
            path = os.path.join("data", "features_data.csv")
            df = pd.read_csv(path)

        return df
    except FileNotFoundError:
        st.error(f"❌ Error: Data file not found at {path}.")
        return None
    except Exception as e:
        st.error(f"⚠️ An error occurred while loading the data: {e}")
        return None

# --- Sidebar Model Config ---
st.sidebar.header("⚙️ Model Configuration")
model_type = st.sidebar.selectbox("Select Model Type", ["Random Forest", "Prophet", "LSTM"])

# Load data dynamically
df = load_data(model_type)

# Regions and coordinates
country_regions = get_country_regions()
all_regions = ["All"] + sorted(list(country_regions.keys()))
country_coordinates = get_country_coordinates()

if df is not None:
    st.success("✅ Data loaded successfully!")

    # --- Display Data Info ---
    st.header("📊 Data Overview")
    st.write("Preview of the processed dataset:")
    st.dataframe(preview_table(df), use_container_width=True)
    st.write(f"**Shape:** {df.shape}")
    st.write(f"**Columns:** {df.columns.tolist()}")

    # --- Data Filtering ---
    st.sidebar.header("🔎 Filter Data")
    selected_region_filter = st.sidebar.selectbox("Select Continent", all_regions)

    all_countries = get_unique_values(df, COUNTRY_COL)

    # handle parameters safely
    if PARAMETER_COL in df.columns:
        all_parameters = get_unique_values(df, PARAMETER_COL)
    else:
        all_parameters = []

    if selected_region_filter != "All":
        countries_in_region = country_regions.get(selected_region_filter, [])
        available_countries_filter = [c for c in all_countries if c in countries_in_region]
    else:
        available_countries_filter = all_countries

    selected_country_filter = st.sidebar.selectbox("Select Country", ["All"] + available_countries_filter)

    if PARAMETER_COL in df.columns:
        selected_parameter_filter = st.sidebar.selectbox("Select Parameter", ["All"] + all_parameters)
    else:
        selected_parameter_filter = "All"

    filtered_df = df.copy()
    if selected_country_filter != "All":
        filtered_df = filter_data(filtered_df, country=selected_country_filter)
    if PARAMETER_COL in df.columns and selected_parameter_filter != "All":
        filtered_df = filter_data(filtered_df, parameter=selected_parameter_filter)

    st.header("📑 Filtered Data")
    st.write(f"Filtered by **Country:** {selected_country_filter} | **Parameter:** {selected_parameter_filter}")
    st.dataframe(preview_table(filtered_df), use_container_width=True)

    # --- Visualizations ---
    st.header("🌐 Data Visualizations")

    if PARAMETER_COL in df.columns:
        # Global Map Visualization
        st.subheader("Global Parameter Distribution")
        map_parameter = st.selectbox("Select Parameter for Global Map", SOLAR_FORECAST_PARAMETERS)

        if map_parameter:
            map_data_parameter = df[df[PARAMETER_COL] == map_parameter].copy()
            if not map_data_parameter.empty:
                avg_param_country = map_data_parameter.groupby(COUNTRY_COL)[VALUE_COL].mean().reset_index()

                # Choropleth Map
                fig_global_map = px.choropleth(
                    avg_param_country,
                    locations=COUNTRY_COL,
                    locationmode="country names",
                    color=VALUE_COL,
                    hover_name=COUNTRY_COL,
                    color_continuous_scale="YlOrRd",
                    title=f"Average {map_parameter} by Country"
                )
                st.plotly_chart(fig_global_map, use_container_width=True)

        # Parameter-specific visualization
        if selected_parameter_filter != "All":
            if selected_country_filter == "All":
                fig_boxplot = plot_parameter_distribution_boxplot(df, selected_parameter_filter)
                st.plotly_chart(fig_boxplot, use_container_width=True)
            else:
                ts_df = filter_data(df, country=selected_country_filter, parameter=selected_parameter_filter)
                if not ts_df.empty:
                    fig_ts = plot_time_series_by_country(df, selected_parameter_filter, [selected_country_filter])
                    st.plotly_chart(fig_ts, use_container_width=True)
    else:
        st.info("⚠️ This dataset doesn’t have a `parameter` column (likely Random Forest). Skipping parameter-based visualizations.")

    # --- Modeling Section ---
    st.header("🤖 Modeling and Forecasting")
    selected_region_model = st.sidebar.selectbox("Select Continent for Modeling", all_regions)
    if selected_region_model != "All":
        countries_in_region_model = country_regions.get(selected_region_model, [])
        available_countries_model = [c for c in all_countries if c in countries_in_region_model]
    else:
        available_countries_model = all_countries

    selected_country_model = st.sidebar.selectbox("Select Country for Modeling", available_countries_model)

    if PARAMETER_COL in df.columns:
        selected_parameter_model = st.sidebar.selectbox(
            "Select Parameter for Modeling",
            all_parameters,
            index=all_parameters.index(TARGET_COL) if TARGET_COL in all_parameters else 0
        )
        model_df = filter_data(df, country=selected_country_model, parameter=selected_parameter_model)
    else:
        selected_parameter_model = TARGET_COL
        model_df = df[df[COUNTRY_COL] == selected_country_model].copy()

    forecast_horizon = st.sidebar.number_input("Forecast Horizon (days)", min_value=MIN_HORIZON, max_value=MAX_HORIZON, value=DEFAULT_HORIZON, step=1)

    if not model_df.empty:
        st.subheader(f"{model_type} Forecasting for {selected_parameter_model} in {selected_country_model}")

        train_size = int(len(model_df) * 0.8)
        train_df = model_df.iloc[:train_size].copy()
        test_df = model_df.iloc[train_size:].copy()

        if st.button(f"🚀 Train {model_type} Model"):
            try:
                if model_type == "Random Forest":
                    rf_train_X, rf_train_y = prepare_data_for_model(train_df, selected_parameter_model, RF_FEATURES, VALUE_COL)
                    rf_test_X, rf_test_y = prepare_data_for_model(test_df, selected_parameter_model, RF_FEATURES, VALUE_COL)
                    model = train_random_forest_model(rf_train_X, rf_train_y)
                    y_pred_eval = predict_random_forest(model, rf_test_X)
                    y_true_eval = rf_test_y.values

                elif model_type == "Prophet":
                    prophet_train_df = make_prophet_frame(train_df, date_col="ds", target_col=VALUE_COL, parameter=selected_parameter_model)
                    model = train_prophet_model(prophet_train_df)
                    future_periods = len(test_df) + forecast_horizon
                    forecast_results_df = forecast_prophet(model, periods=future_periods)

                    test_df["ds"] = pd.to_datetime(test_df["ds"], errors="coerce")
                    forecast_results_df["ds"] = pd.to_datetime(forecast_results_df["ds"])
                    merged_eval_df = pd.merge(test_df, forecast_results_df[["ds","yhat"]], on="ds", how="inner")
                    y_true_eval = merged_eval_df[VALUE_COL].values
                    y_pred_eval = merged_eval_df["yhat"].values

                elif model_type == "LSTM":
                    lstm_train_df = filter_data(train_df, parameter=selected_parameter_model) if PARAMETER_COL in df.columns else train_df
                    lstm_test_df = filter_data(test_df, parameter=selected_parameter_model) if PARAMETER_COL in df.columns else test_df
                    lstm_train_df["ds"] = pd.to_datetime(lstm_train_df["ds"], errors="coerce")
                    lstm_test_df["ds"] = pd.to_datetime(lstm_test_df["ds"], errors="coerce")
                    lstm_train_df.set_index("ds", inplace=True)
                    lstm_test_df.set_index("ds", inplace=True)

                    features = [col for col in LSTM_FEATURES if col in lstm_train_df.columns]
                    X_train_seq, y_train_seq = create_sequences(lstm_train_df[features], lstm_train_df[VALUE_COL], time_steps=10)
                    X_test_seq, y_test_seq = create_sequences(lstm_test_df[features], lstm_test_df[VALUE_COL], time_steps=10)

                    model = train_lstm_model(X_train_seq, y_train_seq, epochs=50)
                    y_pred_eval = predict_lstm(model, X_test_seq)
                    y_true_eval = y_test_seq

                # --- Evaluation ---
                metrics = calculate_regression_metrics(y_true_eval, y_pred_eval)
                st.subheader("📈 Model Evaluation")
                st.write(metrics)

            except Exception as e:
                st.error(f"⚠️ Error during model training/forecasting: {e}")
else:
    st.warning("⚠️ Could not load data. Please check the file path and try again.")

# --- About Section ---
st.sidebar.header("ℹ️ About")
st.sidebar.info("""
This dashboard analyzes climate data from **NASA POWER** for solar forecasting.  
It includes data visualization, exploration, and forecasting using **Random Forest, Prophet, and LSTM models**.  

**Data Source:** NASA POWER Project (power.larc.nasa.gov)
""")
