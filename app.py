# app.py - Updated and hardened version
import streamlit as st
import pandas as pd
import os
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import traceback

# --- Imports from your src package ---
from src.data_utils import (
    load_processed_data,  # expects a function that loads + normalizes date -> 'ds'
    filter_data,
    prepare_data_for_model,
    make_prophet_frame,
    get_unique_values,
)
from src.visualization import (
    preview_table,
    plot_parameter_distribution_boxplot,
    plot_time_series_by_country,
    line_actual_vs_pred,
    prophet_forecast_plot,
)
from src.model_utils import (
    train_random_forest_model, train_prophet_model, train_lstm_model,
    predict_random_forest, forecast_prophet, predict_lstm, create_sequences
)
from src.eval_utils import calculate_regression_metrics
from src.constants import (
    TARGET_COL, COUNTRY_COL, PARAMETER_COL, VALUE_COL,
    RF_FEATURES, LSTM_FEATURES, DEFAULT_HORIZON, MIN_HORIZON,
    MAX_HORIZON, SOLAR_FORECAST_PARAMETERS, PARAMETER_UNITS
)
from src.geo import get_country_regions, get_country_coordinates

# ---------- App config ----------
st.set_page_config(layout="wide", page_title="Global Solar Forecasting Dashboard")
st.title("🌍 Global Solar Forecasting and Monitoring Dashboard")
st.markdown("Explore NASA POWER climate data, visualize it, and run forecasts with Random Forest, Prophet, or LSTM.")

# ---------- Helpers ----------
def plot_forecast_vs_actual(dates, y_true, y_pred, title="Actual vs Predicted", y_label=None):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=y_true, mode="lines", name="Actual"))
    fig.add_trace(go.Scatter(x=dates, y=y_pred, mode="lines", name="Predicted"))
    fig.update_layout(title=title, xaxis_title="Date", yaxis_title=(y_label or VALUE_COL))
    return fig

# ---------- Caching ----------
@st.cache_data
def cached_load_data(model_type: str | None = None):
    """Load processed dataset: features_data.csv for RF, clean_data.csv for Prophet/LSTM."""
    try:
        if model_type == "Random Forest":
            path = os.path.join("data", "features_data.csv")
        else:
            path = os.path.join("data", "clean_data.csv")
        df = load_processed_data(path)  # assumes this will normalize date column to 'ds'
        df["_source_path"] = path
        return df
    except Exception as e:
        raise

# Cache model trainers to avoid re-training during same session
@st.cache_resource
def cached_train_random_forest(X, y):
    return train_random_forest_model(X, y)

@st.cache_resource
def cached_train_prophet(df_prophet):
    return train_prophet_model(df_prophet)

@st.cache_resource
def cached_train_lstm(X, y, epochs=20):
    return train_lstm_model(X, y, epochs=epochs)

# ---------- Sidebar: configuration & guides ----------
st.sidebar.header("⚙️ Configuration")
model_type = st.sidebar.selectbox("Select Model Type", ["Random Forest", "Prophet", "LSTM"])

with st.sidebar.expander("ℹ️ Parameter descriptions", expanded=False):
    st.markdown(
        "- **ALLSKY_KT**: clearness index\n"
        "- **T2M**: temperature at 2m\n"
        "- **WS2M**: wind speed at 2m\n"
        "- **RH2M**: relative humidity at 2m\n"
        "\n(If you have custom parameters, ensure they are included in the dataset.)"
    )

with st.sidebar.expander("📈 Metric guide", expanded=False):
    st.markdown(
        "- **MAE**: Mean Absolute Error (lower is better)\n"
        "- **MSE**: Mean Squared Error (lower is better)\n"
        "- **RMSE**: Root MSE (lower is better)\n"
        "- **R-squared**: closer to 1 is better"
    )

# ---------- Load data ----------
try:
    df = cached_load_data(model_type)
except Exception as e:
    st.error(f"Failed to load data for model_type={model_type}: {e}")
    st.exception(traceback.format_exc())
    st.stop()

# ---------- Basic info & diagnostics ----------
country_regions = get_country_regions()
all_regions = ["All"] + sorted(list(country_regions.keys()))
country_coordinates = get_country_coordinates()

st.sidebar.header("🔎 Quick Diagnostics")
if st.sidebar.button("Show Diagnostics"):
    st.header("🛠 Diagnostics")
    st.write("Source file:", df.get("_source_path", "unknown"))
    st.write("Columns:", df.columns.tolist())
    st.write("Sample:")
    st.dataframe(df.head(10), use_container_width=True)
    st.write("Missing values per column:")
    st.dataframe(df.isna().sum().to_frame("missing_count").sort_values("missing_count", ascending=False), use_container_width=True)

# ---------- Data preview (sampled for speed) ----------
st.header("📊 Data Overview")
st.write("Preview (sample):")
n_sample = min(200, len(df))
if n_sample > 0:
    st.dataframe(preview_table(df.sample(n_sample)), use_container_width=True)
st.write(f"Shape: {df.shape}")

# ---------- Filters (safe wrt PARAMETER_COL) ----------
st.sidebar.header("Filters")
selected_region_filter = st.sidebar.selectbox("Select Continent", all_regions)

all_countries = get_unique_values(df, COUNTRY_COL) if COUNTRY_COL in df.columns else []
if PARAMETER_COL in df.columns:
    all_parameters = get_unique_values(df, PARAMETER_COL)
else:
    all_parameters = []

if selected_region_filter != "All":
    countries_in_region = country_regions.get(selected_region_filter, [])
    available_countries_filter = [c for c in all_countries if c in countries_in_region]
else:
    available_countries_filter = all_countries

selected_country_filter = st.sidebar.selectbox("Select Country (display)", ["All"] + available_countries_filter)

if all_parameters:
    selected_parameter_filter = st.sidebar.selectbox("Select Parameter (display)", ["All"] + all_parameters)
else:
    selected_parameter_filter = "All"

# Apply filters for display
filtered_df = df.copy()
if selected_country_filter != "All":
    filtered_df = filter_data(filtered_df, country=selected_country_filter)
if PARAMETER_COL in df.columns and selected_parameter_filter != "All":
    filtered_df = filter_data(filtered_df, parameter=selected_parameter_filter)

st.header("📑 Filtered Data")
st.write(f"Country: **{selected_country_filter}**, Parameter: **{selected_parameter_filter}**")
st.dataframe(preview_table(filtered_df.sample(min(200, len(filtered_df)))), use_container_width=True)

# ---------- Visualizations ----------
st.header("🌐 Visualizations")

# Global map logic (safe)
if PARAMETER_COL in df.columns:
    st.subheader("Global Parameter Distribution")
    map_parameter = st.selectbox("Select Parameter for Global Map", SOLAR_FORECAST_PARAMETERS)
    if map_parameter:
        map_data_parameter = df[df[PARAMETER_COL] == map_parameter].copy()
        if not map_data_parameter.empty:
            avg_param_country = map_data_parameter.groupby(COUNTRY_COL)[VALUE_COL].mean().reset_index()
            try:
                fig_map = px.choropleth(
                    avg_param_country,
                    locations=COUNTRY_COL,
                    locationmode="country names",
                    color=VALUE_COL,
                    hover_name=COUNTRY_COL,
                    color_continuous_scale="YlOrRd",
                    title=f"Average {map_parameter} by Country"
                )
                st.plotly_chart(fig_map, use_container_width=True)
            except Exception as e:
                st.warning("Could not render choropleth. Falling back to scatter_geo.")
                st.exception(e)
                coords_df = pd.DataFrame.from_dict(country_coordinates, orient='index', columns=['lat', 'lon']).reset_index().rename(columns={'index': COUNTRY_COL})
                map_data = pd.merge(avg_param_country, coords_df, on=COUNTRY_COL, how='left')
                map_data.dropna(subset=['lat', 'lon', VALUE_COL], inplace=True)
                if not map_data.empty:
                    fig_scatter = px.scatter_geo(map_data, lat='lat', lon='lon', hover_name=COUNTRY_COL, size=VALUE_COL, color=VALUE_COL, projection="natural earth")
                    st.plotly_chart(fig_scatter, use_container_width=True)
        else:
            st.info(f"No records for parameter {map_parameter}.")
else:
    # No parameter column - aggregate VALUE_COL per country if possible
    if COUNTRY_COL in df.columns and VALUE_COL in df.columns:
        st.info("Dataset missing parameter column — showing aggregated VALUE per country.")
        agg = df.groupby(COUNTRY_COL)[VALUE_COL].mean().reset_index()
        try:
            fig_map = px.choropleth(
                agg,
                locations=COUNTRY_COL,
                locationmode="country names",
                color=VALUE_COL,
                hover_name=COUNTRY_COL,
                title=f"Average {VALUE_COL} by Country (aggregated)"
            )
            st.plotly_chart(fig_map, use_container_width=True)
        except Exception:
            st.warning("Could not render choropleth for aggregated data.")
    else:
        st.info("Not enough columns (COUNTRY_COL or VALUE_COL) to show global map.")

# Parameter-specific or fallback timeseries
st.subheader("Parameter & Time-series Views")
if PARAMETER_COL in df.columns and selected_parameter_filter != "All":
    if selected_country_filter == "All":
        try:
            fig_box = plot_parameter_distribution_boxplot(df, selected_parameter_filter)
            st.plotly_chart(fig_box, use_container_width=True)
        except Exception as e:
            st.warning("Could not draw boxplot.")
            st.exception(e)
    else:
        ts_df = filter_data(df, country=selected_country_filter, parameter=selected_parameter_filter)
        if not ts_df.empty:
            try:
                fig_ts = plot_time_series_by_country(df, selected_parameter_filter, [selected_country_filter])
                st.plotly_chart(fig_ts, use_container_width=True)
            except Exception as e:
                st.warning("Could not draw time series.")
                st.exception(e)
        else:
            st.info("No data for this country+parameter.")
# Fallback: if no parameter column, but ds/country/value present show per-country time series
elif PARAMETER_COL not in df.columns and 'ds' in df.columns and COUNTRY_COL in df.columns and VALUE_COL in df.columns:
    st.info("Parameter column missing; show VALUE time series for selected country.")
    if selected_country_filter != "All":
        tmp = df[df[COUNTRY_COL] == selected_country_filter].sort_values('ds')
        if not tmp.empty:
            fig = px.line(tmp, x='ds', y=VALUE_COL, title=f"{selected_country_filter} — {VALUE_COL} time series")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No time-series rows for this country.")
    else:
        st.info("Pick a country to see the VALUE time series.")

# ---------- Modeling & Forecasting ----------
st.header("🤖 Modeling & Forecasting")
selected_region_model = st.sidebar.selectbox("Select Continent for Modeling", all_regions)
if selected_region_model != "All":
    countries_in_region_model = country_regions.get(selected_region_model, [])
    available_countries_model = [c for c in all_countries if c in countries_in_region_model]
else:
    available_countries_model = all_countries

selected_country_model = st.sidebar.selectbox("Select Country for Modeling", available_countries_model)

if PARAMETER_COL in df.columns and all_parameters:
    selected_parameter_model = st.sidebar.selectbox(
        "Select Parameter for Modeling",
        all_parameters,
        index=all_parameters.index(TARGET_COL) if TARGET_COL in all_parameters else 0
    )
else:
    selected_parameter_model = TARGET_COL
    st.sidebar.info("No parameter column in dataset — modeling will use country-level VALUE_COL.")

forecast_horizon = st.sidebar.number_input("Forecast Horizon (days)", min_value=MIN_HORIZON, max_value=MAX_HORIZON, value=DEFAULT_HORIZON)

# Build model dataframe safely
if PARAMETER_COL in df.columns and all_parameters:
    model_df = filter_data(df, country=selected_country_model, parameter=selected_parameter_model)
else:
    if COUNTRY_COL in df.columns:
        model_df = df[df[COUNTRY_COL] == selected_country_model].copy()
    else:
        model_df = pd.DataFrame()

if model_df.empty:
    st.warning("Not enough data for modeling (check country/parameter selection and dataset).")
else:
    st.write("Modeling dataset shape:", model_df.shape)
    train_size = int(len(model_df) * 0.8)
    train_df = model_df.iloc[:train_size].copy()
    test_df = model_df.iloc[train_size:].copy()

    if st.button(f"🚀 Train {model_type} model"):
        try:
            with st.spinner(f"Training {model_type}..."):
                y_true_eval = None
                y_pred_eval = None

                # --- Random Forest ---
                if model_type == "Random Forest":
                    # If parameter-based features exist, use prepare_data_for_model, else infer RF_FEATURES
                    if PARAMETER_COL in df.columns and all_parameters:
                        rf_train_X, rf_train_y = prepare_data_for_model(train_df, selected_parameter_model, RF_FEATURES, VALUE_COL)
                        rf_test_X, rf_test_y = prepare_data_for_model(test_df, selected_parameter_model, RF_FEATURES, VALUE_COL)
                    else:
                        X_cols = [c for c in RF_FEATURES if c in train_df.columns]
                        if not X_cols or VALUE_COL not in train_df.columns:
                            st.error("Insufficient columns to train Random Forest on this dataset.")
                            raise RuntimeError("Insufficient columns for RF")
                        rf_train_X = train_df[X_cols].dropna()
                        rf_train_y = train_df.loc[rf_train_X.index, VALUE_COL]
                        rf_test_X = test_df[X_cols].dropna()
                        rf_test_y = test_df.loc[rf_test_X.index, VALUE_COL]

                    model = cached_train_random_forest(rf_train_X, rf_train_y)
                    y_pred_eval = predict_random_forest(model, rf_test_X)
                    y_true_eval = rf_test_y.values

                # --- Prophet ---
                elif model_type == "Prophet":
                    if PARAMETER_COL in df.columns and all_parameters:
                        prophet_train_df = make_prophet_frame(train_df, parameter=selected_parameter_model, target_col=VALUE_COL)
                    else:
                        if 'ds' not in train_df.columns or VALUE_COL not in train_df.columns:
                            st.error("Missing 'ds' or VALUE_COL for Prophet training.")
                            raise RuntimeError("Missing columns for Prophet")
                        prophet_train_df = train_df[['ds', VALUE_COL]].rename(columns={VALUE_COL: 'y'}).dropna()

                    prophet_model = cached_train_prophet(prophet_train_df)
                    future_periods = len(test_df) + forecast_horizon
                    forecast_results_df = forecast_prophet(prophet_model, periods=future_periods)

                    # Align evaluation
                    test_df['ds'] = pd.to_datetime(test_df['ds'], errors='coerce')
                    forecast_results_df['ds'] = pd.to_datetime(forecast_results_df['ds'])
                    merged_eval_df = pd.merge(test_df, forecast_results_df[['ds', 'yhat']], on='ds', how='inner')
                    if not merged_eval_df.empty:
                        y_true_eval = merged_eval_df[VALUE_COL].values
                        y_pred_eval = merged_eval_df['yhat'].values
                        # Plot full forecast with history
                        try:
                            history_for_plot = train_df[['ds', VALUE_COL]].rename(columns={VALUE_COL: 'y'}).sort_values('ds')
                            fig_prophet = prophet_forecast_plot(forecast_results_df, history_df=history_for_plot,
                                                               title=f"Prophet Forecast for {selected_parameter_model} ({selected_country_model})",
                                                               y_label=f"{selected_parameter_model} ({PARAMETER_UNITS.get(selected_parameter_model, '')})")
                            st.plotly_chart(fig_prophet, use_container_width=True)
                        except Exception:
                            st.info("Could not render Prophet plot.")
                    else:
                        st.warning("Could not align Prophet forecast with test data dates for evaluation.")

                # --- LSTM ---
                elif model_type == "LSTM":
                    # prepare series
                    if 'ds' not in train_df.columns or VALUE_COL not in train_df.columns:
                        st.error("Missing 'ds' or VALUE_COL for LSTM training.")
                        raise RuntimeError("Missing columns for LSTM")
                    train_df['ds'] = pd.to_datetime(train_df['ds'], errors='coerce')
                    test_df['ds'] = pd.to_datetime(test_df['ds'], errors='coerce')
                    train_df.set_index('ds', inplace=True)
                    test_df.set_index('ds', inplace=True)

                    features = [c for c in LSTM_FEATURES if c in train_df.columns]
                    if not features:
                        st.error("No numeric LSTM features found in dataset.")
                        raise RuntimeError("No LSTM features")

                    X_train_seq, y_train_seq = create_sequences(train_df[features], train_df[VALUE_COL], time_steps=10)
                    X_test_seq, y_test_seq = create_sequences(test_df[features], test_df[VALUE_COL], time_steps=10)

                    lstm_model = cached_train_lstm(X_train_seq, y_train_seq, epochs=20)
                    y_pred_eval = predict_lstm(lstm_model, X_test_seq)
                    y_true_eval = y_test_seq

                # --- Evaluation & plots ---
                if y_true_eval is not None and y_pred_eval is not None:
                    metrics = calculate_regression_metrics(y_true_eval, y_pred_eval)
                    st.subheader("📈 Model Evaluation")
                    st.json(metrics)

                    # Attempt to produce an actual vs predicted plot with dates if possible
                    try:
                        if model_type == "Prophet" and 'ds' in merged_eval_df.columns:
                            fig_eval = plot_forecast_vs_actual(merged_eval_df['ds'], y_true_eval, y_pred_eval,
                                                               title=f"{model_type} Actual vs Predicted")
                            st.plotly_chart(fig_eval, use_container_width=True)
                        elif model_type == "LSTM":
                            # For LSTM align with test_df index if possible
                            if len(test_df) >= len(y_true_eval):
                                eval_dates = test_df.index[:len(y_true_eval)]
                                fig_eval = plot_forecast_vs_actual(eval_dates, y_true_eval, y_pred_eval, title=f"{model_type} Actual vs Predicted")
                                st.plotly_chart(fig_eval, use_container_width=True)
                        elif model_type == "Random Forest":
                            # If rf_test_X has index aligned with rf_test_y
                            try:
                                idx = rf_test_X.index if 'rf_test_X' in locals() else None
                                if idx is not None:
                                    fig_eval = plot_forecast_vs_actual(idx, y_true_eval, y_pred_eval, title=f"{model_type} Actual vs Predicted")
                                    st.plotly_chart(fig_eval, use_container_width=True)
                            except Exception:
                                pass
                    except Exception:
                        st.info("Could not plot Actual vs Predicted (index alignment issue).")
                else:
                    st.info("No evaluation results to display (predictions or actuals missing).")

        except Exception as e:
            st.error("An error occurred during training/forecasting. See stack trace below.")
            st.exception(traceback.format_exc())

# ---------- About ----------
st.sidebar.header("ℹ️ About")
st.sidebar.info("""
Solar Forecasting Dashboard using NASA POWER data.
- Prefer loading pre-trained models for production deployments (faster & safer).
- Use Git LFS for large model artifacts (>100 MB) or host models externally.
""")
