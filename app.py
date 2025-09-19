# app.py - Wide-format dataset compatible (parameters as columns)
import streamlit as st
import pandas as pd
import os
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import traceback

# --- Imports from src package ---
from src.data_utils import (
    load_processed_data,  # loads a CSV and normalizes date -> 'ds'
    filter_data,          # should filter by country (and date range)
    create_sequences      # sequence builder used by LSTM (if implemented in src)
)
from src.visualization import (
    preview_table,
    plot_parameter_distribution_boxplot,  # may rely on wide-format
    plot_time_series_by_country,
    line_actual_vs_pred,
    prophet_forecast_plot
)
from src.model_utils import (
    train_random_forest_model, train_prophet_model, train_lstm_model,
    predict_random_forest, forecast_prophet, predict_lstm
)
from src.eval_utils import calculate_regression_metrics
from src.constants import (
    DATE_COL, TARGET_COL, COUNTRY_COL,
    RF_FEATURES, LSTM_FEATURES, DEFAULT_HORIZON, MIN_HORIZON,
    MAX_HORIZON, SOLAR_FORECAST_PARAMETERS, PARAMETER_UNITS,
    FEATURES_DATA_PATH, CLEAN_DATA_PATH
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
    fig.update_layout(title=title, xaxis_title="Date", yaxis_title=(y_label or "Value"))
    return fig

# ---------- Caching ----------
@st.cache_data
def cached_load_data_for(model_type: str | None = None):
    """
    Load the correct processed dataset depending on model type.
    Uses load_processed_data() in src.data_utils to normalize date -> 'ds'.
    """
    try:
        # prefer explicit paths from constants if available
        if model_type == "Random Forest":
            path = FEATURES_DATA_PATH if 'FEATURES_DATA_PATH' in globals() else os.path.join("data", "features_data.csv")
        else:
            path = CLEAN_DATA_PATH if 'CLEAN_DATA_PATH' in globals() else os.path.join("data", "clean_data.csv")

        df = load_processed_data(path)
        df["_source_path"] = path
        return df
    except Exception:
        raise

@st.cache_resource
def cached_train_random_forest(X, y):
    return train_random_forest_model(X, y)

@st.cache_resource
def cached_train_prophet(df_prophet):
    return train_prophet_model(df_prophet)

@st.cache_resource
def cached_train_lstm(X, y, epochs=20):
    return train_lstm_model(X, y, epochs=epochs)

# ---------- Sidebar: configuration & info ----------
st.sidebar.header("⚙️ Configuration")
model_type = st.sidebar.selectbox("Select Model Type", ["Random Forest", "Prophet", "LSTM"])

with st.sidebar.expander("ℹ️ Parameter descriptions", expanded=False):
    st.markdown(
        "- **ALLSKY_SFC_SW_DWN**: Downward shortwave radiation at surface\n"
        "- **T2M**: Air temperature at 2m\n"
        "- **WS2M**: Wind speed at 2m\n"
        "- **RH2M**: Relative humidity at 2m\n\n"
        "If your dataset contains additional parameter columns, they will appear in the parameter dropdown."
    )

with st.sidebar.expander("📈 Metric guide", expanded=False):
    st.markdown(
        "- **MAE**: Mean Absolute Error (lower is better)\n"
        "- **MSE**: Mean Squared Error (lower is better)\n"
        "- **RMSE**: Root Mean Squared Error (lower is better)\n"
        "- **R-squared**: closer to 1 is better"
    )

# ---------- Load data ----------
try:
    df = cached_load_data_for(model_type)
except Exception as e:
    st.error(f"Failed to load data for model_type={model_type}: {e}")
    st.exception(traceback.format_exc())
    st.stop()

# ---------- Basic diagnostics ----------
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

# ---------- Data preview (sample) ----------
st.header("📊 Data Overview")
n_sample = min(200, len(df))
if n_sample > 0:
    st.dataframe(preview_table(df.sample(n_sample)), use_container_width=True)
st.write(f"Shape: {df.shape}")

# ---------- Filters ----------
st.sidebar.header("Filters")
selected_region_filter = st.sidebar.selectbox("Select Continent", all_regions)

# countries present
all_countries = sorted(df[COUNTRY_COL].dropna().unique().tolist()) if COUNTRY_COL in df.columns else []

# parameters: from SOLAR_FORECAST_PARAMETERS but only keep those present as columns
available_parameters = [p for p in SOLAR_FORECAST_PARAMETERS if p in df.columns]
# also include TARGET_COL if it's in the dataframe and not in SOLAR_FORECAST_PARAMETERS
if TARGET_COL and TARGET_COL in df.columns and TARGET_COL not in available_parameters:
    available_parameters.insert(0, TARGET_COL)

if selected_region_filter != "All":
    countries_in_region = country_regions.get(selected_region_filter, [])
    available_countries_filter = [c for c in all_countries if c in countries_in_region]
else:
    available_countries_filter = all_countries

selected_country_filter = st.sidebar.selectbox("Select Country (display)", ["All"] + available_countries_filter)
if available_parameters:
    selected_parameter_filter = st.sidebar.selectbox("Select Parameter (display)", ["All"] + available_parameters,
                                                     index=0 if TARGET_COL in available_parameters else 0)
else:
    selected_parameter_filter = "All"

# apply filters for display
filtered_df = df.copy()
if selected_country_filter != "All":
    filtered_df = filter_data(filtered_df, country=selected_country_filter)  # expects country filter only
if selected_parameter_filter != "All" and selected_parameter_filter in df.columns:
    # keep only date, country, and the parameter column to simplify display
    cols = [c for c in [DATE_COL, COUNTRY_COL, selected_parameter_filter] if c in filtered_df.columns]
    filtered_df = filtered_df[cols].copy()

st.header("📑 Filtered Data")
st.write(f"Country: **{selected_country_filter}**, Parameter: **{selected_parameter_filter}**")
st.dataframe(preview_table(filtered_df.sample(min(200, len(filtered_df)))), use_container_width=True)

# ---------- Visualizations ----------
st.header("🌐 Visualizations")

# Global map: aggregate the chosen parameter across countries
st.subheader("Global Parameter Distribution")
if available_parameters:
    map_parameter = st.selectbox("Select Parameter for Global Map", available_parameters, index=0 if TARGET_COL in available_parameters else 0)
    if map_parameter:
        # aggregate mean per country
        if COUNTRY_COL in df.columns and map_parameter in df.columns:
            avg_param_country = df.groupby(COUNTRY_COL)[map_parameter].mean().reset_index()
            try:
                fig_map = px.choropleth(
                    avg_param_country,
                    locations=COUNTRY_COL,
                    locationmode="country names",
                    color=map_parameter,
                    hover_name=COUNTRY_COL,
                    color_continuous_scale="YlOrRd",
                    title=f"Average {map_parameter} by Country"
                )
                st.plotly_chart(fig_map, use_container_width=True)
            except Exception:
                # fallback to scatter_geo if choropleth fails
                coords_df = pd.DataFrame.from_dict(country_coordinates, orient='index', columns=['lat', 'lon']).reset_index().rename(columns={'index': COUNTRY_COL})
                map_data = pd.merge(avg_param_country, coords_df, on=COUNTRY_COL, how='left')
                map_data.dropna(subset=['lat', 'lon'], inplace=True)
                if not map_data.empty:
                    fig_scatter = px.scatter_geo(map_data, lat='lat', lon='lon', hover_name=COUNTRY_COL, size=map_parameter, color=map_parameter, projection="natural earth")
                    st.plotly_chart(fig_scatter, use_container_width=True)
        else:
            st.info("Country or parameter column missing for the selected parameter.")
else:
    st.info("No parameter columns available in this dataset to plot.")

# Parameter-specific or fallback timeseries
st.subheader("Parameter & Time-series Views")
if selected_parameter_filter != "All" and selected_parameter_filter in df.columns:
    if selected_country_filter == "All":
        try:
            fig_box = plot_parameter_distribution_boxplot(df, selected_parameter_filter)
            st.plotly_chart(fig_box, use_container_width=True)
        except Exception:
            st.warning("Could not draw boxplot for parameter distribution.")
    else:
        ts_df = df[(df[COUNTRY_COL] == selected_country_filter) & (selected_parameter_filter in df.columns)]
        if not ts_df.empty:
            try:
                fig_ts = px.line(ts_df.sort_values(DATE_COL), x='ds' if 'ds' in ts_df.columns else DATE_COL, y=selected_parameter_filter, title=f"{selected_parameter_filter} in {selected_country_filter}")
                st.plotly_chart(fig_ts, use_container_width=True)
            except Exception:
                st.warning("Could not draw time series plot.")
        else:
            st.info("No data for this country+parameter.")
# fallback: if no parameter selected, show country-level timeseries for TARGET_COL if available
elif selected_parameter_filter == "All" and TARGET_COL in df.columns:
    st.info(f"No parameter selected; showing {TARGET_COL} time series for selected country when available.")
    if selected_country_filter != "All":
        tmp = df[df[COUNTRY_COL] == selected_country_filter]
        if not tmp.empty and 'ds' in tmp.columns and TARGET_COL in tmp.columns:
            fig = px.line(tmp.sort_values('ds'), x='ds', y=TARGET_COL, title=f"{selected_country_filter} — {TARGET_COL} time series")
            st.plotly_chart(fig, use_container_width=True)

# ---------- Modeling & Forecasting ----------
st.header("🤖 Modeling & Forecasting")
selected_region_model = st.sidebar.selectbox("Select Continent for Modeling", all_regions)
if selected_region_model != "All":
    countries_in_region_model = country_regions.get(selected_region_model, [])
    available_countries_model = [c for c in all_countries if c in countries_in_region_model]
else:
    available_countries_model = all_countries

selected_country_model = st.sidebar.selectbox("Select Country for Modeling", available_countries_model)

# parameter selection for modeling (choose from available parameters)
if available_parameters:
    selected_parameter_model = st.sidebar.selectbox("Select Parameter for Modeling", available_parameters, index=0 if TARGET_COL in available_parameters else 0)
else:
    selected_parameter_model = TARGET_COL if TARGET_COL in df.columns else None
    st.sidebar.info("No parameter columns detected — modeling will use available numeric columns for the chosen country.")

forecast_horizon = st.sidebar.number_input("Forecast Horizon (days)", min_value=MIN_HORIZON, max_value=MAX_HORIZON, value=DEFAULT_HORIZON)

# Build model_df: select country and parameter column (target)
if selected_country_model and COUNTRY_COL in df.columns:
    model_df = df[df[COUNTRY_COL] == selected_country_model].copy()
    if selected_parameter_model and selected_parameter_model in model_df.columns:
        # keep ds, country, target
        keep_cols = [c for c in [DATE_COL, 'ds', COUNTRY_COL, selected_parameter_model] if c in model_df.columns]
        model_df = model_df[keep_cols].copy()
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

                # ---- RANDOM FOREST ----
                if model_type == "Random Forest":
                    # Build X, y directly from selected features if possible
                    X_cols = [c for c in RF_FEATURES if c in train_df.columns and c != selected_parameter_model]
                    # ensure target present
                    if selected_parameter_model not in train_df.columns:
                        st.error("Selected parameter not present in training data for Random Forest.")
                        raise RuntimeError("Missing target column for RF")
                    rf_train_X = train_df[X_cols].dropna()
                    rf_train_y = train_df.loc[rf_train_X.index, selected_parameter_model]
                    rf_test_X = test_df[X_cols].dropna()
                    rf_test_y = test_df.loc[rf_test_X.index, selected_parameter_model]

                    if rf_train_X.shape[0] == 0 or rf_test_X.shape[0] == 0:
                        st.error("Insufficient rows after dropping NaNs for RF training/testing.")
                        raise RuntimeError("Insufficient RF data")

                    rf_model = cached_train_random_forest(rf_train_X, rf_train_y)
                    y_pred_eval = predict_random_forest(rf_model, rf_test_X)
                    y_true_eval = rf_test_y.values
                    eval_index = rf_test_X.index

                # ---- PROPHET ----
                elif model_type == "Prophet":
                    if 'ds' not in train_df.columns:
                        # make sure DATE_COL exists and convert to ds
                        if DATE_COL in train_df.columns:
                            train_df['ds'] = pd.to_datetime(train_df[DATE_COL], errors='coerce')
                            test_df['ds'] = pd.to_datetime(test_df[DATE_COL], errors='coerce')
                        else:
                            st.error("No date column available for Prophet.")
                            raise RuntimeError("Missing date for Prophet")

                    if selected_parameter_model not in train_df.columns:
                        st.error("Selected parameter not available for Prophet training.")
                        raise RuntimeError("Missing target column for Prophet")

                    prophet_train_df = train_df[['ds', selected_parameter_model]].rename(columns={selected_parameter_model: 'y'}).dropna()
                    prophet_model = cached_train_prophet(prophet_train_df)
                    future_periods = len(test_df) + forecast_horizon
                    forecast_results_df = forecast_prophet(prophet_model, periods=future_periods)

                    # Align for evaluation
                    test_df['ds'] = pd.to_datetime(test_df['ds'], errors='coerce')
                    forecast_results_df['ds'] = pd.to_datetime(forecast_results_df['ds'])
                    merged_eval_df = pd.merge(test_df, forecast_results_df[['ds', 'yhat']], on='ds', how='inner')
                    if not merged_eval_df.empty:
                        y_true_eval = merged_eval_df[selected_parameter_model].values
                        y_pred_eval = merged_eval_df['yhat'].values
                        eval_dates = merged_eval_df['ds']
                        # show full forecast
                        try:
                            history_for_plot = prophet_train_df.sort_values('ds')
                            fig_prophet = prophet_forecast_plot(forecast_results_df, history_df=history_for_plot,
                                                               title=f"Prophet Forecast for {selected_parameter_model} ({selected_country_model})",
                                                               y_label=f"{selected_parameter_model} ({PARAMETER_UNITS.get(selected_parameter_model, '')})")
                            st.plotly_chart(fig_prophet, use_container_width=True)
                        except Exception:
                            st.info("Could not render Prophet plot.")
                    else:
                        st.warning("Could not align Prophet forecast with test data dates for evaluation.")
                        eval_dates = None

                # ---- LSTM ----
                elif model_type == "LSTM":
                    # require ds and parameter column
                    if 'ds' not in train_df.columns and DATE_COL in train_df.columns:
                        train_df['ds'] = pd.to_datetime(train_df[DATE_COL], errors='coerce')
                        test_df['ds'] = pd.to_datetime(test_df[DATE_COL], errors='coerce')

                    if selected_parameter_model not in train_df.columns:
                        st.error("Selected parameter not available for LSTM training.")
                        raise RuntimeError("Missing target column for LSTM")

                    train_df.set_index('ds', inplace=True)
                    test_df.set_index('ds', inplace=True)

                    features = [c for c in LSTM_FEATURES if c in train_df.columns and c != selected_parameter_model]
                    if not features:
                        st.error("No LSTM input features found in dataset.")
                        raise RuntimeError("No LSTM features")

                    X_train_seq, y_train_seq = create_sequences(train_df[features], train_df[selected_parameter_model], time_steps=10)
                    X_test_seq, y_test_seq = create_sequences(test_df[features], test_df[selected_parameter_model], time_steps=10)
                    if X_train_seq.shape[0] == 0 or X_test_seq.shape[0] == 0:
                        st.error("Insufficient sequence data for LSTM training/testing.")
                        raise RuntimeError("Insufficient LSTM sequence data")

                    lstm_model = cached_train_lstm(X_train_seq, y_train_seq, epochs=20)
                    y_pred_eval = predict_lstm(lstm_model, X_test_seq)
                    y_true_eval = y_test_seq
                    # eval dates (align with first len(y_true_eval) rows of test_df index)
                    eval_dates = test_df.index[:len(y_true_eval)]

                # --- Evaluation ---
                if y_true_eval is not None and y_pred_eval is not None:
                    metrics = calculate_regression_metrics(y_true_eval, y_pred_eval)
                    st.subheader("📈 Model Evaluation")
                    st.json(metrics)

                    # Plot actual vs predicted
                    try:
                        if model_type == "Prophet" and 'eval_dates' in locals() and eval_dates is not None:
                            fig_eval = plot_forecast_vs_actual(eval_dates, y_true_eval, y_pred_eval, title=f"{model_type} Actual vs Predicted", y_label=selected_parameter_model)
                            st.plotly_chart(fig_eval, use_container_width=True)
                        elif model_type == "LSTM" and 'eval_dates' in locals():
                            fig_eval = plot_forecast_vs_actual(eval_dates, y_true_eval, y_pred_eval, title=f"{model_type} Actual vs Predicted", y_label=selected_parameter_model)
                            st.plotly_chart(fig_eval, use_container_width=True)
                        elif model_type == "Random Forest":
                            idx = eval_index if 'eval_index' in locals() else None
                            if idx is not None:
                                fig_eval = plot_forecast_vs_actual(idx, y_true_eval, y_pred_eval, title=f"{model_type} Actual vs Predicted", y_label=selected_parameter_model)
                                st.plotly_chart(fig_eval, use_container_width=True)
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
- For production, prefer loading pre-trained models for fast inference.
- Use Git LFS / external storage for large model artifacts (>100 MB).
""")
