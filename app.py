# app.py - Expanded to include Model Training, Forecasting, and Evaluation
# Updated to match actual functions in src/

import streamlit as st
import pandas as pd
import sys
import os
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# --- IMPORTANT: make sure you run streamlit from the project root (where app.py is)
# e.g.
# cd /path/to/solarforecastai
# streamlit run app.py

# --- Imports from your src package (must have src/__init__.py present and be run from project root)
from src.data_utils import (
    load_processed_data,
    filter_data,
    prepare_data_for_model,
    make_prophet_frame,
    get_unique_values,
    get_solar_parameters,
)
from src.visualization import (
    preview_table,
    plot_parameter_distribution_boxplot,
    plot_time_series_by_country,
    line_actual_vs_pred,
    prophet_forecast_plot,
    model_comparison_plot,
)
from src.model_utils import (
    train_random_forest_model, train_prophet_model, train_lstm_model,
    save_random_forest_model, save_prophet_model, save_lstm_model,
    load_random_forest_model, load_prophet_model, load_lstm_model,
    predict_random_forest, forecast_prophet, predict_lstm, create_sequences
)
from src.eval_utils import calculate_regression_metrics
from src.constants import (
    DATA_PATH, TARGET_COL, COUNTRY_COL, PARAMETER_COL, VALUE_COL,
    RF_FEATURES, PROPHET_COLS, LSTM_FEATURES, DEFAULT_HORIZON, MIN_HORIZON,
    MAX_HORIZON, SOLAR_FORECAST_PARAMETERS, PARAMETER_UNITS, DATE_COL
)
from src.geo import get_country_regions, get_country_coordinates


# --- Streamlit App ---
st.set_page_config(layout="wide", page_title="Global Solar Forecasting Dashboard")

st.title("Global Solar Forecasting and Monitoring Dashboard")

st.markdown("""
Welcome to the Solar Forecasting and Monitoring Dashboard.
Explore climate data and potential solar forecasting insights across different countries.
""")


# --- Data Loading ---
@st.cache_data
def load_data():
    """Loads the processed data."""
    try:
        df = load_processed_data(DATA_PATH)
        return df
    except FileNotFoundError:
        st.error(f"Error: Data file not found at {DATA_PATH}. Please ensure the processed data CSV is in the correct location.")
        return None
    except Exception as e:
        st.error(f"An error occurred while loading the data: {e}")
        return None


df = load_data()

# Get country regions (continents) and coordinates
country_regions = get_country_regions()
all_regions = ["All"] + sorted(list(country_regions.keys()))
country_coordinates = get_country_coordinates()


if df is not None:
    st.success("Data loaded successfully!")

    # --- Display Data Info ---
    st.header("Data Overview")
    st.write("First 10 rows of the processed data:")
    st.dataframe(preview_table(df))

    st.write(f"Dataset shape: {df.shape}")
    st.write(f"Columns: {df.columns.tolist()}")

    # --- Data Filtering Options ---
    st.sidebar.header("Filter Data")

    # Add Continent dropdown
    selected_region_filter = st.sidebar.selectbox("Select Continent for Filtering", all_regions)

    # Get unique countries and parameters from the loaded dataframe
    all_countries = get_unique_values(df, COUNTRY_COL)
    all_parameters = get_unique_values(df, PARAMETER_COL)

    # Filter countries based on selected continent
    if selected_region_filter != "All":
        countries_in_region = country_regions.get(selected_region_filter, [])
        available_countries_filter = [country for country in all_countries if country in countries_in_region]
    else:
        available_countries_filter = all_countries  # Show all countries if "All" continent is selected

    selected_country_filter = st.sidebar.selectbox("Select Country for Filtering", ["All"] + available_countries_filter)

    # Parameters dropdown
    selected_parameter_filter = st.sidebar.selectbox("Select Parameter for Filtering", ["All"] + all_parameters)

    # Apply filtering for display and initial exploration
    filtered_df = df.copy()
    if selected_country_filter != "All":
        filtered_df = filter_data(filtered_df, country=selected_country_filter)
    if selected_parameter_filter != "All":
        filtered_df = filter_data(filtered_df, parameter=selected_parameter_filter)

    st.header("Filtered Data")
    st.write(f"Displaying data filtered by Country: **{selected_country_filter}** and Parameter: **{selected_parameter_filter}**")
    st.dataframe(preview_table(filtered_df))

    # --- Visualizations ---
    st.header("Data Visualizations")

    # Global Map Visualization
    st.subheader("Global Parameter Distribution")
    map_parameter = st.selectbox("Select Parameter to Display on Map", SOLAR_FORECAST_PARAMETERS)

    if map_parameter:
        # Filter the original df for the selected parameter
        map_data_parameter = df[df[PARAMETER_COL] == map_parameter].copy()

        if not map_data_parameter.empty:
            # Calculate average value per country
            average_parameter_by_country = map_data_parameter.groupby(COUNTRY_COL)[VALUE_COL].mean().reset_index()

            # Build coords df from country_coordinates (dict country -> [lat, lon])
            coords_df = pd.DataFrame.from_dict(country_coordinates, orient='index', columns=['lat', 'lon']).reset_index().rename(columns={'index': COUNTRY_COL})

            # Merge average values with coordinates
            map_data = pd.merge(average_parameter_by_country, coords_df, on=COUNTRY_COL, how='left')

            # Drop rows where coordinates were not found
            map_data.dropna(subset=['lat', 'lon', VALUE_COL], inplace=True)

            if not map_data.empty:
                st.write(f"Average {map_parameter} ({PARAMETER_UNITS.get(map_parameter, 'Unknown Unit')}) by Country")

                # Create a global scatter map using Plotly Express
                fig_global_map = px.scatter_geo(
                    map_data,
                    lat='lat',
                    lon='lon',
                    hover_name=COUNTRY_COL,
                    size=VALUE_COL,
                    color=VALUE_COL,
                    projection="natural earth",
                    title=f"Average {map_parameter} by Country"
                )
                st.plotly_chart(fig_global_map, use_container_width=True)
            else:
                st.warning(f"No data with valid coordinates and values available to plot the map for {map_parameter}.")
        else:
            st.warning(f"No data available for parameter '{map_parameter}' to plot on the map.")

    # Visualization based on filtered data
    if selected_parameter_filter != "All":
        # Plot distribution for the selected parameter across all countries (if 'All' country is selected)
        if selected_country_filter == "All" and selected_parameter_filter in all_parameters:
            st.subheader(f"Distribution of {selected_parameter_filter} Across Countries")
            fig_boxplot = plot_parameter_distribution_boxplot(df, selected_parameter_filter)  # Use original df for all countries
            st.plotly_chart(fig_boxplot, use_container_width=True)

        # Plot time series for the selected parameter in the selected country (if a single country is selected)
        if selected_country_filter != "All" and selected_parameter_filter in all_parameters:
            st.subheader(f"Time Series of {selected_parameter_filter} in {selected_country_filter}")
            ts_df = filter_data(df, country=selected_country_filter, parameter=selected_parameter_filter)
            if not ts_df.empty:
                fig_timeseries = plot_time_series_by_country(df, selected_parameter_filter, [selected_country_filter])  # uses original df
                st.plotly_chart(fig_timeseries, use_container_width=True)
            else:
                st.write("No data available for the selected country and parameter.")

    # --- Model Training and Forecasting ---
    st.sidebar.header("Model Configuration")

    # Add Continent dropdown for Model Configuration
    selected_region_model = st.sidebar.selectbox("Select Continent for Modeling", all_regions)

    # Filter countries based on selected continent for modeling
    if selected_region_model != "All":
        countries_in_region_model = country_regions.get(selected_region_model, [])
        available_countries_model = [country for country in all_countries if country in countries_in_region_model]
    else:
        available_countries_model = all_countries  # Show all countries if "All" continent is selected

    model_type = st.sidebar.selectbox("Select Model Type", ["Random Forest", "Prophet", "LSTM"])
    selected_country_model = st.sidebar.selectbox("Select Country for Modeling", available_countries_model)
    # default to TARGET_COL if present in parameters, otherwise first parameter
    selected_parameter_model = st.sidebar.selectbox(
        "Select Parameter for Modeling",
        all_parameters,
        index=all_parameters.index(TARGET_COL) if TARGET_COL in all_parameters else 0
    )

    forecast_horizon = st.sidebar.number_input(
        "Forecast Horizon (days)",
        min_value=MIN_HORIZON,
        max_value=MAX_HORIZON,
        value=DEFAULT_HORIZON,
        step=1
    )

    # Filter data for modeling
    model_df = filter_data(df, country=selected_country_model, parameter=selected_parameter_model)

    if model_df.empty:
        st.warning(f"No data available for modeling for {selected_parameter_model} in {selected_country_model}.")
    else:
        st.header(f"{model_type} Model Training and Forecasting for {selected_parameter_model} in {selected_country_model}")

        # Time-based split
        train_size = int(len(model_df) * 0.8)
        train_df = model_df.iloc[:train_size].copy()
        test_df = model_df.iloc[train_size:].copy()

        st.write(f"Training data shape: {train_df.shape}")
        st.write(f"Testing data shape: {test_df.shape}")

        if st.button(f"Train {model_type} Model"):
            st.info(f"Training {model_type} model...")

            try:
                model = None
                y_true_eval = None
                y_pred_eval = None
                forecast_results_df = None
                evaluation_dates = None

                if model_type == "Random Forest":
                    rf_train_X, rf_train_y = prepare_data_for_model(
                        train_df,
                        target_parameter=selected_parameter_model,
                        features=RF_FEATURES,
                        target_col_name=VALUE_COL
                    )
                    rf_test_X, rf_test_y = prepare_data_for_model(
                        test_df,
                        target_parameter=selected_parameter_model,
                        features=RF_FEATURES,
                        target_col_name=VALUE_COL
                    )

                    if not rf_train_X.empty and not rf_test_X.empty:
                        model = train_random_forest_model(rf_train_X, rf_train_y)
                        y_true_eval = rf_test_y.values if hasattr(rf_test_y, "values") else rf_test_y
                        y_pred_eval = predict_random_forest(model, rf_test_X)
                        st.success(f"{model_type} model trained.")
                    else:
                        st.warning("Not enough data with required features for Random Forest training.")

                elif model_type == "Prophet":
                    prophet_train_df = make_prophet_frame(
                        train_df,
                        date_col=DATE_COL,
                        target_col=VALUE_COL,
                        parameter=selected_parameter_model
                    )

                    if not prophet_train_df.empty:
                        model = train_prophet_model(prophet_train_df)
                        future_periods = len(test_df) + forecast_horizon
                        forecast_results_df = forecast_prophet(model, periods=future_periods)

                        # Align actual test values with forecast for evaluation
                        if not test_df.empty and DATE_COL in test_df.columns:
                            test_df_eval = test_df.copy()
                            test_df_eval[DATE_COL] = pd.to_datetime(test_df_eval[DATE_COL], errors='coerce')
                            forecast_results_df['ds'] = pd.to_datetime(forecast_results_df['ds'])
                            test_df_eval = test_df_eval.sort_values(DATE_COL)
                            forecast_results_df = forecast_results_df.sort_values('ds')

                            merged_eval_df = pd.merge(
                                test_df_eval,
                                forecast_results_df[['ds', 'yhat']],
                                left_on=DATE_COL,
                                right_on='ds',
                                how='inner'
                            )

                            if not merged_eval_df.empty:
                                y_true_eval = merged_eval_df[VALUE_COL].values
                                y_pred_eval = merged_eval_df['yhat'].values
                                evaluation_dates = merged_eval_df[DATE_COL]
                            else:
                                st.warning("Could not align Prophet forecast with test data dates for evaluation.")
                                y_true_eval = None
                                y_pred_eval = None
                                evaluation_dates = None
                        else:
                            st.warning("Test data or date column missing for Prophet evaluation alignment.")
                            y_true_eval = None
                            y_pred_eval = None
                            evaluation_dates = None

                        st.success(f"{model_type} model trained and forecasted.")
                    else:
                        st.warning("Not enough data with 'ds' and 'y' columns for Prophet training.")

                elif model_type == "LSTM":
                    lstm_train_df = filter_data(train_df, parameter=selected_parameter_model)
                    lstm_test_df = filter_data(test_df, parameter=selected_parameter_model)

                    lstm_time_steps = 10
                    st.write(f"Using LSTM time steps: {lstm_time_steps}")

                    if not lstm_train_df.empty and not lstm_test_df.empty:
                        all_numeric_cols = lstm_train_df.select_dtypes(include=np.number).columns.tolist()
                        lstm_features_cols = [col for col in LSTM_FEATURES if col in all_numeric_cols]

                        if not lstm_features_cols:
                            st.warning("No suitable numeric features found for LSTM training based on LSTM_FEATURES and data.")
                            model = None
                            evaluation_dates = None
                        else:
                            st.write(f"Using features for LSTM: {lstm_features_cols}")
                            X_train_lstm_data = lstm_train_df[lstm_features_cols]
                            y_train_lstm_data = lstm_train_df[VALUE_COL]

                            X_test_lstm_data = lstm_test_df[lstm_features_cols]
                            y_test_lstm_data = lstm_test_df[VALUE_COL]

                            X_train_seq, y_train_seq = create_sequences(X_train_lstm_data, y_train_lstm_data, time_steps=lstm_time_steps)
                            X_test_seq, y_test_seq = create_sequences(X_test_lstm_data, y_test_lstm_data, time_steps=lstm_time_steps)

                            if X_train_seq.shape[0] > 0 and X_test_seq.shape[0] > 0:
                                model = train_lstm_model(X_train_seq, y_train_seq, epochs=50)
                                y_pred_eval = predict_lstm(model, X_test_seq)
                                y_true_eval = y_test_seq

                                start_idx = lstm_time_steps
                                end_idx = lstm_time_steps + len(y_true_eval)
                                if len(lstm_test_df) >= end_idx:
                                    evaluation_dates = lstm_test_df[DATE_COL].iloc[start_idx:end_idx].reset_index(drop=True)
                                else:
                                    evaluation_dates = None
                                    st.warning("Could not determine evaluation dates for LSTM plotting due to insufficient test data length for sequences.")

                                st.success(f"{model_type} model trained.")
                            else:
                                st.warning(f"Not enough data to create sequences for LSTM training/testing with time_steps={lstm_time_steps}.")
                                model = None
                                evaluation_dates = None
                    else:
                        st.warning("Not enough data for LSTM training/testing.")
                        evaluation_dates = None

                # --- Model Evaluation ---
                if y_true_eval is not None and y_pred_eval is not None and len(y_true_eval) > 0:
                    st.subheader("Model Evaluation on Test Data")
                    metrics = calculate_regression_metrics(y_true_eval, y_pred_eval)
                    st.write(f"**MAE:** {metrics.get('MAE', np.nan):.4f}")
                    st.write(f"**MSE:** {metrics.get('MSE', np.nan):.4f}")
                    st.write(f"**RMSE:** {metrics.get('RMSE', np.nan):.4f}")
                    st.write(f"**R-squared:** {metrics.get('R-squared', np.nan):.4f}")

                    # Plot actual vs predicted for evaluation period
                    st.subheader("Actual vs Predicted on Test Data")

                    if model_type == "Prophet" and forecast_results_df is not None:
                        prophet_history_for_plot = train_df.rename(columns={DATE_COL: 'ds', VALUE_COL: 'y'})[['ds', 'y']].sort_values('ds')
                        fig_prophet_forecast = prophet_forecast_plot(
                            forecast_results_df,
                            history_df=prophet_history_for_plot,
                            title=f"{model_type} Forecast for {selected_parameter_model} in {selected_country_model}",
                            y_label=f"{selected_parameter_model} ({PARAMETER_UNITS.get(selected_parameter_model, 'Unknown Unit')})"
                        )
                        st.plotly_chart(fig_prophet_forecast, use_container_width=True)
                    else:
                        if evaluation_dates is not None and len(evaluation_dates) == len(y_true_eval):
                            y_true_eval_series = pd.Series(y_true_eval, index=evaluation_dates)
                            y_pred_eval_series = pd.Series(y_pred_eval, index=evaluation_dates)
                            fig_actual_pred = line_actual_vs_pred(
                                y_true_eval_series,
                                y_pred_eval_series,
                                title=f"Actual vs Predicted ({model_type}) for {selected_parameter_model} in {selected_country_model} (Test Data)",
                                y_label=f"{selected_parameter_model} ({PARAMETER_UNITS.get(selected_parameter_model, 'Unknown Unit')})"
                            )
                            st.plotly_chart(fig_actual_pred, use_container_width=True)
                        else:
                            st.warning("Could not align test data dates with actual/predicted values for plotting.")

                # --- Future Forecasting Plot (for Prophet) ---
                if model_type == "Prophet" and forecast_results_df is not None:
                    st.subheader(f"{model_type} Future Forecast (Next {forecast_horizon} days)")
                    last_train_date = train_df[DATE_COL].max()
                    future_forecast_df = forecast_results_df[forecast_results_df['ds'] > last_train_date].iloc[:forecast_horizon].copy()

                    if not future_forecast_df.empty:
                        fig_future_forecast = prophet_forecast_plot(
                            future_forecast_df,
                            history_df=None,
                            title=f"{model_type} Future Forecast for {selected_parameter_model} in {selected_country_model}",
                            y_label=f"{selected_parameter_model} ({PARAMETER_UNITS.get(selected_parameter_model, 'Unknown Unit')})"
                        )
                        st.plotly_chart(fig_future_forecast, use_container_width=True)
                    else:
                        st.warning("Could not generate future forecast plot.")

            except RuntimeError as re:
                st.error(f"Dependency Error: {re}")
                st.exception(re)
            except ValueError as ve:
                st.error(f"Data Error: {ve}")
                st.exception(ve)
            except Exception as e:
                st.error(f"An unexpected error occurred during model training or forecasting: {e}")
                st.exception(e)


else:
    st.warning("Could not load data. Please check the file path and try again.")

# --- About Section ---
st.sidebar.header("About")
st.sidebar.info(
    """
    This dashboard analyzes climate data from NASA POWER for solar forecasting.
    It showcases data visualization, exploration, and machine learning model
    (Random Forest, Prophet, LSTM) capabilities.

    Data Source: NASA POWER Project (power.larc.nasa.gov)
    """
)
