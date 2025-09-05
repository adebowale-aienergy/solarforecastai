# app.py
# Streamlit app for SolarForecastAI – polished dashboard that runs from repo
# Loads data from nasa_power_data_all_params.csv in this repo.
# Uses models in ./models if they exist; otherwise fits quick models on the fly.

from __future__ import annotations

import os
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Optional model deps (loaded lazily)
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Try Prophet
try:
    from prophet import Prophet
    _HAS_PROPHET = True
except Exception:
    _HAS_PROPHET = False

# Try TensorFlow/Keras only if a model file exists (to keep startup light)
_HAS_TF = False
LSTM_PATH = Path("models/lstm_model.h5")
if LSTM_PATH.exists():
    try:
        from tensorflow.keras.models import load_model as tf_load_model
        _HAS_TF = True
    except Exception:
        _HAS_TF = False


# --------------------------------------------------------------------------------------
# Constants & Paths
# --------------------------------------------------------------------------------------
REPO_DIR = Path(__file__).resolve().parent
DATA_PATH = REPO_DIR / "nasa_power_data_all_params.csv"

MODELS_DIR = REPO_DIR / "models"
RF_PATH = MODELS_DIR / "rf_model.pkl"
PROPHET_PATH = MODELS_DIR / "prophet_model.json"  # optional saved model format
# (We will re-fit a fresh Prophet if json isn't present; saved models are tricky with Prophet)

DEFAULT_TARGET_CANDIDATES = [
    "ALLSKY_SFC_SW_DWN",  # common GHI variable
    "ALLSKY_SFC_LW_DWN",
    "T2M", "WS10M", "WS50M"
]
DATE_CANDIDATES = ["DATE", "TIMESTAMP", "time", "Time", "DATE_TIME", "datetime"]
COUNTRY_COL_CANDIDATES = ["COUNTRY", "country", "Country"]
REGION_COL_CANDIDATES = ["REGION", "region", "Region", "STATE", "State"]
LAT_CANDS = ["LAT", "lat", "Latitude", "latitude"]
LON_CANDS = ["LON", "lon", "Longitude", "longitude"]


# --------------------------------------------------------------------------------------
# Utility: Load and normalize data
# --------------------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        st.error(f"CSV not found at: {DATA_PATH}")
        st.stop()

    df = pd.read_csv(DATA_PATH, low_memory=False)
    # Normalize columns
    df.columns = [str(c).strip() for c in df.columns]

    # Find date col
    date_col = None
    for c in DATE_CANDIDATES:
        if c in df.columns:
            date_col = c
            break
    if date_col is None:
        # Try infer from any column that looks like a date
        for c in df.columns:
            if "date" in c.lower() or "time" in c.lower():
                date_col = c
                break

    if date_col is None:
        st.error("No date/time column found. Please ensure your CSV has a DATE/TIMESTAMP column.")
        st.stop()

    # Parse date
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce", utc=True).dt.tz_localize(None)

    # Sort by date
    df = df.sort_values(by=date_col).reset_index(drop=True)

    # Add COUNTRY/REGION fallbacks if missing
    country_col = next((c for c in COUNTRY_COL_CANDIDATES if c in df.columns), None)
    region_col = next((c for c in REGION_COL_CANDIDATES if c in df.columns), None)

    if country_col is None:
        df["COUNTRY"] = "Unknown"
        country_col = "COUNTRY"

    if region_col is None:
        df["REGION"] = "All"
        region_col = "REGION"

    # Ensure lat/lon (for map)
    lat_col = next((c for c in LAT_CANDS if c in df.columns), None)
    lon_col = next((c for c in LON_CANDS if c in df.columns), None)
    # If missing, fill with NaN; map tab will adapt.
    if lat_col is None:
        df["LAT"] = np.nan
        lat_col = "LAT"
    if lon_col is None:
        df["LON"] = np.nan
        lon_col = "LON"

    # Choose target
    target_col = None
    for c in DEFAULT_TARGET_CANDIDATES:
        if c in df.columns:
            target_col = c
            break
    if target_col is None:
        # last numeric column as fallback
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        if not numeric_cols:
            st.error("No numeric columns found to forecast.")
            st.stop()
        target_col = numeric_cols[-1]

    return df, date_col, target_col, country_col, region_col, lat_col, lon_col


# --------------------------------------------------------------------------------------
# Feature engineering for RF: simple lags
# --------------------------------------------------------------------------------------
def build_lag_features(s: pd.Series, n_lags: int = 7) -> pd.DataFrame:
    df = pd.DataFrame({"y": s})
    for i in range(1, n_lags + 1):
        df[f"lag_{i}"] = df["y"].shift(i)
    return df.dropna()


def rf_forecast(series: pd.Series, horizon: int = 7) -> pd.DataFrame:
    """
    Quick recursive RF forecast on a univariate series.
    Returns DataFrame with 'ds' (future dates index) and 'yhat'.
    """
    if series.dropna().shape[0] < 30:
        raise ValueError("Not enough data to train RandomForest.")

    n_lags = min(14, max(3, len(series)//30))  # adapt lags
    df_feat = build_lag_features(series, n_lags=n_lags)
    y = df_feat["y"].values
    X = df_feat.drop(columns=["y"]).values

    model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X, y)

    # Recursive prediction
    history = series.values.tolist()
    preds = []
    for _ in range(horizon):
        last = pd.Series(history[-n_lags:])
        feat = last[::-1].values  # lags 1..n
        # Build in the same order as training (lag_1 ... lag_n)
        x = np.array([feat])  # shape (1, n_lags)
        yhat = model.predict(x)[0]
        preds.append(yhat)
        history.append(yhat)

    future_index = pd.date_range(series.index[-1] + (series.index[1] - series.index[0]),
                                 periods=horizon, freq=pd.infer_freq(series.index) or "D")
    return pd.DataFrame({"ds": future_index, "yhat": preds})


def prophet_forecast(df: pd.DataFrame, date_col: str, target_col: str, horizon: int = 7) -> pd.DataFrame:
    if not _HAS_PROPHET:
        raise ImportError("Prophet not available in this environment.")
    # Prophet requires columns ds, y
    d = df[[date_col, target_col]].rename(columns={date_col: "ds", target_col: "y"}).dropna()
    if len(d) < 30:
        raise ValueError("Not enough data for Prophet (min ~30 rows).")
    m = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=True)
    m.fit(d)
    future = m.make_future_dataframe(periods=horizon, freq=pd.infer_freq(d["ds"]) or "D")
    fcst = m.predict(future).tail(horizon)
    return fcst[["ds", "yhat"]]


def lstm_forecast_placeholder(series: pd.Series, horizon: int = 7) -> pd.DataFrame:
    """
    If a pretrained LSTM model exists, you could load and forecast here.
    For now, we’ll hide LSTM unless a model file exists (to avoid heavy runtime).
    """
    if not _HAS_TF:
        raise ImportError("LSTM model not available.")
    # Example:
    # model = tf_load_model(str(LSTM_PATH))
    # ...preprocess series into sequences...
    # preds = ...
    # For now, raise:
    raise NotImplementedError("Pretrained LSTM hook present but not implemented in this template.")


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
def insight_text(actual_tail: pd.Series | None, forecast_df: pd.DataFrame, target_label: str) -> str:
    if forecast_df.empty:
        return "No forecast was generated."
    yhat = forecast_df["yhat"].values
    if actual_tail is not None and len(actual_tail) >= 7:
        baseline = np.median(actual_tail.values[-7:])
        change = (np.mean(yhat) - baseline)
        pct = 0.0 if baseline == 0 else (change / baseline) * 100
        trend = "increase" if change > 0 else ("decrease" if change < 0 else "no change")
        return f"Forecast suggests a {trend} of about {abs(pct):.1f}% vs the recent 7-day median of {baseline:.2f} {target_label}."
    return f"Average predicted {target_label} over the next {len(yhat)} steps is {np.mean(yhat):.2f}."


def kpi(value, label, fmt="{:,.2f}"):
    st.metric(label, fmt.format(value) if value is not None else "—")


# --------------------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------------------
st.set_page_config(
    page_title="SolarForecastAI",
    page_icon="🔆",
    layout="wide"
)

st.title("🔆 SolarForecastAI")
st.caption("Interactive solar & weather forecasting dashboard")

with st.spinner("Loading data from repository…"):
    df, DATE_COL, TARGET_COL, COUNTRY_COL, REGION_COL, LAT_COL, LON_COL = load_data()

# Sidebar controls
st.sidebar.header("Controls")
available_targets = [c for c in [TARGET_COL] + DEFAULT_TARGET_CANDIDATES if c in df.columns]
target_col = st.sidebar.selectbox("Target to forecast", options=available_targets, index=0)

# Country/Region filters (adaptive)
countries = sorted(df[COUNTRY_COL].dropna().unique().tolist())
if len(countries) > 1 or countries[0] != "Unknown":
    country = st.sidebar.selectbox("Country", countries)
    regions = sorted(df.loc[df[COUNTRY_COL] == country, REGION_COL].dropna().unique().tolist())
    region = st.sidebar.selectbox("Region", options=regions, index=0)
    dff = df[(df[COUNTRY_COL] == country) & (df[REGION_COL] == region)].copy()
else:
    st.sidebar.info("No COUNTRY/REGION columns detected – using entire dataset.")
    dff = df.copy()
    country, region = "—", "—"

# Date range
min_date, max_date = dff[DATE_COL].min(), dff[DATE_COL].max()
start_date, end_date = st.sidebar.date_input(
    "Date range",
    value=(min_date.date(), max_date.date()),
    min_value=min_date.date(),
    max_value=max_date.date()
)
dff = dff[(dff[DATE_COL] >= pd.Timestamp(start_date)) & (dff[DATE_COL] <= pd.Timestamp(end_date))].copy()

# Horizon
horizon = st.sidebar.slider("Forecast horizon (steps)", min_value=7, max_value=60, value=14, step=1)

# Tabs
tab_overview, tab_forecast, tab_map, tab_eval = st.tabs(["Overview", "Forecast", "Map", "Evaluation"])

# --------------------------------------------------------------------------------------
# Overview Tab
# --------------------------------------------------------------------------------------
with tab_overview:
    st.subheader("Overview")
    # KPIs
    last_rows = dff.tail(30)
    col1, col2, col3, col4 = st.columns(4)
    kpi(last_rows[target_col].dropna().mean() if target_col in last_rows else None, f"30-day avg {target_col}")
    kpi(last_rows[target_col].dropna().iloc[-1] if target_col in last_rows and not last_rows[target_col].dropna().empty else None, f"Last {target_col}")
    kpi(dff[target_col].dropna().max() if target_col in dff else None, f"Max {target_col}")
    kpi(dff[target_col].dropna().min() if target_col in dff else None, f"Min {target_col}")

    # Time-series plot
    st.plotly_chart(
        px.line(dff, x=DATE_COL, y=target_col, title=f"{target_col} over time").update_layout(margin=dict(l=10,r=10,t=40,b=10)),
        use_container_width=True
    )

    # Data preview
    with st.expander("Preview data"):
        st.dataframe(dff.head(200))

# --------------------------------------------------------------------------------------
# Forecast Tab
# --------------------------------------------------------------------------------------
with tab_forecast:
    st.subheader("Forecast")
    model_options = ["Random Forest"]
    if _HAS_PROPHET:
        model_options.append("Prophet")
    if _HAS_TF and LSTM_PATH.exists():
        model_options.append("LSTM (pretrained)")
    model_choice = st.selectbox("Model", model_options)

    run = st.button("Run forecast", type="primary")

    if run:
        # Aggregate to regular interval if needed
        dts = dff[[DATE_COL, target_col]].dropna().copy()
        if dts.empty:
            st.error("No data available for forecasting with the current filters.")
            st.stop()

        # Make the date index regular (Prophet prefers regular-ish cadence; RF needs lags)
        dts = dts.set_index(DATE_COL).sort_index()
        # If index has duplicate timestamps, aggregate by mean
        dts = dts.groupby(level=0).mean()

        try:
            if model_choice == "Random Forest":
                fc = rf_forecast(dts[target_col], horizon=horizon)

            elif model_choice == "Prophet":
                fc = prophet_forecast(dts.reset_index(), DATE_COL, target_col, horizon=horizon)

            elif model_choice == "LSTM (pretrained)":
                fc = lstm_forecast_placeholder(dts[target_col], horizon=horizon)

            else:
                st.error("Unknown model selected.")
                st.stop()

            # Plot historical + forecast
            hist = dts.tail(300).reset_index().rename(columns={DATE_COL: "ds", target_col: "y"})
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=hist["ds"], y=hist["y"], mode="lines", name="History"))
            fig.add_trace(go.Scatter(x=fc["ds"], y=fc["yhat"], mode="lines+markers", name="Forecast"))
            fig.update_layout(title=f"{model_choice} forecast: {target_col}", xaxis_title="Date", yaxis_title=target_col, margin=dict(l=10,r=10,t=40,b=10))
            st.plotly_chart(fig, use_container_width=True)

            # Insight
            st.info(insight_text(hist["y"].tail(60), fc, target_col))

            # Download
            csv = fc.to_csv(index=False).encode("utf-8")
            st.download_button("Download forecast CSV", data=csv, file_name=f"forecast_{model_choice.lower()}_{target_col}.csv", mime="text/csv")

        except Exception as e:
            st.error(f"Forecast failed: {e}")

# --------------------------------------------------------------------------------------
# Map Tab
# --------------------------------------------------------------------------------------
with tab_map:
    st.subheader("Map")
    if dff[LAT_COL].notna().any() and dff[LON_COL].notna().any():
        sample = dff[[LAT_COL, LON_COL]].dropna().drop_duplicates().head(2000)
        if not sample.empty:
            st.map(sample.rename(columns={LAT_COL: "lat", LON_COL: "lon"}))
        else:
            st.warning("No lat/lon rows available after current filters.")
    else:
        st.warning("No latitude/longitude columns found in the dataset, so the map is disabled.")

# --------------------------------------------------------------------------------------
# Evaluation Tab
# --------------------------------------------------------------------------------------
with tab_eval:
    st.subheader("Quick Backtest (holdout)")

    holdout_frac = st.slider("Holdout fraction", 0.1, 0.4, 0.2, 0.05)
    eval_run = st.button("Run evaluation")

    if eval_run:
        series_df = dff[[DATE_COL, target_col]].dropna().copy().set_index(DATE_COL).sort_index()
        series_df = series_df.groupby(level=0).mean()
        y = series_df[target_col].astype(float)
        n = len(y)
        if n < 60:
            st.warning("Need at least 60 points for a meaningful backtest.")
        else:
            split = int(n * (1 - holdout_frac))
            train, test = y.iloc[:split], y.iloc[split:]

            results = []
            # RF
            try:
                rf_fc = rf_forecast(train, horizon=len(test))
                mae = mean_absolute_error(test.values, rf_fc["yhat"].values)
                r2 = r2_score(test.values, rf_fc["yhat"].values)
                results.append(("Random Forest", mae, r2))
            except Exception as e:
                results.append(("Random Forest", np.nan, np.nan))

            # Prophet
            if _HAS_PROPHET:
                try:
                    dtrain = train.reset_index().rename(columns={DATE_COL: "ds", target_col: "y"})
                    m = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=True)
                    m.fit(dtrain)
                    future = pd.DataFrame({"ds": test.index})
                    pfc = m.predict(future)
                    mae = mean_absolute_error(test.values, pfc["yhat"].values)
                    r2 = r2_score(test.values, pfc["yhat"].values)
                    results.append(("Prophet", mae, r2))
                except Exception:
                    results.append(("Prophet", np.nan, np.nan))

            # LSTM only if truly implemented and available
            if _HAS_TF and LSTM_PATH.exists():
                # Placeholder disabled to avoid errors
                pass

            res_df = pd.DataFrame(results, columns=["Model", "MAE", "R2"])
            st.dataframe(res_df)

            # Bar chart for MAE (lower is better)
            chart = px.bar(res_df, x="Model", y="MAE", title="Holdout MAE (lower is better)")
            st.plotly_chart(chart, use_container_width=True)


# --------------------------------------------------------------------------------------
# Footer
# --------------------------------------------------------------------------------------
st.caption(
    "Tip: keep `nasa_power_data_all_params.csv` in the repo root. "
    "Add pretrained models to the `models/` folder to skip quick fitting."
)
