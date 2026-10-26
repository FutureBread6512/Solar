import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import timedelta, datetime
import warnings

from openstef_core.datasets import TimeSeriesDataset
from openstef_core.types import LeadTime, Q
from openstef_models.presets import ForecastingWorkflowConfig, create_forecasting_workflow

warnings.filterwarnings("ignore")

# ==========================================
# НАСТРОЙКА СТРАНИЦЫ STREAMLIT
# ==========================================
st.set_page_config(page_title="Solar Forecast ML", layout="wide")


# ==========================================
# 1. ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ
# ==========================================
@st.cache_data
def load_custom_data(csv_path: str) -> TimeSeriesDataset:
    raw = pd.read_csv(csv_path, parse_dates=["timestamp_utc"], encoding="utf-8-sig")
    raw = raw.set_index("timestamp_utc").sort_index()
    if raw.index.tz is None:
        raw.index = raw.index.tz_localize("UTC")
    raw.index.name = "timestamp"

    df = raw.rename(columns={
        "power_actual_kw": "solar_generation",
        "ambient_temperature_c": "temperature_2m",
        "wind_speed_ms": "wind_speed_10m",
        "poa_irradiance_wm2": "shortwave_radiation",
    })

    df = df.select_dtypes(include=[np.number])
    df["shortwave_radiation"] = df["shortwave_radiation"].fillna(0.0)
    df["relative_humidity_2m"] = 70.0
    df["surface_pressure"] = 1013.0
    df["solar_generation"] = df["solar_generation"].clip(lower=0.0)
    df = df.dropna(subset=["solar_generation"])

    return TimeSeriesDataset(df, sample_interval=timedelta(hours=1))


# ==========================================
# 2. ОСНОВНОЙ БЛОК ВЫПОЛНЕНИЯ
# ==========================================
def main():
    st.title("☀️ Прогноз выработки солнечной электроэнергии")
    st.markdown("Сравнение ML-модели (XGBoost) и наивного прогноза за 3–9 июня 2024 года.")

    with st.spinner("Загрузка данных и подготовка..."):
        dataset = load_custom_data("2107_hourly_apr_jul_2024.csv")

        # Наивный прогноз
        df_naive = dataset.data[["solar_generation", "shortwave_radiation"]].copy()
        df_naive["solar_naive"] = df_naive["solar_generation"].shift(24)
        df_naive.loc[df_naive["shortwave_radiation"] < 50, "solar_naive"] = 0.0

    # Обучение ML-модели
    with st.spinner("Обучение ML-модели (XGBoost)... Это может занять около минуты."):
        train_end = datetime.fromisoformat("2024-06-01T00:00:00Z")
        predict_end = datetime.fromisoformat("2024-06-10T00:00:00Z")

        train_dataset = dataset.filter_by_range(start=None, end=train_end)
        predict_dataset = dataset.filter_by_range(
            start=train_end - timedelta(days=14),
            end=predict_end,
        )

        workflow = create_forecasting_workflow(
            config=ForecastingWorkflowConfig(
                model_id="solar_forecast",
                model="xgboost",
                horizons=[LeadTime.from_string("PT1H")],
                quantiles=[Q(0.5)],
                target_column="solar_generation",
                temperature_column="temperature_2m",
                relative_humidity_column="relative_humidity_2m",
                wind_speed_column="wind_speed_10m",
                radiation_column="shortwave_radiation",
                pressure_column="surface_pressure",
                verbosity=0,
                mlflow_storage=None,
            )
        )
        workflow.fit(train_dataset)
        ml_forecast = workflow.predict(predict_dataset, forecast_start=train_end)

    # ==========================================
    # 3. ФИЗИЧЕСКАЯ ОЧИСТКА И ПОДГОТОВКА
    # ==========================================
    results = dataset.data[["solar_generation", "shortwave_radiation"]].copy()
    results["solar_naive"] = df_naive["solar_naive"]
    results["solar_ml"] = ml_forecast.data['quantile_P50']

    results["solar_ml"] = results["solar_ml"].clip(lower=0.0)
    results.loc[results["shortwave_radiation"] < 50, "solar_ml"] = 0.0

    start_plot = pd.Timestamp("2024-06-03 00:00:00", tz="UTC")
    end_plot = pd.Timestamp("2024-06-09 23:59:59", tz="UTC")
    results_june = results.loc[start_plot:end_plot].dropna()

    # ==========================================
    # 4. РАСЧЕТ МЕТРИК (ТОЛЬКО ДНЕМ)
    # ==========================================
    daytime_mask = results_june["solar_generation"] > 0
    y_true = results_june["solar_generation"][daytime_mask]
    y_naive = results_june["solar_naive"][daytime_mask]
    y_ml = results_june["solar_ml"][daytime_mask]

    # Ошибки моделей
    mae_naive = np.mean(np.abs(y_true - y_naive))
    mae_ml = np.mean(np.abs(y_true - y_ml))
    rmse_naive = np.sqrt(np.mean((y_true - y_naive) ** 2))
    rmse_ml = np.sqrt(np.mean((y_true - y_ml) ** 2))
    max_naive = np.max(np.abs(y_true - y_naive))
    max_ml = np.max(np.abs(y_true - y_ml))

    # Статистика по ФАКТУ (для добавления в таблицу)
    mean_actual = np.mean(y_true)  # Средняя выработка
    max_actual = np.max(y_true)  # Пиковая выработка

    improvement = (mae_naive - mae_ml) / mae_naive * 100

    # Вывод красивых плашек с метриками в Streamlit
    col1, col2, col3 = st.columns(3)
    col1.metric(label="MAE (Наивный прогноз)", value=f"{mae_naive:.2f} кВт")
    col2.metric(label="MAE (ML XGBoost)", value=f"{mae_ml:.2f} кВт", delta=f"-{improvement:.1f}% (ошибка ниже)",
                delta_color="inverse")
    col3.metric(label="Улучшение точности", value=f"{improvement:.1f}%")

    # ==========================================
    # 5. ПОСТРОЕНИЕ ГРАФИКА + ТАБЛИЦА
    # ==========================================
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.15,
        subplot_titles=(
            "Сравнение реальной выработки и прогнозов",
            "Сравнительный анализ ошибок (Только дневное время)"
        ),
        row_heights=[0.65, 0.35],
        specs=[[{"type": "xy"}], [{"type": "domain"}]]
    )

    # --- СТРОКА 1: ГРАФИК ---
    fig.add_trace(go.Scatter(
        x=results_june.index, y=results_june["solar_generation"],
        mode="lines", name="Факт (Реальность)",
        line=dict(color="#000000", width=3)
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=results_june.index, y=results_june["solar_ml"],
        mode="lines", name="ML-прогноз (XGBoost)",
        line=dict(color="#059669", width=2.5)
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=results_june.index, y=results_june["solar_naive"],
        mode="lines", name="Наивный прогноз",
        line=dict(color="#ea580c", width=2, dash="dash")
    ), row=1, col=1)

    # --- СТРОКА 2: ТАБЛИЦА (ДОБАВЛЕНЫ КОЛОНКИ ФАКТА) ---
    fig.add_trace(
        go.Table(
            header=dict(
                values=[
                    "<b>Модель</b>",
                    "<b>MAE (кВт)</b>",
                    "<b>RMSE (кВт)</b>",
                    "<b>Макс. ошибка (кВт)</b>",
                    "<b>Ср. факт (кВт)</b>",
                    "<b>Макс. факт (кВт)</b>"
                ],
                fill_color='#f3f4f6',
                align='center',
                font=dict(size=12, color='black')
            ),
            cells=dict(
                values=[
                    ["Наивный прогноз", "ML (XGBoost)"],
                    [f"{mae_naive:.2f}", f"{mae_ml:.2f}"],
                    [f"{rmse_naive:.2f}", f"{rmse_ml:.2f}"],
                    [f"{max_naive:.2f}", f"{max_ml:.2f}"],
                    [f"{mean_actual:.2f}", f"{mean_actual:.2f}"],
                    [f"{max_actual:.2f}", f"{max_actual:.2f}"]
                ],
                fill_color='#ffffff',
                align='center',
                font=dict(size=12, color='black')
            )
        ),
        row=2, col=1
    )

    # Настройка общего вида
    fig.update_layout(
        height=800,
        template="plotly_white",
        showlegend=True,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        margin=dict(l=40, r=40, t=60, b=40)
    )

    fig.update_yaxes(title_text="кВт", row=1, col=1)

    # Вывод на сайт
    st.plotly_chart(fig, width='stretch')


if __name__ == "__main__":
    main()