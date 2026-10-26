import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import timedelta, datetime
import warnings
import os
import webbrowser
import time
import xgboost as xgb

from openstef_core.datasets import TimeSeriesDataset
from openstef_core.types import LeadTime, Q
from openstef_models.presets import ForecastingWorkflowConfig, create_forecasting_workflow

warnings.filterwarnings("ignore")

# ==========================================
# 1. ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ
# ==========================================
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

print("Загрузка данных...")
dataset = load_custom_data("2107_hourly_apr_jul_2024.csv")

# Определяем периоды
train_end = datetime.fromisoformat("2024-06-01T00:00:00Z")
predict_end = datetime.fromisoformat("2024-06-10T00:00:00Z")
train_dataset = dataset.filter_by_range(start=None, end=train_end)

# ==========================================
# МЕТОД 1: ИСТОРИЧЕСКАЯ (Persistence)
# ==========================================
print("Расчет М1 (Историческая)...")
df_naive = dataset.data[["solar_generation", "shortwave_radiation"]].copy()
df_naive["m1_historical"] = df_naive["solar_generation"].shift(24)
df_naive.loc[df_naive["shortwave_radiation"] < 50, "m1_historical"] = 0.0

# ==========================================
# МЕТОД 2: КЛИМАТИЧЕСКАЯ (Только погода, без лагов)
# ==========================================
print("Обучение М2 (Климатическая)...")
m2_features = ["shortwave_radiation", "temperature_2m", "wind_speed_10m"]
# Берем только погодные признаки
X_train_m2 = train_dataset.data[m2_features].fillna(0)
y_train_m2 = train_dataset.data["solar_generation"]

model_m2 = xgb.XGBRegressor(n_estimators=50, max_depth=4, random_state=42)
model_m2.fit(X_train_m2, y_train_m2)

# Предсказание для М2 на всем датасете
X_all_m2 = dataset.data[m2_features].fillna(0)
m2_predictions = model_m2.predict(X_all_m2)
m2_predictions = np.clip(m2_predictions, 0, None) # Убираем отрицательные

# ==========================================
# МЕТОД 3: КОМБИНИРОВАННАЯ (OpenSTEF XGBoost)
# ==========================================
print("Обучение М3 (Комбинированная - OpenSTEF)...")
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
# 4. СБОР РЕЗУЛЬТАТОВ И ОЧИСТКА
# ==========================================
results = dataset.data[["solar_generation", "shortwave_radiation"]].copy()
results["m1_historical"] = df_naive["m1_historical"]
results["m2_climatic"] = m2_predictions
results["m3_combined"] = ml_forecast.data['quantile_P50']

# Физическая очистка: ночью генерация = 0
for col in ["m2_climatic", "m3_combined"]:
    results[col] = results[col].clip(lower=0.0)
    results.loc[results["shortwave_radiation"] < 50, col] = 0.0

start_plot = pd.Timestamp("2024-06-03 00:00:00", tz="UTC")
end_plot = pd.Timestamp("2024-06-09 23:59:59", tz="UTC")
results_june = results.loc[start_plot:end_plot].dropna()

# ==========================================
# 5. РАСЧЕТ МЕТРИК ДЛЯ ВСЕХ ТРЕХ МЕТОДОВ
# ==========================================
daytime_mask = results_june["solar_generation"] > 0
y_true = results_june["solar_generation"][daytime_mask]

def calc_metrics(y_pred):
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    max_err = np.max(np.abs(y_true - y_pred))
    return mae, rmse, max_err

mae_m1, rmse_m1, max_m1 = calc_metrics(results_june["m1_historical"][daytime_mask])
mae_m2, rmse_m2, max_m2 = calc_metrics(results_june["m2_climatic"][daytime_mask])
mae_m3, rmse_m3, max_m3 = calc_metrics(results_june["m3_combined"][daytime_mask])

mean_actual = np.mean(y_true)
max_actual = np.max(y_true)

# ==========================================
# 6. ПОСТРОЕНИЕ ТРЕХ ГРАФИКОВ И ТАБЛИЦЫ
# ==========================================
fig = make_subplots(
    rows=4, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.08,
    subplot_titles=(
        "<b>1. М1: Историческая модель vs Факт</b>",
        "<b>2. М2: Климатическая модель (только погода) vs Факт</b>",
        "<b>3. М3: Комбинированная модель (погода + история) vs Факт</b>",
        "<b>4. Сравнительный анализ ошибок (Только дневное время)</b>"
    ),
    row_heights=[0.2, 0.2, 0.2, 0.4],
    specs=[[{"type": "xy"}], [{"type": "xy"}], [{"type": "xy"}], [{"type": "domain"}]]
)

# --- ГРАФИК 1: ФАКТ vs М1 ---
fig.add_trace(go.Scatter(x=results_june.index, y=results_june["solar_generation"],
    mode="lines", name="Факт", line=dict(color="#000000", width=3)), row=1, col=1)
fig.add_trace(go.Scatter(x=results_june.index, y=results_june["m1_historical"],
    mode="lines", name="М1 (Историческая)", line=dict(color="#ea580c", width=2, dash="dash")), row=1, col=1)

# --- ГРАФИК 2: ФАКТ vs М2 ---
fig.add_trace(go.Scatter(x=results_june.index, y=results_june["solar_generation"],
    mode="lines", name="Факт", line=dict(color="#000000", width=3), showlegend=False), row=2, col=1)
fig.add_trace(go.Scatter(x=results_june.index, y=results_june["m2_climatic"],
    mode="lines", name="М2 (Климатическая)", line=dict(color="#8b5cf6", width=2, dash="dot")), row=2, col=1)

# --- ГРАФИК 3: ФАКТ vs М3 ---
fig.add_trace(go.Scatter(x=results_june.index, y=results_june["solar_generation"],
    mode="lines", name="Факт", line=dict(color="#000000", width=3), showlegend=False), row=3, col=1)
fig.add_trace(go.Scatter(x=results_june.index, y=results_june["m3_combined"],
    mode="lines", name="М3 (Комбинированная)", line=dict(color="#059669", width=2.5)), row=3, col=1)

# --- ГРАФИК 4: ТАБЛИЦА ---
fig.add_trace(
    go.Table(
        header=dict(
            values=["<b>Метод</b>", "<b>MAE (кВт)</b>", "<b>RMSE (кВт)</b>", "<b>Макс. ошибка (кВт)</b>", "<b>Ср. факт (кВт)</b>", "<b>Макс. факт (кВт)</b>"],
            fill_color='#f3f4f6', align='center', font=dict(size=14, color='black')
        ),
        cells=dict(
            values=[
                ["М1. Историческая", "М2. Климатическая", "<b>М3. Комбинированная</b>"],
                [f"{mae_m1:.2f}", f"{mae_m2:.2f}", f"<b>{mae_m3:.2f}</b>"],
                [f"{rmse_m1:.2f}", f"{rmse_m2:.2f}", f"<b>{rmse_m3:.2f}</b>"],
                [f"{max_m1:.2f}", f"{max_m2:.2f}", f"<b>{max_m3:.2f}</b>"],
                [f"{mean_actual:.2f}", f"{mean_actual:.2f}", f"{mean_actual:.2f}"],
                [f"{max_actual:.2f}", f"{max_actual:.2f}", f"{max_actual:.2f}"]
            ],
            fill_color=['#ffffff', '#ffffff', '#e8f5e9'], # Выделяем М3 зеленым
            align='center', font=dict(size=14, color='black')
        )
    ), row=4, col=1
)

# Настройка оформления
fig.update_layout(
    title_text="<b>Сравнение трех методов прогнозирования выработки</b>",
    title_x=0.5, title_font=dict(size=24), height=1400,
    template="plotly_white", showlegend=True,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5, font=dict(size=16)),
    margin=dict(l=60, r=60, t=120, b=60)
)

fig.update_xaxes(title_font=dict(size=16), tickfont=dict(size=14))
fig.update_yaxes(title_font=dict(size=16), tickfont=dict(size=14))
fig.update_yaxes(title_text="кВт", row=1, col=1)
fig.update_yaxes(title_text="кВт", row=2, col=1)
fig.update_yaxes(title_text="кВт", row=3, col=1)

# ==========================================
# 7. СОХРАНЕНИЕ И ОТКРЫТИЕ БЕЗ КЭША
# ==========================================
html_filename = "index.html"
fig.write_html(html_filename)
file_path = os.path.abspath(html_filename)

print(f"\n✅ Успех! Файл сохранен: {file_path}")
webbrowser.open(f"file://{file_path}?v={int(time.time())}")
print("Готово!")