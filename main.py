from contextlib import asynccontextmanager

import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# =====================================================
# CONFIG — satu-satunya tempat nama file di-set.
# =====================================================

DATA_CSV = "kurs_jisdor.csv"
MODEL_PKL = "prophet_model.pkl"
EVAL_CSV = "evaluation_results.csv"
DATA_SOURCE_URL = "https://www.bi.go.id/id/statistik/informasi-kurs/jisdor/Default.aspx"


def find_column(columns, keyword):
    """Cari nama kolom yang mengandung `keyword` (case-insensitive)."""
    for c in columns:
        if keyword.upper() in c.upper():
            return c
    return None


# =====================================================
# STATE — dimuat SEKALI saat server start (bukan tiap request),
# supaya endpoint tetap cepat.
# =====================================================

state = {}


def load_all():
    df = pd.read_csv(DATA_CSV)
    date_col = find_column(df.columns, "tanggal") or find_column(df.columns, "date")
    kurs_col = find_column(df.columns, "kurs") or find_column(df.columns, "rate")
    if date_col is None or kurs_col is None:
        raise RuntimeError(f"Kolom tanggal/kurs tidak ditemukan. Kolom tersedia: {list(df.columns)}")

    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).set_index(date_col)
    series = df[kurs_col].asfreq("B").ffill()
    series.name = "USD/IDR Exchange Rate (IDR)"

    model = joblib.load(MODEL_PKL)

    eval_df = pd.read_csv(EVAL_CSV)
    mae_c = find_column(eval_df.columns, "MAE")
    rmse_c = find_column(eval_df.columns, "RMSE")
    mape_c = find_column(eval_df.columns, "MAPE")
    model_c = find_column(eval_df.columns, "Model") or eval_df.columns[0]
    missing = [n for n, c in [("MAE", mae_c), ("RMSE", rmse_c), ("MAPE", mape_c)] if c is None]
    if missing:
        raise RuntimeError(f"Kolom {missing} tidak ditemukan di {EVAL_CSV}. Kolom tersedia: {list(eval_df.columns)}")

    deployed = eval_df[eval_df[model_c].str.contains("Prophet", case=False, na=False)]
    deployed_row = deployed.iloc[0] if not deployed.empty else eval_df.sort_values(rmse_c).iloc[0]
    best_row = eval_df.sort_values(rmse_c).iloc[0]

    state.update(
        series=series,
        model=model,
        eval_df=eval_df,
        cols=(mae_c, rmse_c, mape_c, model_c),
        deployed_row=deployed_row,
        best_row=best_row,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_all()
    yield


app = FastAPI(title="JISDOR Forecast API", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")


# =====================================================
# RESPONSE MODELS
# =====================================================

class PricePoint(BaseModel):
    tanggal: str
    nilai: float


class EvalRow(BaseModel):
    model: str
    mae: float
    rmse: float
    mape: float


class Summary(BaseModel):
    records: int
    start_date: str
    end_date: str
    last_value: float
    model_name: str
    model_params: dict
    data_source_url: str
    best_model: str
    is_deployed_best: bool


# =====================================================
# ROUTES
# =====================================================

@app.get("/", response_class=HTMLResponse)
def index():
    with open("static/index.html", encoding="utf-8") as f:
        return f.read()

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "JISDOR Forecast API",
        "model_loaded": "model" in state,
    }

@app.get("/api/summary", response_model=Summary)
def get_summary():
    series = state["series"]
    model = state["model"]
    deployed_row = state["deployed_row"]
    best_row = state["best_row"]
    _, _, _, model_c = state["cols"]

    return Summary(
        records=len(series),
        start_date=series.index.min().strftime("%Y-%m-%d"),
        end_date=series.index.max().strftime("%Y-%m-%d"),
        last_value=float(series.iloc[-1]),
        model_name=str(deployed_row[model_c]),
        model_params={
            "seasonality_mode": model.seasonality_mode,
            "seasonality_prior_scale": model.seasonality_prior_scale,
            "yearly_seasonality": bool(model.yearly_seasonality),
            "frequency": "Business day (B), libur nasional di-forward-fill",
        },
        data_source_url=DATA_SOURCE_URL,
        best_model=str(best_row[model_c]),
        is_deployed_best=str(deployed_row[model_c]) == str(best_row[model_c]),
    )


@app.get("/api/historical", response_model=list[PricePoint])
def get_historical(window: int = Query(90, ge=7, le=1000, description="Jumlah hari kerja terakhir")):
    series = state["series"].tail(window)
    return [
        PricePoint(tanggal=d.strftime("%Y-%m-%d"), nilai=float(v))
        for d, v in series.items()
    ]


@app.get("/api/forecast", response_model=list[PricePoint])
def get_forecast(horizon: int = Query(30, description="Horizon forecast dalam hari kerja")):
    if horizon not in (7, 14, 30):
        raise HTTPException(status_code=400, detail="Horizon harus 7, 14, atau 30")

    model = state["model"]
    future = model.make_future_dataframe(periods=horizon, freq="B")
    forecast = model.predict(future)
    forecast_future = forecast[["ds", "yhat"]].tail(horizon)

    return [
        PricePoint(tanggal=row.ds.strftime("%Y-%m-%d"), nilai=float(row.yhat))
        for row in forecast_future.itertuples()
    ]


@app.get("/api/evaluation", response_model=list[EvalRow])
def get_evaluation():
    eval_df = state["eval_df"]
    mae_c, rmse_c, mape_c, model_c = state["cols"]

    return [
        EvalRow(model=str(r[model_c]), mae=float(r[mae_c]), rmse=float(r[rmse_c]), mape=float(r[mape_c]))
        for _, r in eval_df.sort_values(rmse_c).iterrows()
    ]
