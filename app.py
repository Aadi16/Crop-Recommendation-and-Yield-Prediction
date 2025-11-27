import os
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go

# ---------------------------------------------------------------------
# Page config (must be the first Streamlit command)
# ---------------------------------------------------------------------
st.set_page_config(page_title="Crop & Yield Prediction", layout="wide")


# ---------------------------------------------------------------------
# Google Drive utilities
# ---------------------------------------------------------------------
def extract_file_id(shared_link: str) -> str:
    """
    Extract a Google Drive file ID from a shareable link.
    Supports common formats like:
      - https://drive.google.com/file/d/<ID>/view?usp=sharing
      - https://drive.google.com/open?id=<ID>
      - https://drive.google.com/uc?export=download&id=<ID>
    """
    patterns = [
        r"/d/([a-zA-Z0-9_-]+)",
        r"id=([a-zA-Z0-9_-]+)",
        r"uc\?export=download&id=([a-zA-Z0-9_-]+)",
    ]
    for p in patterns:
        m = re.search(p, shared_link)
        if m:
            return m.group(1)
    raise ValueError("Invalid Google Drive link format.")


def download_from_drive(shared_link: str, output_path: Path) -> None:
    """
    Download a file from Google Drive given a share link and save it to output_path.
    """
    file_id = extract_file_id(shared_link)
    url = f"https://drive.google.com/uc?export=download&id=1jwZZyBjXcRo6K9e-6N1CXHnS6Xv0mJOw"

    response = requests.get(url)
    if response.status_code != 200:
        raise ValueError(f"Failed to download model. HTTP {response.status_code}")

    with open(output_path, "wb") as f:
        f.write(response.content)


# ---------------------------------------------------------------------
# Load models and resources
# ---------------------------------------------------------------------
@st.cache_resource
def load_resources():
    # Google Drive link for the XGBoost yield ensemble
    GOOGLE_DRIVE_SHARED_LINK = (
        "https://drive.google.com/file/d/1jwZZyBjXcRo6K9e-6N1CXHnS6Xv0mJOw/view?usp=sharing"
    )
    local_model_path = Path("Saved_Models/xgb_yield_ensemble.joblib")

    if not local_model_path.exists():
        local_model_path.parent.mkdir(parents=True, exist_ok=True)
        download_from_drive(GOOGLE_DRIVE_SHARED_LINK, local_model_path)

    yield_models = joblib.load(local_model_path)

    crop_rec_model = joblib.load("Saved_Models/crop_rec_model.pkl")
    crop_label_encoder = joblib.load("Saved_Models/crop_label_encoder.pkl")
    le_crop = joblib.load("Saved_Models/le_crop.pkl")
    le_state = joblib.load("Saved_Models/le_state.pkl")
    le_season = joblib.load("Saved_Models/le_season.pkl")
    valid_crop_map = joblib.load("Saved_Models/valid_crop_map.pkl")

    suitability_df = pd.read_csv("crop_suitability.csv")

    return (
        yield_models,
        crop_rec_model,
        crop_label_encoder,
        le_crop,
        le_state,
        le_season,
        valid_crop_map,
        suitability_df,
    )


(
    yield_ensemble,
    crop_rec_model,
    crop_label_encoder,
    le_crop,
    le_state,
    le_season,
    valid_crop_map,
    suitability_df,
) = load_resources()


# ---------------------------------------------------------------------
# Agronomic scoring
# ---------------------------------------------------------------------
def compute_suitability_score(row, temp, rain, humidity, ph, season, state):
    values = []

    # Temperature
    values.append(
        1.0
        if row.TempMin <= temp <= row.TempMax
        else max(0, 1 - abs(temp - (row.TempMin + row.TempMax) / 2) / 10)
    )

    # Annual rainfall
    values.append(
        1.0
        if row.RainMin <= rain <= row.RainMax
        else max(0, 1 - abs(rain - (row.RainMin + row.RainMax) / 2) / 1500)
    )

    # Humidity
    values.append(min(1.0, humidity / max(row.HumidityMin, 1)))

    # Soil pH
    values.append(
        1.0
        if row.PHmin <= ph <= row.PHmax
        else max(0, 1 - abs(ph - (row.PHmin + row.PHmax) / 2) / 2)
    )

    # Season
    values.append(1.0 if season in row.SeasonsAllowed.split(",") else 0.3)

    # State / region
    values.append(1.0 if state in row.StatesAllowed.split(",") else 0.3)

    return np.mean(values)


# ---------------------------------------------------------------------
# Crop recommendation (ML + agronomic rules)
# ---------------------------------------------------------------------
def fused_recommendation(N, P, K, temp, humidity, ph, rainfall, state, season):
    df_inp = pd.DataFrame(
        [[N, P, K, temp, humidity, ph, rainfall]],
        columns=["N", "P", "K", "temperature", "humidity", "ph", "rainfall"],
    )

    probs = crop_rec_model.predict_proba(df_inp)[0]
    crops = crop_label_encoder.classes_
    annual_rain = rainfall * 12

    scores = []

    for i, crop in enumerate(crops):
        ml_score = probs[i]
        row = suitability_df[suitability_df["Crop"].str.lower() == crop.lower()]

        if row.empty:
            suitability = 0.5
        else:
            suitability = compute_suitability_score(
                row.iloc[0], temp, annual_rain, humidity, ph, season, state
            )

        final_score = 0.6 * ml_score + 0.4 * suitability
        scores.append((crop, final_score))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[0], scores


# ---------------------------------------------------------------------
# Yield ensemble prediction (XGBoost ensemble)
# ---------------------------------------------------------------------
def ensemble_predict(models, X):
    preds = np.array([m.predict(X)[0] for m in models])
    return float(preds.mean()), float(preds.std(ddof=1))


# ---------------------------------------------------------------------
# Global explainability (yield model)
# ---------------------------------------------------------------------
def global_importance(models, feature_names):
    importance = np.mean([m.feature_importances_ for m in models], axis=0)
    return dict(zip(feature_names, importance))


# ---------------------------------------------------------------------
# Local explainability (yield what-if)
# ---------------------------------------------------------------------
def compute_local_explainability(models, X):
    base_pred = float(np.mean([m.predict(X)[0] for m in models]))
    results = {}

    for col in X.columns:
        X_down = X.copy()
        X_up = X.copy()

        X_down[col] *= 0.90
        X_up[col] *= 1.10

        y_down = float(np.mean([m.predict(X_down)[0] for m in models]))
        y_up = float(np.mean([m.predict(X_up)[0] for m in models]))

        results[col] = {
            "baseline": base_pred,
            "decrease": y_down,
            "increase": y_up,
        }

    return results


# ---------------------------------------------------------------------
# Local explainability (crop recommendation what-if)
# ---------------------------------------------------------------------
def compute_rec_local_explainability(
    best_crop,
    ranking,
    N,
    P,
    K,
    temp,
    humidity,
    ph,
    rainfall,
    state,
    season,
):
    base_scores = dict(ranking)
    base_value = base_scores.get(best_crop, None)

    features = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
    results = {}

    for ft in features:
        params = dict(
            N=N,
            P=P,
            K=K,
            temperature=temp,
            humidity=humidity,
            ph=ph,
            rainfall=rainfall,
        )

        p_dec = params.copy()
        p_inc = params.copy()
        p_dec[ft] *= 0.90
        p_inc[ft] *= 1.10

        _, r_dec = fused_recommendation(
            p_dec["N"],
            p_dec["P"],
            p_dec["K"],
            p_dec["temperature"],
            p_dec["humidity"],
            p_dec["ph"],
            p_dec["rainfall"],
            state,
            season,
        )
        _, r_inc = fused_recommendation(
            p_inc["N"],
            p_inc["P"],
            p_inc["K"],
            p_inc["temperature"],
            p_inc["humidity"],
            p_inc["ph"],
            p_inc["rainfall"],
            state,
            season,
        )

        results[ft] = {
            "baseline": base_value,
            "decrease": dict(r_dec).get(best_crop, None),
            "increase": dict(r_inc).get(best_crop, None),
        }

    return results


# ---------------------------------------------------------------------
# Hybrid prediction pipeline
# ---------------------------------------------------------------------
def hybrid_predict(
    N,
    P,
    K,
    temp,
    humidity,
    ph,
    rainfall,
    state,
    season,
    year,
    area,
    production,
):
    best_crop, ranking = fused_recommendation(
        N, P, K, temp, humidity, ph, rainfall, state, season
    )

    final_crop = best_crop[0]
    yield_crop = valid_crop_map.get(final_crop.lower(), final_crop)

    df = pd.DataFrame(
        {
            "Crop": [yield_crop],
            "State": [state],
            "Season": [season],
            "Year_int": [year],
            "Area": [area],
            "Production": [production],
        }
    )

    df["Crop_enc"] = le_crop.transform(df["Crop"])
    df["State_enc"] = le_state.transform(df["State"])
    df["Season_enc"] = le_season.transform(df["Season"])

    X = df[["Crop_enc", "State_enc", "Season_enc", "Year_int", "Area", "Production"]]

    mean_y, std_y = ensemble_predict(yield_ensemble, X)
    ci_low = mean_y - 1.96 * std_y
    ci_high = mean_y + 1.96 * std_y

    global_exp = global_importance(yield_ensemble, X.columns.tolist())
    local_exp = compute_local_explainability(yield_ensemble, X)
    rec_local_exp = compute_rec_local_explainability(
        final_crop, ranking, N, P, K, temp, humidity, ph, rainfall, state, season
    )

    return {
        "Crop": final_crop,
        "Confidence": best_crop[1] * 100,
        "Yield": mean_y,
        "Std": std_y,
        "CI": (ci_low, ci_high),
        "GlobalExplain": global_exp,
        "LocalExplain": local_exp,
        "RecLocalExplain": rec_local_exp,
    }


# ---------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------
st.title("Crop Recommendation and Yield Prediction")

st.subheader("Input Parameters")

c1, c2, c3 = st.columns(3)
with c1:
    N = st.number_input("Nitrogen (N)", 0, 200, 90)
    P = st.number_input("Phosphorus (P)", 0, 200, 40)
    K = st.number_input("Potassium (K)", 0, 200, 40)

with c2:
    temp = st.number_input("Temperature (°C)", -10.0, 60.0, 30.0)
    humidity = st.number_input("Humidity (%)", 0, 100, 80)
    ph = st.number_input("Soil pH", 0.0, 14.0, 6.5)

with c3:
    rainfall = st.number_input("Rainfall (mm/month)", 0.0, 5000.0, 250.0)
    area = st.number_input("Area (ha)", 0.0, 10000.0, 10.0)
    production = st.number_input("Production (tonnes)", 0.0, 10000.0, 30.0)

st.subheader("Location and Time")
c4, c5, c6 = st.columns(3)
with c4:
    state = st.selectbox("State", options=le_state.classes_.tolist())
with c5:
    season = st.selectbox("Season", options=le_season.classes_.tolist())
with c6:
    year = st.number_input("Year", min_value=2000, max_value=2050, value=2024)


# ---------------------------------------------------------------------
# Prediction button
# ---------------------------------------------------------------------
if st.button("Predict", use_container_width=True):
    res = hybrid_predict(
        N,
        P,
        K,
        temp,
        humidity,
        ph,
        rainfall,
        state,
        season,
        year,
        area,
        production,
    )

    crop_name = res["Crop"].upper()
    st.subheader(f"Recommended Crop: {crop_name}")

    st.subheader("Yield Prediction")
    st.metric("Predicted Yield (kg/ha)", f"{res['Yield']:.2f}")
    st.write(f"Uncertainty (Std Dev): {res['Std']:.2f}")
    st.write(f"Confidence Interval: {res['CI'][0]:.2f} – {res['CI'][1]:.2f}")
    st.write(f"(Recommendation Confidence: {res['Confidence']:.1f}%)")

    # Global explainability
    st.subheader("Global Explainability (Yield Model)")
    st.dataframe(
        pd.DataFrame(
            {
                "Feature": list(res["GlobalExplain"].keys()),
                "Importance": list(res["GlobalExplain"].values()),
            }
        ).sort_values("Importance", ascending=False),
        use_container_width=True,
    )

    # Crop recommendation: local explainability
    st.subheader("Local Explainability (Crop Recommendation)")
    tab1, tab2 = st.tabs(["Interactive Graph", "Raw Values"])

    with tab1:
        rec_local = res["RecLocalExplain"]
        features = []
        dec_vals = []
        base_vals = []
        inc_vals = []

        for ft, vals in rec_local.items():
            if vals["baseline"] is None:
                continue
            features.append(ft)
            dec_vals.append(vals["decrease"])
            base_vals.append(vals["baseline"])
            inc_vals.append(vals["increase"])

        if features:
            fig_rec = go.Figure()
            fig_rec.add_trace(go.Bar(name="Decrease 10%", x=features, y=dec_vals))
            fig_rec.add_trace(go.Bar(name="Baseline", x=features, y=base_vals))
            fig_rec.add_trace(go.Bar(name="Increase 10%", x=features, y=inc_vals))
            fig_rec.update_layout(
                barmode="group",
                xaxis_title="Input Feature",
                yaxis_title=f"Hybrid recommendation score ({crop_name})",
                height=450,
            )
            st.plotly_chart(fig_rec, use_container_width=True)

    with tab2:
        if features:
            st.dataframe(
                pd.DataFrame(
                    {
                        "Feature": features,
                        "Decrease 10%": dec_vals,
                        "Baseline": base_vals,
                        "Increase 10%": inc_vals,
                    }
                ),
                use_container_width=True,
            )

    # Yield local explainability
    st.subheader("Local Explainability (Yield What-If)")
    t3, t4 = st.tabs(["Interactive Graph", "Raw Values"])

    with t3:
        local = res["LocalExplain"]
        y_features = []
        y_dec = []
        y_base = []
        y_inc = []

        for ft, vals in local.items():
            y_features.append(ft)
            y_dec.append(vals["decrease"])
            y_base.append(vals["baseline"])
            y_inc.append(vals["increase"])

        fig_y = go.Figure()
        fig_y.add_trace(go.Bar(name="Decrease 10%", x=y_features, y=y_dec))
        fig_y.add_trace(go.Bar(name="Baseline", x=y_features, y=y_base))
        fig_y.add_trace(go.Bar(name="Increase 10%", x=y_features, y=y_inc))
        fig_y.update_layout(
            barmode="group",
            xaxis_title="Feature",
            yaxis_title="Predicted Yield (kg/ha)",
            height=450,
        )
        st.plotly_chart(fig_y, use_container_width=True)

    with t4:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Feature": ft,
                        "Decrease 10%": vals["decrease"],
                        "Baseline": vals["baseline"],
                        "Increase 10%": vals["increase"],
                    }
                    for ft, vals in res["LocalExplain"].items()
                ]
            ),
            use_container_width=True,
        )
