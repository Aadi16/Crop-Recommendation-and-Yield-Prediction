import os
import requests
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path
import plotly.graph_objects as go


# ---------------------------------------------------------------
# Google Drive Downloader
# ---------------------------------------------------------------
def download_from_drive(shared_link, output_path):
    """
    Downloads a file from Google Drive using a shareable link.
    """
    try:
        file_id = shared_link.split("/d/")[1].split("/")[0]
    except Exception:
        raise ValueError("Invalid Google Drive link format.")

    url = f"https://drive.google.com/file/d/1jwZZyBjXcRo6K9e-6N1CXHnS6Xv0mJOw/view?usp=sharing"
    response = requests.get(url)

    if response.status_code != 200:
        raise ValueError("Failed to download model from Google Drive.")

    with open(output_path, "wb") as f:
        f.write(response.content)


# ---------------------------------------------------------------
# Load Models
# ---------------------------------------------------------------
@st.cache_resource
def load_resources():
    GOOGLE_DRIVE_SHARED_LINK = "PUT_YOUR_SHAREABLE_LINK_HERE"
    local_model_path = Path("Saved_Models/xgb_yield_ensemble.joblib")

    # Download model if not already present
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


# ---------------------------------------------------------------
# Agronomic Scoring
# ---------------------------------------------------------------
def compute_suitability_score(row, temp, rain, humidity, ph, season, state):
    values = []

    values.append(
        1.0 if row.TempMin <= temp <= row.TempMax
        else max(0, 1 - abs(temp - (row.TempMin + row.TempMax) / 2) / 10)
    )

    values.append(
        1.0 if row.RainMin <= rain <= row.RainMax
        else max(0, 1 - abs(rain - (row.RainMin + row.RainMax) / 2) / 1500)
    )

    values.append(min(1.0, humidity / max(row.HumidityMin, 1)))

    values.append(
        1.0 if row.PHmin <= ph <= row.PHmax
        else max(0, 1 - abs(ph - (row.PHmin + row.PHmax) / 2) / 2)
    )

    values.append(1.0 if season in row.SeasonsAllowed.split(",") else 0.3)
    values.append(1.0 if state in row.StatesAllowed.split(",") else 0.3)

    return np.mean(values)


# ---------------------------------------------------------------
# Crop Recommendation (ML + Agronomic Rules)
# ---------------------------------------------------------------
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


# ---------------------------------------------------------------
# Yield Ensemble Prediction
# ---------------------------------------------------------------
def ensemble_predict(models, X):
    predictions = np.array([m.predict(X)[0] for m in models])
    return float(predictions.mean()), float(predictions.std(ddof=1))


# ---------------------------------------------------------------
# Global Explainability (Yield Model)
# ---------------------------------------------------------------
def global_importance(models, feature_names):
    importance = np.mean([m.feature_importances_ for m in models], axis=0)
    return dict(zip(feature_names, importance))


# ---------------------------------------------------------------
# Local Explainability (Yield What-If)
# ---------------------------------------------------------------
def compute_local_explainability(models, X):
    base_pred = float(np.mean([m.predict(X)[0] for m in models]))
    results = {}

    for col in X.columns:
        Xd = X.copy()
        Xu = X.copy()

        Xd[col] *= 0.90
        Xu[col] *= 1.10

        yd = float(np.mean([m.predict(Xd)[0] for m in models]))
        yu = float(np.mean([m.predict(Xu)[0] for m in models]))

        results[col] = {"baseline": base_pred, "decrease": yd, "increase": yu}

    return results


# ---------------------------------------------------------------
# Local Explainability (Crop Recommendation)
# ---------------------------------------------------------------
def compute_rec_local_explainability(best_crop, ranking, N, P, K, temp, humidity, ph, rainfall, state, season):
    base_scores = dict(ranking)
    base_value = base_scores.get(best_crop, None)

    features = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
    results = {}

    for ft in features:
        params = dict(N=N, P=P, K=K, temperature=temp, humidity=humidity, ph=ph, rainfall=rainfall)

        p_dec = params.copy()
        p_inc = params.copy()
        p_dec[ft] *= 0.90
        p_inc[ft] *= 1.10

        _, r_dec = fused_recommendation(
            p_dec["N"], p_dec["P"], p_dec["K"], p_dec["temperature"],
            p_dec["humidity"], p_dec["ph"], p_dec["rainfall"], state, season
        )
        _, r_inc = fused_recommendation(
            p_inc["N"], p_inc["P"], p_inc["K"], p_inc["temperature"],
            p_inc["humidity"], p_inc["ph"], p_inc["rainfall"], state, season
        )

        results[ft] = {
            "baseline": base_value,
            "decrease": dict(r_dec).get(best_crop, None),
            "increase": dict(r_inc).get(best_crop, None),
        }

    return results


# ---------------------------------------------------------------
# Hybrid Pipeline
# ---------------------------------------------------------------
def hybrid_predict(N, P, K, temp, humidity, ph, rainfall, state, season, year, area, production):
    best_crop, ranking = fused_recommendation(N, P, K, temp, humidity, ph, rainfall, state, season)

    final_crop = best_crop[0]
    yield_crop = valid_crop_map.get(final_crop.lower(), final_crop)

    df = pd.DataFrame({
        "Crop": [yield_crop],
        "State": [state],
        "Season": [season],
        "Year_int": [year],
        "Area": [area],
        "Production": [production],
    })

    df["Crop_enc"] = le_crop.transform(df["Crop"])
    df["State_enc"] = le_state.transform(df["State"])
    df["Season_enc"] = le_season.transform(df["Season"])

    X = df[["Crop_enc", "State_enc", "Season_enc", "Year_int", "Area", "Production"]]

    mean_y, std_y = ensemble_predict(yield_ensemble, X)
    ci_low = mean_y - 1.96 * std_y
    ci_high = mean_y + 1.96 * std_y

    return {
        "Crop": final_crop,
        "Confidence": best_crop[1] * 100,
        "Yield": mean_y,
        "Std": std_y,
        "CI": (ci_low, ci_high),
        "GlobalExplain": global_importance(yield_ensemble, X.columns.tolist()),
        "LocalExplain": compute_local_explainability(yield_ensemble, X),
        "RecLocalExplain": compute_rec_local_explainability(
            final_crop, ranking, N, P, K, temp, humidity, ph, rainfall, state, season
        ),
    }


# ---------------------------------------------------------------
# Streamlit Interface
# ---------------------------------------------------------------
st.set_page_config(page_title="Crop & Yield Prediction", layout="wide")
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


# ---------------------------------------------------------------
# Prediction Button
# ---------------------------------------------------------------
if st.button("Predict", use_container_width=True):
    res = hybrid_predict(
        N, P, K, temp, humidity, ph, rainfall, state, season, year, area, production
    )

    crop_name = res["Crop"].upper()
    st.subheader(f"Recommended Crop: {crop_name}")

    st.subheader("Yield Prediction")
    st.metric("Predicted Yield (kg/ha)", f"{res['Yield']:.2f}")
    st.write(f"Uncertainty (Std Dev): {res['Std']:.2f}")
    st.write(f"Confidence Interval: {res['CI'][0]:.2f} – {res['CI'][1]:.2f}")
    st.write(f"(Recommendation Confidence: {res['Confidence']:.1f}%)")

    st.subheader("Global Explainability (Yield Model)")
    st.dataframe(
        pd.DataFrame({
            "Feature": list(res["GlobalExplain"].keys()),
            "Importance": list(res["GlobalExplain"].values()),
        }).sort_values("Importance", ascending=False),
        use_container_width=True,
    )

    st.subheader("Local Explainability (Crop Recommendation)")
    tab1, tab2 = st.tabs(["Interactive Graph", "Raw Values"])

    with tab1:
        r = res["RecLocalExplain"]
        features = []
        dec, base, inc = [], [], []

        for ft, vals in r.items():
            if vals["baseline"] is None:
                continue
            features.append(ft)
            dec.append(vals["decrease"])
            base.append(vals["baseline"])
            inc.append(vals["increase"])

        if features:
            fig = go.Figure()
            fig.add_trace(go.Bar(name="Decrease 10%", x=features, y=dec))
            fig.add_trace(go.Bar(name="Baseline", x=features, y=base))
            fig.add_trace(go.Bar(name="Increase 10%", x=features, y=inc))
            fig.update_layout(
                barmode="group",
                xaxis_title="Input Feature",
                yaxis_title=f"Hybrid Recommendation Score ({crop_name})",
                height=450,
            )
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        if features:
            st.dataframe(
                pd.DataFrame(
                    {"Feature": features, "Decrease 10%": dec,
                     "Baseline": base, "Increase 10%": inc}
                ),
                use_container_width=True,
            )

    st.subheader("Local Explainability (Yield What-If)")

    t3, t4 = st.tabs(["Interactive Graph", "Raw Values"])

    with t3:
        y = res["LocalExplain"]
        y_features = []
        yd, yb, yi = [], [], []

        for ft, vals in y.items():
            y_features.append(ft)
            yd.append(vals["decrease"])
            yb.append(vals["baseline"])
            yi.append(vals["increase"])

        fig2 = go.Figure()
        fig2.add_trace(go.Bar(name="Decrease 10%", x=y_features, y=yd))
        fig2.add_trace(go.Bar(name="Baseline", x=y_features, y=yb))
        fig2.add_trace(go.Bar(name="Increase 10%", x=y_features, y=yi))
        fig2.update_layout(
            barmode="group",
            xaxis_title="Feature",
            yaxis_title="Predicted Yield (kg/ha)",
            height=450,
        )
        st.plotly_chart(fig2, use_container_width=True)

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
