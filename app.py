import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go


# ======================
# LOAD MODELS
# ======================
@st.cache_resource
def load_resources():

    # Load XGBoost Ensemble (replaces RF ensemble)
    xgb_ensemble_path = "Saved_Models/xgb_yield_ensemble.joblib"
    if not os.path.exists(xgb_ensemble_path):
        raise FileNotFoundError("XGBoost yield ensemble file not found.")
    yield_models = joblib.load(xgb_ensemble_path)

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



# =====================================
# AGRONOMIC SOFT CONSTRAINTS
# =====================================
def compute_suitability_score(row, temp, rain, humidity, ph, season, state):
    parts = []

    parts.append(1.0 if row.TempMin <= temp <= row.TempMax
                 else max(0, 1 - abs(temp - (row.TempMin + row.TempMax) / 2) / 10))

    parts.append(1.0 if row.RainMin <= rain <= row.RainMax
                 else max(0, 1 - abs(rain - (row.RainMin + row.RainMax) / 2) / 1500))

    parts.append(min(1.0, humidity / max(row.HumidityMin, 1)))

    parts.append(1.0 if row.PHmin <= ph <= row.PHmax
                 else max(0, 1 - abs(ph - (row.PHmin + row.PHmax) / 2) / 2))

    parts.append(1.0 if season in row.SeasonsAllowed.split(",") else 0.3)
    parts.append(1.0 if state in row.StatesAllowed.split(",") else 0.3)

    return np.mean(parts)



# =====================================
# HYBRID RECOMMENDATION (ML + RULES)
# =====================================
def fused_recommendation(N, P, K, temp, humidity, ph, rainfall, state, season):

    inp = pd.DataFrame(
        [[N, P, K, temp, humidity, ph, rainfall]],
        columns=["N", "P", "K", "temperature", "humidity", "ph", "rainfall"],
    )

    probs = crop_rec_model.predict_proba(inp)[0]
    crop_classes = crop_label_encoder.classes_

    annual_rain = rainfall * 12
    scores = []

    for i, crop in enumerate(crop_classes):
        ml_score = probs[i]

        row = suitability_df[suitability_df["Crop"].str.lower() == crop.lower()]
        suitability = (
            compute_suitability_score(row.iloc[0], temp, annual_rain, humidity, ph, season, state)
            if not row.empty else 0.5
        )

        final_score = 0.6 * ml_score + 0.4 * suitability
        scores.append((crop, final_score))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[0], scores



# =====================================
# XGBOOST ENSEMBLE: PREDICT + UNCERTAINTY
# =====================================
def ensemble_predict(models, X):
    preds = np.array([m.predict(X)[0] for m in models])
    return float(preds.mean()), float(preds.std(ddof=1))



# =====================================
# YIELD EXPLAINABILITY
# =====================================
def global_importance(models, feature_names):
    fi = np.mean([m.feature_importances_ for m in models], axis=0)
    return dict(zip(feature_names, fi))


def compute_local_explainability(models, X):
    base = float(np.mean([m.predict(X)[0] for m in models]))
    results = {}

    for feature in X.columns:
        Xd = X.copy()
        Xu = X.copy()

        Xd[feature] *= 0.90
        Xu[feature] *= 1.10

        yd = float(np.mean([m.predict(Xd)[0] for m in models]))
        yu = float(np.mean([m.predict(Xu)[0] for m in models]))

        results[feature] = {"baseline": base, "decrease": yd, "increase": yu}

    return results



# =====================================
# CROP RECOMMENDER EXPLAINABILITY
# =====================================
def compute_rec_local_explainability(best_crop, ranking, N, P, K, temp, humidity, ph, rainfall, state, season):

    base_scores = dict(ranking)
    base_value = base_scores.get(best_crop, None)

    features = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
    results = {}

    for f in features:

        params = dict(N=N, P=P, K=K, temperature=temp, humidity=humidity, ph=ph, rainfall=rainfall)

        params_dec = params.copy()
        params_inc = params.copy()
        params_dec[f] *= 0.90
        params_inc[f] *= 1.10

        _, r_dec = fused_recommendation(
            params_dec["N"], params_dec["P"], params_dec["K"],
            params_dec["temperature"], params_dec["humidity"],
            params_dec["ph"], params_dec["rainfall"],
            state, season,
        )
        _, r_inc = fused_recommendation(
            params_inc["N"], params_inc["P"], params_inc["K"],
            params_inc["temperature"], params_inc["humidity"],
            params_inc["ph"], params_inc["rainfall"],
            state, season,
        )

        dec_val = dict(r_dec).get(best_crop, None)
        inc_val = dict(r_inc).get(best_crop, None)

        results[f] = {"baseline": base_value, "decrease": dec_val, "increase": inc_val}

    return results



# =====================================
# HYBRID PIPELINE
# =====================================
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



# =====================================
# STREAMLIT UI
# =====================================
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



# =====================================
# PREDICT BUTTON
# =====================================
if st.button("Predict", use_container_width=True):

    res = hybrid_predict(
        N, P, K, temp, humidity, ph, rainfall,
        state, season, year, area, production
    )

    crop_name = res["Crop"].upper()
    st.subheader(f"Recommended Crop: {crop_name}")

    st.subheader("Yield Prediction")
    st.metric("Predicted Yield (kg/ha)", f"{res['Yield']:.2f}")
    st.write(f"Uncertainty (Std Dev): {res['Std']:.2f}")
    st.write(f"Confidence Interval: {res['CI'][0]:.2f} – {res['CI'][1]:.2f}")
    st.write(f"(Recommendation Confidence: {res['Confidence']:.1f}%)")

    # Global Explainability
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

    # Crop Recommendation Local Explainability
    st.subheader("Local Explainability (Crop Recommendation)")
    tab1, tab2 = st.tabs(["Interactive Graph", "Raw Values"])

    with tab1:
        rec_local = res["RecLocalExplain"]

        feats = []
        dec = []
        base = []
        inc = []

        for f, vals in rec_local.items():
            if vals["baseline"] is None:
                continue
            feats.append(f)
            dec.append(vals["decrease"])
            base.append(vals["baseline"])
            inc.append(vals["increase"])

        if feats:
            fig_rec = go.Figure()
            fig_rec.add_trace(go.Bar(name="Decrease 10%", x=feats, y=dec))
            fig_rec.add_trace(go.Bar(name="Baseline", x=feats, y=base))
            fig_rec.add_trace(go.Bar(name="Increase 10%", x=feats, y=inc))
            fig_rec.update_layout(
                barmode="group",
                xaxis_title="Input feature",
                yaxis_title=f"Hybrid recommendation score for {crop_name}",
                height=450,
            )
            st.plotly_chart(fig_rec, use_container_width=True)

    with tab2:
        if feats:
            st.dataframe(
                pd.DataFrame(
                    {
                        "Feature": feats,
                        "Decrease 10%": dec,
                        "Baseline": base,
                        "Increase 10%": inc,
                    }
                ),
                use_container_width=True,
            )

    # Yield Local Explainability
    st.subheader("Local Explainability (Yield What-If)")
    t3, t4 = st.tabs(["Interactive Graph", "Raw Values"])

    with t3:
        local = res["LocalExplain"]

        y_feats = []
        y_dec = []
        y_base = []
        y_inc = []

        for f, vals in local.items():
            y_feats.append(f)
            y_dec.append(vals["decrease"])
            y_base.append(vals["baseline"])
            y_inc.append(vals["increase"])

        fig_y = go.Figure()
        fig_y.add_trace(go.Bar(name="Decrease 10%", x=y_feats, y=y_dec))
        fig_y.add_trace(go.Bar(name="Baseline", x=y_feats, y=y_base))
        fig_y.add_trace(go.Bar(name="Increase 10%", x=y_feats, y=y_inc))
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
                        "Feature": f,
                        "Decrease 10%": vals["decrease"],
                        "Baseline": vals["baseline"],
                        "Increase 10%": vals["increase"],
                    }
                    for f, vals in res["LocalExplain"].items()
                ]
            ),
            use_container_width=True,
        )
