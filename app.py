"""
Melbourne Housing — Price Predictor
Streamlit Deploy Version (Google Drive Models)
"""

import streamlit as st
import pickle, os
import numpy as np
import gdown

st.set_page_config(page_title="House Price Predictor", page_icon="🏠")

# ── Google Drive Loader ───────────────────────────────
@st.cache_resource
def load_model(file_id, filename):
    url = f"https://drive.google.com/uc?id={file_id}"
    if not os.path.exists(filename):
        gdown.download(url, filename, quiet=False)
    with open(filename, "rb") as f:
        return pickle.load(f)

# ── YOUR FILE IDS ─────────────────────────────────────
CLS_ID = "12n1fS_ZZ_q8UbZ6R1Z8fjCj3CVztFcZB"
REG_ID = "1CwsVw8pNAlhkmZD9OjUTDiTr6Ryf73G-"
SCALER_ID = "1tsddE5GtQgg4Iz9dnzm9z63Uf3A3O7DM"

# ── Load Models ───────────────────────────────────────
cls_data = load_model(CLS_ID, "cls.pkl")
reg_data = load_model(REG_ID, "reg.pkl")
scaler = load_model(SCALER_ID, "scaler.pkl")["scaler"]

# ── UI ────────────────────────────────────────────────
st.title("🏠 Melbourne House Price Predictor")

col1, col2, col3 = st.columns(3)

with col1:
    rooms = st.number_input("Rooms", 1, 10, 3)
    bath = st.number_input("Bathrooms", 1, 5, 2)

with col2:
    car = st.number_input("Car", 0, 5, 1)
    land = st.number_input("Land Size", 0, 10000, 500)

with col3:
    build = st.number_input("Building Area", 0, 1000, 120)
    year = st.number_input("Year Built", 1900, 2023, 2000)

distance = st.slider("Distance from CBD", 0.0, 50.0, 10.0)

ptype = st.selectbox("Property Type", ["House", "Townhouse", "Unit"])
region = st.selectbox("Region", ["Eastern Metro", "Northern Metro", "Southern Metro"])

# ── Feature Engineering ───────────────────────────────
type_enc = ["House","Townhouse","Unit"].index(ptype)
region_enc = ["Eastern Metro","Northern Metro","Southern Metro"].index(region)

house_age = 2024 - year
build_ratio = build / max(land, 1)
rooms_per_km = rooms / max(distance, 0.1)

row = np.array([[rooms, bath, car, land, build, distance,
                 5000, -37.81, 144.96, 2017, 6,
                 house_age, build_ratio, rooms_per_km, rooms,
                 type_enc, 0, region_enc, 0]])

# ── Prediction ────────────────────────────────────────
if st.button("🔮 Predict Price"):

    X = scaler.transform(row)

    # Classification
    probs = cls_data['model'].predict_proba(X)[0]
    idx = int(np.argmax(probs))
    classes = ["Budget", "Mid-Range", "Premium", "Luxury"]

    # Regression
    price = reg_data['model'].predict(X)[0]

    st.success(f"🏷 Category: {classes[idx]}")
    st.success(f"💰 Estimated Price: A${price:,.0f}")