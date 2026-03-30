"""
Melbourne Housing — Price Predictor
Clean UI: Input form -> Result only.
Run: streamlit run app.py
"""

import streamlit as st
import os, json, pickle
import numpy as np

st.set_page_config(
    page_title="Melbourne House Price Predictor",
    page_icon="🏠",
    layout="centered",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Syne', sans-serif; }
.stApp { background-color: #0d0d0d; color: #f0f0f0; }
#MainMenu, footer, header { visibility: hidden; }

.title    { text-align:center; font-size:2.4rem; font-weight:800; color:#e94560; margin:20px 0 4px 0; }
.subtitle { text-align:center; color:#666; font-family:'Space Mono',monospace;
            font-size:0.82rem; margin-bottom:28px; }
.section-label {
    font-size:0.7rem; font-weight:700; color:#555; letter-spacing:2px;
    text-transform:uppercase; margin:22px 0 6px 0;
    border-left:3px solid #e94560; padding-left:8px;
}
div[data-testid="stButton"] > button {
    width:100%; background:linear-gradient(90deg,#e94560,#533483);
    color:white; font-weight:800; font-size:1.1rem; padding:16px;
    border:none; border-radius:12px; margin-top:20px; letter-spacing:1px;
}
div[data-testid="stButton"] > button:hover { opacity:0.88; }
.result-card {
    background:#111827; border-radius:16px; padding:28px 20px;
    text-align:center; border:2px solid #e94560; margin-top:8px;
}
.result-icon  { font-size:2.4rem; margin-bottom:6px; }
.result-label { font-size:0.72rem; color:#555; font-family:'Space Mono',monospace;
                letter-spacing:2px; text-transform:uppercase; margin-bottom:10px; }
.result-main  { font-size:1.9rem; font-weight:800; line-height:1.1; }
.result-sub   { font-size:0.82rem; color:#666; font-family:monospace; margin-top:6px; }
.model-footer { text-align:center; margin-top:18px; font-family:monospace;
                font-size:0.75rem; color:#444; }
</style>
""", unsafe_allow_html=True)

OUTPUT_DIR  = './outputs'
CLASS_NAMES = ['Budget', 'Mid-Range', 'Premium', 'Luxury']
CLASS_EMOJI = ['🟢', '🟡', '🟠', '🔴']
CLASS_COLOR = ['#06d6a0', '#ffd166', '#f4a261', '#e94560']

# ── Helper ────────────────────────────────────────────
def band_range(idx, q1, q2, q3):
    return [
        f"Below A${q1/1e6:.2f}M",
        f"A${q1/1e6:.2f}M to A${q2/1e6:.2f}M",
        f"A${q2/1e6:.2f}M to A${q3/1e6:.2f}M",
        f"Above A${q3/1e6:.2f}M",
    ][idx]

# ── Load best models ──────────────────────────────────
best_path = os.path.join(OUTPUT_DIR, 'best_models.json')
if not os.path.exists(best_path):
    st.error("No trained models found. Please run first:")
    st.code("python train.py", language="bash")
    st.stop()

with open(best_path) as f:
    info = json.load(f)

feat_cols   = info['feature_cols']
class_names = info['class_names']
q1, q2, q3 = info['price_quartiles']
best_cls    = info['best_classifier']
best_reg    = info['best_regressor']

@st.cache_resource
def load_model(pkl_file):
    path = os.path.join(OUTPUT_DIR, pkl_file)
    if not os.path.exists(path): return None
    with open(path, 'rb') as f: return pickle.load(f)

@st.cache_resource
def load_scaler():
    path = os.path.join(OUTPUT_DIR, 'scaler.pkl')
    if not os.path.exists(path): return None
    with open(path, 'rb') as f: return pickle.load(f)['scaler']

cls_data = load_model(best_cls['pkl_file'])
reg_data = load_model(best_reg['pkl_file'])
scaler   = load_scaler()

# ── Header ────────────────────────────────────────────
st.markdown("<div class='title'>🏠 Melbourne House Price</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Fill in the details below to predict the price</div>",
            unsafe_allow_html=True)

# ── Input Form ────────────────────────────────────────
st.markdown("<div class='section-label'>Property Details</div>", unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)
with c1:
    rooms      = st.number_input("Rooms",        1, 10,    3)
    bathroom   = st.number_input("Bathrooms",    1,  6,    2)
with c2:
    car        = st.number_input("Car Spaces",   0,  5,    1)
    landsize   = st.number_input("Land (m²)",    0, 50000, 550)
with c3:
    build_area = st.number_input("Build Area (m²)", 0, 1000, 120)
    year_built = st.number_input("Year Built",   1880, 2023, 1990)

st.markdown("<div class='section-label'>Location</div>", unsafe_allow_html=True)

c4, c5 = st.columns(2)
with c4:
    distance  = st.number_input("Distance from CBD (km)", 0.0, 50.0, 10.0, step=0.5)
    prop_type = st.selectbox("Property Type", ['House', 'Townhouse', 'Unit'])
with c5:
    region = st.selectbox("Region", [
        'Eastern Metro', 'Northern Metro', 'Southern Metro',
        'Western Metro', 'South-Eastern Metro',
        'Eastern Victoria', 'Northern Victoria', 'Western Victoria'
    ])
    prop_count = st.number_input("Properties in Suburb", 100, 30000, 5000)

# ── Build Feature Vector ──────────────────────────────
type_enc   = ['House', 'Townhouse', 'Unit'].index(prop_type)
region_map = {
    'Eastern Metro': 0, 'Eastern Victoria': 1, 'Northern Metro': 2,
    'Northern Victoria': 3, 'South-Eastern Metro': 4, 'Southern Metro': 5,
    'Western Metro': 6, 'Western Victoria': 7
}
region_enc  = region_map.get(region, 0)
house_age   = 2017 - year_built
build_ratio = build_area / max(landsize, 1)
rooms_pkm   = rooms / max(distance, 0.1)

inp_map = {
    'Rooms': rooms, 'Bathroom': bathroom, 'Car': car,
    'Landsize': landsize, 'BuildingArea': build_area,
    'Distance': distance, 'Propertycount': prop_count,
    'Lattitude': -37.81, 'Longtitude': 144.96,
    'sale_year': 2017, 'sale_month': 6,
    'house_age': house_age, 'build_ratio': build_ratio,
    'rooms_per_km': rooms_pkm, 'Bedroom2': rooms,
    'Type_enc': type_enc, 'Method_enc': 0,
    'Regionname_enc': region_enc, 'CouncilArea_enc': 0,
}
row = np.array([[inp_map.get(f, 0) for f in feat_cols]])

# ── Predict Button ────────────────────────────────────
if st.button("🔮  PREDICT PRICE"):

    if not cls_data or not reg_data:
        st.error("Model files missing. Run: python train.py")
        st.stop()

    Xcls = scaler.transform(row) if cls_data.get('scaled') and scaler else row
    Xreg = scaler.transform(row) if reg_data.get('scaled') and scaler else row

    # Classification
    probs    = cls_data['model'].predict_proba(Xcls)[0]
    pred_idx = int(np.argmax(probs))
    band     = class_names[pred_idx]
    emoji    = CLASS_EMOJI[pred_idx]
    color    = CLASS_COLOR[pred_idx]

    # Regression
    price = max(float(reg_data['model'].predict(Xreg)[0]), 100_000)

    # ── Show Results ──────────────────────────
    st.markdown("---")
    col_band, col_price = st.columns(2)

    with col_band:
        st.markdown(f"""
        <div class='result-card'>
            <div class='result-icon'>{emoji}</div>
            <div class='result-label'>Price Category</div>
            <div class='result-main' style='color:{color}'>{band}</div>
            <div class='result-sub'>{band_range(pred_idx, q1, q2, q3)}</div>
        </div>
        """, unsafe_allow_html=True)

    with col_price:
        st.markdown(f"""
        <div class='result-card'>
            <div class='result-icon'>💰</div>
            <div class='result-label'>Estimated Price</div>
            <div class='result-main' style='color:#06d6a0'>A${price/1e6:.2f}M</div>
            <div class='result-sub'>A${price:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class='model-footer'>
        {best_cls['name']} (Acc {best_cls['accuracy']*100:.1f}%)
        &nbsp;·&nbsp;
        {best_reg['name']} (R² {best_reg['r2']:.3f})
    </div>
    """, unsafe_allow_html=True)