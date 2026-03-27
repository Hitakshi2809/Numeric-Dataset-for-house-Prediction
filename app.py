"""
🏠 Melbourne Housing Market — Streamlit Dashboard
Run: streamlit run app.py
"""

import streamlit as st
import json, os, pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Melbourne Housing ML", page_icon="🏠",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap');
html,body,[class*="css"]{font-family:'Syne',sans-serif;}
.stApp{background-color:#0d0d0d;color:#f0f0f0;}
.metric-card{background:linear-gradient(135deg,#1a1a2e,#16213e);
    border:1px solid #e94560;border-radius:12px;padding:20px;text-align:center;margin:8px 0;}
.metric-value{font-size:2rem;font-weight:800;color:#e94560;}
.metric-label{font-size:0.75rem;color:#aaa;font-family:'Space Mono',monospace;letter-spacing:1px;}
.section-header{background:linear-gradient(90deg,#e94560,#0f3460);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
    font-size:1.8rem;font-weight:800;margin:20px 0 10px 0;}
.predict-box{background:#1a1a2e;border:2px solid #e94560;border-radius:16px;
    padding:24px;margin:12px 0;text-align:center;}
.pred-label{font-size:2rem;font-weight:800;}
.pred-conf{font-size:0.95rem;color:#aaa;font-family:monospace;}
</style>
""", unsafe_allow_html=True)

PALETTE     = ['#e94560','#0f3460','#533483','#06d6a0','#ffd166']
OUTPUT_DIR  = './outputs'
BG          = '#1a1a2e'
CLASS_NAMES = ['Budget','Mid-Range','Premium','Luxury']
CLASS_EMOJI = ['🟢','🟡','🟠','🔴']

def set_style():
    plt.rcParams.update({
        'figure.facecolor':BG,'axes.facecolor':BG,'axes.edgecolor':'#333',
        'axes.labelcolor':'#ccc','text.color':'#ccc','xtick.color':'#aaa',
        'ytick.color':'#aaa','grid.color':'#2a2a2a','grid.alpha':0.5,
    })

# ── Load results ─────────────────────────────────────
@st.cache_data
def load_results():
    cls, reg = {}, {}
    cp = os.path.join(OUTPUT_DIR,'cls_results.json')
    rp = os.path.join(OUTPUT_DIR,'reg_results.json')
    if os.path.exists(cp):
        with open(cp) as f: cls = json.load(f)
    if os.path.exists(rp):
        with open(rp) as f: reg = json.load(f)
    return cls, reg

@st.cache_resource
def load_pkl_models():
    models = {}
    if not os.path.exists(OUTPUT_DIR): return models
    for f in os.listdir(OUTPUT_DIR):
        if f.endswith('.pkl') and ('cls_' in f or 'reg_' in f):
            with open(os.path.join(OUTPUT_DIR,f),'rb') as fp:
                d = pickle.load(fp)
            models[d['model_name']] = d
    return models

@st.cache_resource
def load_scaler():
    p = os.path.join(OUTPUT_DIR,'scaler.pkl')
    if not os.path.exists(p): return None, []
    with open(p,'rb') as f: d=pickle.load(f)
    return d['scaler'], d['feature_cols']

# ── Header ────────────────────────────────────────────
st.markdown("""
<h1 style='text-align:center;color:#e94560;font-size:3rem;font-weight:800;margin-bottom:0'>
🏠 Melbourne Housing Market
</h1>
<p style='text-align:center;color:#aaa;font-family:monospace;margin-top:4px'>
Price Prediction & Classification · Multi-Algorithm ML Dashboard
</p>""", unsafe_allow_html=True)
st.markdown("---")

if not os.path.exists(os.path.join(OUTPUT_DIR,'cls_results.json')):
    st.error("⚠️ Run training first:")
    st.code("python train.py", language="bash")
    st.stop()

cls_results, reg_results = load_results()
pkl_models               = load_pkl_models()
scaler, feat_cols        = load_scaler()

cls_names = list(cls_results.keys())
reg_names = list(reg_results.keys())

with st.sidebar:
    st.markdown("### ⚙️ Controls")
    sel_cls = st.multiselect("Classification Models", cls_names, default=cls_names)
    sel_reg = st.multiselect("Regression Models",     reg_names, default=reg_names)
    st.markdown("---")
    st.markdown("### 📊 Dataset")
    st.info("Melbourne Housing Market\nKaggle · Anthony Pino\n21 columns · ~34k rows")
    st.markdown("### 🎯 Tasks")
    st.markdown("**Classification**: Price Band\n\n🟢 Budget | 🟡 Mid | 🟠 Premium | 🔴 Luxury")
    st.markdown("**Regression**: Exact Price (AUD)")

cls_f = {k:v for k,v in cls_results.items() if k in sel_cls}
reg_f = {k:v for k,v in reg_results.items() if k in sel_reg}

# ── Tabs ──────────────────────────────────────────────
tabs = st.tabs(["🏆 Overview","🔮 Predict","🔲 Confusion Matrix",
                "📉 ROC Curves","📊 Feature Importance",
                "📈 Regression","🖼️ All Graphs"])
tab1,tab2,tab3,tab4,tab5,tab6,tab7 = tabs

# ══ TAB 1: OVERVIEW ══════════════════════════════════
with tab1:
    st.markdown("<div class='section-header'>Model Performance Overview</div>",
                unsafe_allow_html=True)

    st.markdown("#### 🎯 Classification — Price Band Prediction")
    if cls_f:
        best_cls = max(cls_f, key=lambda n: cls_f[n]['accuracy'])
        cols = st.columns(len(cls_f))
        for col,(name,r) in zip(cols,cls_f.items()):
            badge = " 🏆" if name==best_cls else ""
            with col:
                st.markdown(f"<div class='metric-card'>"
                            f"<div class='metric-value'>{r['accuracy']*100:.1f}%</div>"
                            f"<div class='metric-label'>{name}{badge}</div></div>",
                            unsafe_allow_html=True)

        c1,c2,c3,c4 = st.columns(4)
        for col, label, key in zip([c1,c2,c3,c4],
            ['BEST F1','BEST PRECISION','BEST RECALL','BEST AUC'],
            ['f1','precision','recall','roc_auc']):
            val = max(r[key] for r in cls_f.values())
            with col:
                st.markdown(f"<div class='metric-card'>"
                            f"<div class='metric-value'>{val:.3f}</div>"
                            f"<div class='metric-label'>{label}</div></div>",
                            unsafe_allow_html=True)

        set_style()
        metric_names = ['Accuracy','F1','Precision','Recall','AUC']
        x = np.arange(len(metric_names)); n_m = len(cls_f); width = 0.8/n_m
        fig, ax = plt.subplots(figsize=(13,5))
        for i,(name,r) in enumerate(cls_f.items()):
            vals   = [r['accuracy'],r['f1'],r['precision'],r['recall'],r['roc_auc']]
            offset = (i-n_m/2+0.5)*width
            bars   = ax.bar(x+offset,vals,width,label=name,
                            color=PALETTE[i%5],edgecolor='#0d0d0d')
            for bar,val in zip(bars,vals):
                ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.005,
                        f'{val:.2f}',ha='center',va='bottom',fontsize=8,color='#ccc')
        ax.set_xticks(x); ax.set_xticklabels(metric_names,fontsize=11)
        ax.set_ylim(0,1.15); ax.set_ylabel('Score')
        ax.set_title('Classification Metrics',color='#f0f0f0',fontsize=14,fontweight='bold')
        ax.legend(fontsize=9,framealpha=0.3); ax.grid(axis='y',alpha=0.3)
        fig.patch.set_facecolor(BG)
        st.pyplot(fig); plt.close()

    st.markdown("#### 📐 Regression — Price Prediction")
    if reg_f:
        best_reg = max(reg_f, key=lambda n: reg_f[n]['r2'])
        cols = st.columns(len(reg_f))
        for col,(name,r) in zip(cols,reg_f.items()):
            badge = " 🏆" if name==best_reg else ""
            with col:
                st.markdown(f"<div class='metric-card'>"
                            f"<div class='metric-value'>R²={r['r2']:.3f}</div>"
                            f"<div class='metric-label'>{name}{badge}</div></div>",
                            unsafe_allow_html=True)

        import pandas as pd
        df_reg = pd.DataFrame({
            name:{'R²':r['r2'],'MAE':r['mae'],'RMSE':r['rmse'],'MAPE%':r['mape']}
            for name,r in reg_f.items()}).T
        st.dataframe(df_reg.style.format({'R²':'{:.4f}','MAE':'{:,.0f}',
                                          'RMSE':'{:,.0f}','MAPE%':'{:.2f}'})\
                     .background_gradient(cmap='YlOrRd',subset=['R²']),
                     use_container_width=True)

# ══ TAB 2: PREDICT ════════════════════════════════════
with tab2:
    st.markdown("<div class='section-header'>Live House Price Prediction</div>",
                unsafe_allow_html=True)

    task = st.radio("Task", ["🎯 Classify Price Band","💰 Predict Exact Price"], horizontal=True)

    c1,c2,c3 = st.columns(3)
    with c1:
        rooms    = st.number_input("Rooms", 1, 10, 3)
        bathroom = st.number_input("Bathrooms", 1, 5, 2)
        car      = st.number_input("Car Spaces", 0, 5, 1)
        landsize = st.number_input("Land Size (m²)", 0, 50000, 550)
    with c2:
        build_area = st.number_input("Building Area (m²)", 0, 1000, 120)
        distance   = st.number_input("Distance from CBD (km)", 0.0, 50.0, 10.0)
        year_built = st.number_input("Year Built", 1880, 2023, 1990)
        prop_count = st.number_input("Property Count", 100, 30000, 5000)
    with c3:
        lat   = st.number_input("Latitude",  -38.5, -37.0, -37.8)
        lon   = st.number_input("Longitude",  144.0, 145.5, 145.0)
        type_enc   = st.selectbox("Property Type", [0,1,2],
                                  format_func=lambda x:['House','Townhouse','Unit'][x])
        region_enc = st.selectbox("Region", list(range(8)),
                                  format_func=lambda x: [
                                    'Eastern Metro','Eastern Victoria',
                                    'Northern Metro','Northern Victoria',
                                    'South-East Metro','Southern Metro',
                                    'Western Metro','Western Victoria'][x])

    if st.button("🔮 Predict", use_container_width=True):
        sale_year  = 2017; sale_month = 6
        house_age  = sale_year - year_built
        build_ratio= build_area / max(landsize, 1)
        rooms_pkm  = rooms / max(distance, 0.1)

        inp_map = {
            'Rooms':rooms,'Bathroom':bathroom,'Car':car,
            'Landsize':landsize,'BuildingArea':build_area,
            'Distance':distance,'Propertycount':prop_count,
            'Lattitude':lat,'Longtitude':lon,
            'sale_year':sale_year,'sale_month':sale_month,
            'house_age':house_age,'build_ratio':build_ratio,
            'rooms_per_km':rooms_pkm,'Bedroom2':rooms,
            'Type_enc':type_enc,'Method_enc':0,
            'Regionname_enc':region_enc,'CouncilArea_enc':0,
        }
        row = np.array([[inp_map.get(f,0) for f in feat_cols]])

        if "Classify" in task:
            # Use best classifier
            best = max(cls_results, key=lambda n: cls_results[n]['accuracy'])
            md   = pkl_models.get(best)
            if md:
                m   = md['model']
                Xin = scaler.transform(row) if md['scaled'] and scaler else row
                prob = m.predict_proba(Xin)[0]
                pred = int(np.argmax(prob))
                conf = float(prob[pred])
                emoji = CLASS_EMOJI[pred]
                label = CLASS_NAMES[pred]
                color = [PALETTE[3],PALETTE[5] if len(PALETTE)>5 else PALETTE[1],
                         PALETTE[4],PALETTE[0]][pred]

                st.markdown(f"""
                <div class='predict-box'>
                    <div class='pred-label'>{emoji} {label}</div>
                    <div class='pred-conf'>Confidence: {conf*100:.1f}%  |  Model: {best}</div>
                </div>""", unsafe_allow_html=True)

                set_style()
                fig, ax = plt.subplots(figsize=(8,2.5))
                ax.barh(CLASS_NAMES, prob*100,
                        color=PALETTE[:4], edgecolor='#0d0d0d')
                ax.set_xlim(0,100); ax.set_xlabel('Probability %')
                ax.set_title('Class Probabilities', color='#f0f0f0', fontsize=11)
                fig.patch.set_facecolor(BG)
                st.pyplot(fig); plt.close()
        else:
            best = max(reg_results, key=lambda n: reg_results[n]['r2'])
            md   = pkl_models.get(best)
            if md:
                m   = md['model']
                Xin = scaler.transform(row) if md['scaled'] and scaler else row
                price = float(m.predict(Xin)[0])
                cat   = int(np.digitize(price,
                        [df['Price'].quantile(q) if 'Price' in dir() else price*0.5
                         for q in [0.25,0.5,0.75]]))
                cat = min(cat, 3)

                st.markdown(f"""
                <div class='predict-box'>
                    <div class='pred-label' style='color:#06d6a0'>
                        💰 ${price/1e6:.3f}M AUD
                    </div>
                    <div class='pred-conf'>
                        = ${price:,.0f} AUD  |  Model: {best}
                    </div>
                </div>""", unsafe_allow_html=True)

# ══ TAB 3: CONFUSION MATRIX ══════════════════════════
with tab3:
    st.markdown("<div class='section-header'>Confusion Matrices</div>", unsafe_allow_html=True)
    if cls_f:
        set_style()
        cols_cm = st.columns(min(len(cls_f),3))
        for idx,(name,r) in enumerate(cls_f.items()):
            with cols_cm[idx%3]:
                fig, ax = plt.subplots(figsize=(5,4))
                cm = np.array(r['cm'])
                cm_pct = cm.astype(float)/cm.sum(axis=1,keepdims=True)*100
                sns.heatmap(cm,annot=False,
                            cmap=['Blues','Oranges','Greens','Reds','Purples'][idx%5],
                            ax=ax,xticklabels=CLASS_NAMES,yticklabels=CLASS_NAMES,
                            linewidths=0.5,linecolor='#0d0d0d')
                for i in range(4):
                    for j in range(4):
                        ax.text(j+0.5,i+0.35,str(cm[i,j]),ha='center',va='center',
                                fontsize=8,fontweight='bold',color='white')
                        ax.text(j+0.5,i+0.65,f'({cm_pct[i,j]:.0f}%)',ha='center',
                                va='center',fontsize=6,color='#ddd')
                ax.set_title(name,color='#f0f0f0',fontsize=11,fontweight='bold')
                ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
                plt.setp(ax.get_xticklabels(),rotation=30,ha='right',fontsize=8)
                plt.setp(ax.get_yticklabels(),fontsize=8)
                fig.patch.set_facecolor(BG)
                st.pyplot(fig); plt.close()

        for name,r in cls_f.items():
            with st.expander(f"📋 {name} — Classification Report"):
                import pandas as pd
                rdf = pd.DataFrame(r['report']).T
                st.dataframe(rdf.style.format('{:.3f}').background_gradient(
                    cmap='YlOrRd',subset=['precision','recall','f1-score']),
                    use_container_width=True)

# ══ TAB 4: ROC CURVES ════════════════════════════════
with tab4:
    st.markdown("<div class='section-header'>ROC Curves</div>", unsafe_allow_html=True)
    if cls_f:
        set_style()
        col1,col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(6,5))
            for i,(name,r) in enumerate(cls_f.items()):
                ax.bar(i, r['roc_auc'], color=PALETTE[i%5], edgecolor='#0d0d0d', width=0.5)
                ax.text(i, r['roc_auc']+0.005, f"{r['roc_auc']:.3f}",
                        ha='center',fontsize=9,color='#ccc',fontweight='bold')
            ax.set_xticks(range(len(cls_f)))
            ax.set_xticklabels(list(cls_f.keys()),rotation=15,ha='right',fontsize=9)
            ax.set_ylim(0,1.15); ax.set_ylabel('Macro ROC-AUC')
            ax.set_title('AUC Comparison',color='#f0f0f0',fontsize=12,fontweight='bold')
            ax.grid(axis='y',alpha=0.3)
            fig.patch.set_facecolor(BG)
            st.pyplot(fig); plt.close()

        with col2:
            best_name = max(cls_f, key=lambda n: cls_f[n]['accuracy'])
            r_b = cls_f[best_name]
            y_true_arr = np.array(r_b['y_true'])
            y_prob_arr = np.array(r_b['y_prob'])
            fig, ax = plt.subplots(figsize=(6,5))
            for c_idx,c_name in enumerate(CLASS_NAMES):
                fpr,tpr,_ = roc_curve((y_true_arr==c_idx).astype(int),
                                       y_prob_arr[:,c_idx])
                ax.plot(fpr,tpr,color=PALETTE[c_idx],lw=2,label=c_name)
            ax.plot([0,1],[0,1],'--',color='#555',lw=1.5)
            ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
            ax.set_title(f'ROC per Class — {best_name}',
                         color='#f0f0f0',fontsize=11,fontweight='bold')
            ax.legend(fontsize=9,framealpha=0.3); ax.grid(alpha=0.3)
            fig.patch.set_facecolor(BG)
            st.pyplot(fig); plt.close()

# ══ TAB 5: FEATURE IMPORTANCE ════════════════════════
with tab5:
    st.markdown("<div class='section-header'>Feature Importance</div>", unsafe_allow_html=True)
    for name,md in pkl_models.items():
        m = md.get('model')
        if m and hasattr(m,'feature_importances_') and feat_cols:
            set_style()
            fi = pd.DataFrame({'Feature':feat_cols,'Importance':m.feature_importances_})\
                   .sort_values('Importance',ascending=True).tail(15)
            fig, ax = plt.subplots(figsize=(9,5))
            clrs = plt.cm.RdYlGn(np.linspace(0.2,0.9,len(fi)))
            ax.barh(fi['Feature'],fi['Importance'],color=clrs)
            ax.set_title(f'Feature Importance — {name}',
                         color='#f0f0f0',fontsize=12,fontweight='bold')
            ax.set_xlabel('Importance')
            fig.patch.set_facecolor(BG)
            st.pyplot(fig); plt.close()

# ══ TAB 6: REGRESSION ════════════════════════════════
with tab6:
    st.markdown("<div class='section-header'>Regression Analysis</div>", unsafe_allow_html=True)
    if reg_f:
        set_style()
        col1,col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(6,4))
            r2_vals = [reg_f[n]['r2'] for n in reg_f]
            bars = ax.bar(list(reg_f.keys()),r2_vals,
                          color=PALETTE[:len(reg_f)],edgecolor='#0d0d0d',width=0.5)
            for bar,val in zip(bars,r2_vals):
                ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.005,
                        f'{val:.4f}',ha='center',fontsize=10,color='#ccc',fontweight='bold')
            ax.set_ylim(0,1.1); ax.set_title('R² Score',color='#f0f0f0',fontsize=12,fontweight='bold')
            ax.grid(axis='y',alpha=0.3)
            plt.setp(ax.get_xticklabels(),rotation=15,ha='right',fontsize=8)
            fig.patch.set_facecolor(BG)
            st.pyplot(fig); plt.close()

        with col2:
            fig, ax = plt.subplots(figsize=(6,4))
            mae_vals = [reg_f[n]['mae']/1000 for n in reg_f]
            bars = ax.bar(list(reg_f.keys()),mae_vals,
                          color=PALETTE[:len(reg_f)],edgecolor='#0d0d0d',width=0.5)
            for bar,val in zip(bars,mae_vals):
                ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()*1.02,
                        f'${val:.0f}k',ha='center',fontsize=10,color='#ccc',fontweight='bold')
            ax.set_title('MAE (AUD thousands)',color='#f0f0f0',fontsize=12,fontweight='bold')
            ax.set_ylabel('MAE (k AUD)'); ax.grid(axis='y',alpha=0.3)
            plt.setp(ax.get_xticklabels(),rotation=15,ha='right',fontsize=8)
            fig.patch.set_facecolor(BG)
            st.pyplot(fig); plt.close()

        # Actual vs Predicted (best model)
        best_reg = max(reg_f, key=lambda n: reg_f[n]['r2'])
        r_b = reg_f[best_reg]
        set_style()
        fig, (ax1,ax2) = plt.subplots(1,2,figsize=(12,5))
        fig.suptitle(f'Regression Analysis — {best_reg}',
                     fontsize=13,fontweight='bold',color='#f0f0f0')

        actual = np.array(r_b['y_true'])/1e6
        pred   = np.array(r_b['y_pred'])/1e6
        ax1.scatter(actual,pred,alpha=0.2,s=8,color=PALETTE[0],edgecolors='none')
        mn,mx = min(actual.min(),pred.min()),max(actual.max(),pred.max())
        ax1.plot([mn,mx],[mn,mx],'--',color=PALETTE[3],lw=2)
        ax1.set_xlabel('Actual (M AUD)'); ax1.set_ylabel('Predicted (M AUD)')
        ax1.set_title(f'Actual vs Predicted (R²={r_b["r2"]:.3f})',
                      color='#f0f0f0',fontweight='bold')

        res = (np.array(r_b['y_true'])-np.array(r_b['y_pred']))/1e6
        ax2.scatter(pred,res,alpha=0.2,s=8,color=PALETTE[2],edgecolors='none')
        ax2.axhline(0,color=PALETTE[0],lw=2,ls='--')
        ax2.set_xlabel('Predicted (M AUD)'); ax2.set_ylabel('Residual (M AUD)')
        ax2.set_title('Residuals',color='#f0f0f0',fontweight='bold')

        fig.patch.set_facecolor(BG)
        st.pyplot(fig); plt.close()

# ══ TAB 7: ALL GRAPHS ════════════════════════════════
with tab7:
    st.markdown("<div class='section-header'>All Saved Graphs</div>", unsafe_allow_html=True)
    if os.path.exists(OUTPUT_DIR):
        all_pngs = sorted([f for f in os.listdir(OUTPUT_DIR) if f.endswith('.png')])
        for fname in all_pngs:
            caption = fname.replace('_',' ').replace('.png','').title()
            st.image(os.path.join(OUTPUT_DIR,fname),
                     caption=caption, use_container_width=True)
    else:
        st.warning("Run python train.py first to generate graphs.")

st.markdown("---")
st.markdown("<p style='text-align:center;color:#555;font-family:monospace;font-size:0.8rem'>"
            "Melbourne Housing Market · sklearn · pkl · Streamlit</p>", unsafe_allow_html=True)