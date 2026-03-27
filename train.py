"""
🏠 Melbourne Housing Market — ML Pipeline
  Dataset  : https://www.kaggle.com/datasets/anthonypino/melbourne-housing-market
  Task 1   : REGRESSION  — predict house Price
  Task 2   : CLASSIFICATION — predict Price Category (Budget/Mid/Premium/Luxury)
  Models   : Random Forest, Gradient Boosting, Linear/Logistic Regression, Decision Tree, XGBoost
  Saves    : Each model as .pkl
  Graphs   : EDA, Confusion Matrix, ROC, Metrics, Feature Importance, Residuals, etc.
"""

import os, json, pickle, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection   import train_test_split, cross_val_score, KFold
from sklearn.preprocessing     import LabelEncoder, StandardScaler
from sklearn.ensemble          import RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble          import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model      import LogisticRegression, LinearRegression, Ridge
from sklearn.tree              import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics           import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve,
    mean_absolute_error, mean_squared_error, r2_score
)

try:
    from xgboost import XGBClassifier, XGBRegressor
    XGB_OK = True
except ImportError:
    XGB_OK = False
    print("⚠️  XGBoost not installed (pip install xgboost) — skipping")

OUTPUT_DIR = './outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)
PALETTE = ['#e94560','#0f3460','#533483','#06d6a0','#ffd166']
BG      = '#1a1a2e'

print("="*65)
print("  🏠 Melbourne Housing Market — ML Pipeline")
print("="*65)

# ─────────────────────────────────────────────
# STEP 1 · LOAD DATA
# ─────────────────────────────────────────────
print("\n[1/9] Loading dataset...")



df = pd.read_csv("Data/Melbourne_housing_FULL.csv")
print(f"  ✅ Shape   : {df.shape}")
print(f"  ✅ Columns : {list(df.columns)}")
print(f"\n{df.dtypes}\n")

# ─────────────────────────────────────────────
# STEP 2 · CLEAN & PREPROCESS
# ─────────────────────────────────────────────
print("\n[2/9] Cleaning data...")

# Drop rows without Price (target)
df = df.dropna(subset=['Price'])
df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
df = df.dropna(subset=['Price'])
print(f"  After dropping null Price : {len(df):,} rows")

# Fill numeric NaNs with median
num_cols = ['Rooms','Bathroom','Car','Landsize','BuildingArea',
            'YearBuilt','Lattitude','Longtitude','Propertycount',
            'Distance','Postcode','Bedroom2']
for col in num_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col].fillna(df[col].median(), inplace=True)

# Date features
if 'Date' in df.columns:
    df['Date']        = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
    df['sale_year']   = df['Date'].dt.year.fillna(2017).astype(int)
    df['sale_month']  = df['Date'].dt.month.fillna(6).astype(int)
    df['house_age']   = df['sale_year'] - df['YearBuilt'].fillna(1970)

# Encode categoricals
le = LabelEncoder()
cat_encode = ['Type','Method','Regionname','CouncilArea']
for col in cat_encode:
    if col in df.columns:
        df[col+'_enc'] = le.fit_transform(df[col].astype(str))

# Derived features
if 'Landsize' in df.columns and 'BuildingArea' in df.columns:
    df['build_ratio'] = df['BuildingArea'] / df['Landsize'].replace(0, 1)
if 'Rooms' in df.columns and 'Distance' in df.columns:
    df['rooms_per_km'] = df['Rooms'] / df['Distance'].replace(0, 0.1)

# ─────────────────────────────────────────────
# STEP 3 · DEFINE FEATURES
# ─────────────────────────────────────────────
print("\n[3/9] Defining features...")

feature_cols = [c for c in [
    'Rooms','Bathroom','Car','Landsize','BuildingArea',
    'Distance','Propertycount','Lattitude','Longtitude',
    'sale_year','sale_month','house_age','build_ratio','rooms_per_km',
    'Type_enc','Method_enc','Regionname_enc','CouncilArea_enc'
] if c in df.columns]

print(f"  ✅ {len(feature_cols)} features: {feature_cols}")

X_raw = df[feature_cols].copy().fillna(0)
y_reg = df['Price'].copy()

# ── CLASSIFICATION TARGET ── Price bands ──────
q1,q2,q3 = y_reg.quantile([0.25,0.5,0.75])
def price_band(p):
    if p < q1:   return 0  # Budget
    elif p < q2: return 1  # Mid
    elif p < q3: return 2  # Premium
    else:        return 3  # Luxury

y_cls = y_reg.apply(price_band)
CLASS_NAMES = ['Budget','Mid-Range','Premium','Luxury']
print(f"\n  Price bands (classification target):")
for i,name in enumerate(CLASS_NAMES):
    n = (y_cls==i).sum()
    print(f"    {name:12} : {n:,} ({n/len(y_cls)*100:.1f}%)")

# ─────────────────────────────────────────────
# STEP 4 · SPLIT & SCALE
# ─────────────────────────────────────────────
print("\n[4/9] Splitting and scaling...")

X_train, X_test, y_cls_train, y_cls_test = train_test_split(
    X_raw, y_cls, test_size=0.2, random_state=42, stratify=y_cls)
_, _, y_reg_train, y_reg_test = train_test_split(
    X_raw, y_reg, test_size=0.2, random_state=42)

scaler    = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

print(f"  Train: {len(X_train):,} | Test: {len(X_test):,}")

with open(os.path.join(OUTPUT_DIR,'scaler.pkl'),'wb') as f:
    pickle.dump({'scaler':scaler,'feature_cols':feature_cols,
                 'class_names':CLASS_NAMES}, f)
print("  ✅ Saved: scaler.pkl")

# ─────────────────────────────────────────────
# STEP 5 · CLASSIFICATION MODELS
# ─────────────────────────────────────────────
print("\n[5/9] Training Classification models...\n")

cls_configs = {
    'Random Forest'     : (RandomForestClassifier(n_estimators=100,random_state=42,n_jobs=-1), False),
   
    'Logistic Regression':(LogisticRegression(max_iter=1000,random_state=42,multi_class='auto'),True),
    'Decision Tree'     : (DecisionTreeClassifier(max_depth=8,random_state=42),                False),
}
cls_results = {}

for name,(model,scaled) in cls_configs.items():
    print(f"  ▶ {name} ...")
    Xtr = X_train_s if scaled else X_train
    Xte = X_test_s  if scaled else X_test

    model.fit(Xtr, y_cls_train)
    y_pred = model.predict(Xte)
    y_prob = model.predict_proba(Xte)
    cv     = cross_val_score(model, Xtr, y_cls_train, cv=5, scoring='accuracy')

    # Multiclass ROC-AUC (one-vs-rest)
    try:
        auc = roc_auc_score(y_cls_test, y_prob, multi_class='ovr', average='macro')
    except Exception:
        auc = 0.0

    r = {
        'model':     model, 'scaled': scaled,
        'y_pred':    y_pred.tolist(),
        'y_prob':    y_prob.tolist(),
        'y_true':    y_cls_test.tolist(),
        'accuracy':  accuracy_score(y_cls_test, y_pred),
        'f1':        f1_score(y_cls_test, y_pred, average='weighted'),
        'precision': precision_score(y_cls_test, y_pred, average='weighted', zero_division=0),
        'recall':    recall_score(y_cls_test, y_pred, average='weighted'),
        'roc_auc':   auc,
        'cm':        confusion_matrix(y_cls_test, y_pred).tolist(),
        'cv_mean':   cv.mean(), 'cv_std': cv.std(),
        'report':    classification_report(y_cls_test, y_pred,
                         target_names=CLASS_NAMES, output_dict=True),
    }
    cls_results[name] = r

    print(f"     Acc={r['accuracy']:.4f} F1={r['f1']:.4f} AUC={auc:.4f} CV={cv.mean():.4f}±{cv.std():.4f}")

    pkl = os.path.join(OUTPUT_DIR, f'cls_{name.replace(" ","_")}.pkl')
    with open(pkl,'wb') as f:
        pickle.dump({'model':model,'model_name':name,'task':'classification',
                     'scaled':scaled,'feature_cols':feature_cols,
                     'class_names':CLASS_NAMES,
                     'accuracy':r['accuracy'],'f1':r['f1'],'roc_auc':auc}, f)
    print(f"     💾 {pkl}")

# ─────────────────────────────────────────────
# STEP 6 · REGRESSION MODELS
# ─────────────────────────────────────────────
print("\n[6/9] Training Regression models...\n")

reg_configs = {
    'RF Regressor'  : (RandomForestRegressor(n_estimators=100,random_state=42,n_jobs=-1), False),
    'GB Regressor'  : (GradientBoostingRegressor(n_estimators=100,random_state=42),       False),
    'Ridge Regression':(Ridge(alpha=1.0),                                                 True),
    'DT Regressor'  : (DecisionTreeRegressor(max_depth=8,random_state=42),                False),
}
if XGB_OK:
    reg_configs['XGB Regressor'] = (
        XGBRegressor(n_estimators=100,random_state=42,verbosity=0), False)

reg_results = {}

for name,(model,scaled) in reg_configs.items():
    print(f"  ▶ {name} ...")
    Xtr = X_train_s if scaled else X_train
    Xte = X_test_s  if scaled else X_test

    model.fit(Xtr, y_reg_train)
    y_pred = model.predict(Xte)

    mae  = mean_absolute_error(y_reg_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_reg_test, y_pred))
    r2   = r2_score(y_reg_test, y_pred)
    mape = np.mean(np.abs((y_reg_test - y_pred) / y_reg_test.replace(0,1))) * 100

    reg_results[name] = {
        'model':model,'scaled':scaled,
        'y_pred':y_pred.tolist(),'y_true':y_reg_test.tolist(),
        'mae':mae,'rmse':rmse,'r2':r2,'mape':mape,
    }
    print(f"     MAE={mae:,.0f}  RMSE={rmse:,.0f}  R²={r2:.4f}  MAPE={mape:.2f}%")

    pkl = os.path.join(OUTPUT_DIR, f'reg_{name.replace(" ","_")}.pkl')
    with open(pkl,'wb') as f:
        pickle.dump({'model':model,'model_name':name,'task':'regression',
                     'scaled':scaled,'feature_cols':feature_cols,
                     'mae':mae,'rmse':rmse,'r2':r2,'mape':mape}, f)
    print(f"     💾 {pkl}")

# Save JSON for Streamlit
def serial(d):
    return {n:{k:v for k,v in r.items() if k!='model'} for n,r in d.items()}

with open(os.path.join(OUTPUT_DIR,'cls_results.json'),'w') as f:
    json.dump(serial(cls_results), f)
with open(os.path.join(OUTPUT_DIR,'reg_results.json'),'w') as f:
    json.dump(serial(reg_results), f)
print("\n  ✅ Saved: cls_results.json, reg_results.json")

# ─────────────────────────────────────────────
# STEP 7 · METRICS TABLE
# ─────────────────────────────────────────────
print("\n[7/9] Metrics Summary")
print("\n── CLASSIFICATION ──")
print("="*80)
print(f"{'Model':<22}{'Accuracy':>9}{'F1':>8}{'Precision':>11}{'Recall':>8}{'AUC':>8}{'CV':>8}")
print("="*80)
for name,r in cls_results.items():
    print(f"{name:<22}{r['accuracy']:>9.4f}{r['f1']:>8.4f}"
          f"{r['precision']:>11.4f}{r['recall']:>8.4f}"
          f"{r['roc_auc']:>8.4f}{r['cv_mean']:>8.4f}")
print("="*80)

print("\n── REGRESSION ──")
print("="*70)
print(f"{'Model':<22}{'MAE':>12}{'RMSE':>14}{'R²':>8}{'MAPE%':>8}")
print("="*70)
for name,r in reg_results.items():
    print(f"{name:<22}{r['mae']:>12,.0f}{r['rmse']:>14,.0f}{r['r2']:>8.4f}{r['mape']:>8.2f}")
print("="*70)

# ─────────────────────────────────────────────
# STEP 8 · ALL GRAPHS
# ─────────────────────────────────────────────
print("\n[8/9] Saving graphs...")

def style():
    plt.rcParams.update({
        'figure.facecolor':BG,'axes.facecolor':BG,'axes.edgecolor':'#333',
        'axes.labelcolor':'#ccc','text.color':'#ccc','xtick.color':'#aaa',
        'ytick.color':'#aaa','grid.color':'#2a2a2a','grid.alpha':0.5,
    })

def save(fname):
    plt.savefig(os.path.join(OUTPUT_DIR,fname),dpi=150,bbox_inches='tight')
    plt.close()
    print(f"  ✅ {fname}")

# ── EDA 1: Price Distribution ─────────────────
style()
fig, axes = plt.subplots(1,3,figsize=(16,4))
fig.suptitle('Price Distribution — Melbourne Housing', fontsize=14,
             fontweight='bold', color='#f0f0f0')

axes[0].hist(y_reg/1e6, bins=60, color=PALETTE[0], edgecolor='#0d0d0d', alpha=0.85)
axes[0].set_title('Price Distribution (M$)', color='#f0f0f0', fontweight='bold')
axes[0].set_xlabel('Price (Million AUD)'); axes[0].set_ylabel('Count')

axes[1].hist(np.log1p(y_reg), bins=60, color=PALETTE[1], edgecolor='#0d0d0d', alpha=0.85)
axes[1].set_title('Log-Price Distribution', color='#f0f0f0', fontweight='bold')
axes[1].set_xlabel('Log(Price)')

counts = y_cls.value_counts().sort_index()
axes[2].bar(CLASS_NAMES, counts.values,
            color=PALETTE[:4], edgecolor='#0d0d0d')
for i,v in enumerate(counts.values):
    axes[2].text(i, v+50, f'{v:,}', ha='center', color='#ccc', fontsize=9, fontweight='bold')
axes[2].set_title('Price Category Counts', color='#f0f0f0', fontweight='bold')
axes[2].set_ylabel('Count')
plt.setp(axes[2].get_xticklabels(), rotation=15, ha='right')

fig.patch.set_facecolor(BG); plt.tight_layout(); save('eda_price_distribution.png')

# ── EDA 2: Feature Distributions ──────────────
style()
plot_cols = [c for c in ['Rooms','Bathroom','Car','Distance',
                         'Landsize','BuildingArea'] if c in df.columns]
fig, axes = plt.subplots(2, 3, figsize=(15,8))
fig.suptitle('Feature Distributions', fontsize=14, fontweight='bold', color='#f0f0f0')
for ax, col in zip(axes.flat, plot_cols):
    ax.hist(df[col].dropna(), bins=40, color=PALETTE[2], edgecolor='#0d0d0d', alpha=0.8)
    ax.set_title(col, color='#f0f0f0', fontweight='bold')
    ax.set_xlabel(col)
for ax in axes.flat[len(plot_cols):]: ax.set_visible(False)
fig.patch.set_facecolor(BG); plt.tight_layout(); save('eda_feature_distributions.png')

# ── EDA 3: Price vs Key Features ──────────────
style()
scatter_cols = [c for c in ['Rooms','Distance','BuildingArea','Landsize'] if c in df.columns]
fig, axes = plt.subplots(1, len(scatter_cols), figsize=(5*len(scatter_cols), 4))
fig.suptitle('Price vs Key Features', fontsize=13, fontweight='bold', color='#f0f0f0')
sample = df.sample(min(3000, len(df)), random_state=42)
for ax, col in zip(axes, scatter_cols):
    ax.scatter(sample[col], sample['Price']/1e6, alpha=0.3,
               color=PALETTE[0], edgecolors='none', s=8)
    ax.set_xlabel(col); ax.set_ylabel('Price (M AUD)')
    ax.set_title(col, color='#f0f0f0', fontweight='bold')
fig.patch.set_facecolor(BG); plt.tight_layout(); save('eda_price_vs_features.png')

# ── EDA 4: Price by Type & Region ─────────────
style()
fig, axes = plt.subplots(1,2,figsize=(14,5))
fig.suptitle('Price by Type & Region', fontsize=13, fontweight='bold', color='#f0f0f0')

if 'Type' in df.columns:
    type_price = df.groupby('Type')['Price'].median()/1e6
    axes[0].bar(type_price.index, type_price.values,
                color=PALETTE[:len(type_price)], edgecolor='#0d0d0d')
    for i,v in enumerate(type_price.values):
        axes[0].text(i, v+0.02, f'${v:.2f}M', ha='center', color='#ccc', fontsize=9)
    axes[0].set_title('Median Price by Property Type', color='#f0f0f0', fontweight='bold')
    axes[0].set_ylabel('Median Price (M AUD)')

if 'Regionname' in df.columns:
    reg_price = df.groupby('Regionname')['Price'].median()/1e6
    reg_price = reg_price.sort_values(ascending=True)
    axes[1].barh(range(len(reg_price)), reg_price.values,
                 color=PALETTE[1], edgecolor='#0d0d0d')
    axes[1].set_yticks(range(len(reg_price)))
    axes[1].set_yticklabels(reg_price.index, fontsize=8)
    axes[1].set_title('Median Price by Region', color='#f0f0f0', fontweight='bold')
    axes[1].set_xlabel('Median Price (M AUD)')

fig.patch.set_facecolor(BG); plt.tight_layout(); save('eda_price_by_type_region.png')

# ── EDA 5: Correlation Heatmap ────────────────
style()
num_heat = [c for c in feature_cols if X_raw[c].dtype in ['float64','int64','int32']]
corr     = X_raw[num_heat[:12]].assign(Price=y_reg).corr()
fig, ax  = plt.subplots(figsize=(12,9))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdYlGn_r', center=0, ax=ax,
            linewidths=0.5, linecolor='#0d0d0d', cbar_kws={'shrink':0.8}, annot_kws={'size':8})
ax.set_title('Feature Correlation Heatmap', color='#f0f0f0', fontsize=13, fontweight='bold')
fig.patch.set_facecolor(BG); plt.tight_layout(); save('eda_correlation_heatmap.png')

# ── CLS 1: Confusion Matrices ─────────────────
style()
n = len(cls_results)
cols_r = min(n,3); rows_r = (n+cols_r-1)//cols_r
fig, axes = plt.subplots(rows_r, cols_r, figsize=(6*cols_r, 5*rows_r))
axes = np.array(axes).flatten()
fig.suptitle('Confusion Matrices — Classification', fontsize=14,
             fontweight='bold', color='#f0f0f0')
for ax,(name,r),cmap in zip(axes,cls_results.items(),
                            ['Blues','Oranges','Greens','Reds','Purples']):
    cm = np.array(r['cm'])
    cm_pct = cm.astype(float)/cm.sum(axis=1,keepdims=True)*100
    sns.heatmap(cm, annot=False, cmap=cmap, ax=ax,
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                linewidths=0.5, linecolor='#0d0d0d')
    for i in range(4):
        for j in range(4):
            ax.text(j+0.5, i+0.35, str(cm[i,j]),
                    ha='center',va='center',fontsize=8,fontweight='bold',color='white')
            ax.text(j+0.5, i+0.65, f'({cm_pct[i,j]:.0f}%)',
                    ha='center',va='center',fontsize=6,color='#ddd')
    ax.set_title(name, color='#f0f0f0', fontsize=11, fontweight='bold')
    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right', fontsize=8)
    plt.setp(ax.get_yticklabels(), fontsize=8)
for ax in axes[n:]: ax.set_visible(False)
fig.patch.set_facecolor(BG); plt.tight_layout(); save('cls_confusion_matrices.png')

# ── CLS 2: Metrics Comparison ─────────────────
style()
metric_names = ['Accuracy','F1 Score','Precision','Recall','ROC-AUC']
x = np.arange(len(metric_names)); width = 0.8/n
fig, ax = plt.subplots(figsize=(14,5))
for i,(name,r) in enumerate(cls_results.items()):
    vals   = [r['accuracy'],r['f1'],r['precision'],r['recall'],r['roc_auc']]
    offset = (i-n/2+0.5)*width
    bars   = ax.bar(x+offset,vals,width,label=name,
                    color=PALETTE[i%len(PALETTE)],edgecolor='#0d0d0d')
    for bar,val in zip(bars,vals):
        ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.003,
                f'{val:.2f}',ha='center',va='bottom',fontsize=7,color='#ccc')
ax.set_xticks(x); ax.set_xticklabels(metric_names,fontsize=12)
ax.set_ylim(0,1.15); ax.set_ylabel('Score')
ax.set_title('Classification Metrics Comparison',color='#f0f0f0',fontsize=14,fontweight='bold')
ax.legend(fontsize=9,framealpha=0.3); ax.grid(axis='y',alpha=0.3)
fig.patch.set_facecolor(BG); plt.tight_layout(); save('cls_metrics_comparison.png')

# ── CLS 3: ROC Curves (one-vs-rest per class) ─
style()
fig, axes = plt.subplots(1,2,figsize=(14,5))
fig.suptitle('ROC Curves — Classification', fontsize=13, fontweight='bold', color='#f0f0f0')

# Macro AUC per model
for i,(name,r) in enumerate(cls_results.items()):
    axes[0].bar(i, r['roc_auc'], color=PALETTE[i%len(PALETTE)], edgecolor='#0d0d0d', width=0.5)
    axes[0].text(i, r['roc_auc']+0.005, f"{r['roc_auc']:.3f}",
                 ha='center',fontsize=9,color='#ccc',fontweight='bold')
axes[0].set_xticks(range(n))
axes[0].set_xticklabels(list(cls_results.keys()), rotation=15, ha='right', fontsize=9)
axes[0].set_ylim(0,1.15); axes[0].set_ylabel('Macro ROC-AUC')
axes[0].set_title('AUC Comparison', color='#f0f0f0', fontweight='bold')
axes[0].grid(axis='y',alpha=0.3)

# Best model ROC per class
best_cls = max(cls_results, key=lambda x: cls_results[x]['accuracy'])
r_best   = cls_results[best_cls]
y_true_arr = np.array(r_best['y_true'])
y_prob_arr = np.array(r_best['y_prob'])
for c_idx, c_name in enumerate(CLASS_NAMES):
    fpr,tpr,_ = roc_curve((y_true_arr==c_idx).astype(int), y_prob_arr[:,c_idx])
    axes[1].plot(fpr,tpr,color=PALETTE[c_idx],lw=2,label=c_name)
axes[1].plot([0,1],[0,1],'--',color='#555',lw=1.5)
axes[1].set_xlabel('FPR'); axes[1].set_ylabel('TPR')
axes[1].set_title(f'ROC per Class — {best_cls}', color='#f0f0f0', fontweight='bold')
axes[1].legend(fontsize=9,framealpha=0.3); axes[1].grid(alpha=0.3)

fig.patch.set_facecolor(BG); plt.tight_layout(); save('cls_roc_curves.png')

# ── CLS 4: Radar Chart ────────────────────────
style()
fig, ax = plt.subplots(figsize=(7,7), subplot_kw=dict(polar=True))
angles = np.linspace(0,2*np.pi,len(metric_names),endpoint=False).tolist()+[0]
for i,(name,r) in enumerate(cls_results.items()):
    vals = [r['accuracy'],r['f1'],r['precision'],r['recall'],r['roc_auc']]+[r['accuracy']]
    ax.plot(angles,vals,color=PALETTE[i%len(PALETTE)],lw=2.5,label=name)
    ax.fill(angles,vals,color=PALETTE[i%len(PALETTE)],alpha=0.1)
ax.set_thetagrids(np.degrees(angles[:-1]),metric_names,color='#ccc',fontsize=10)
ax.set_ylim(0,1)
ax.set_title('Radar Chart — Classification',color='#f0f0f0',fontsize=13,fontweight='bold',pad=20)
ax.legend(loc='upper right',bbox_to_anchor=(1.4,1.15),framealpha=0.3,fontsize=8)
ax.set_facecolor(BG); fig.patch.set_facecolor(BG)
plt.tight_layout(); save('cls_radar_chart.png')

# ── CLS 5: Cross Validation ───────────────────
style()
NAMES_CLS = list(cls_results.keys())
fig, ax = plt.subplots(figsize=(10,5))
cv_m = [cls_results[n]['cv_mean'] for n in NAMES_CLS]
cv_s = [cls_results[n]['cv_std']  for n in NAMES_CLS]
bars = ax.bar(NAMES_CLS,cv_m,yerr=cv_s,color=PALETTE[:len(NAMES_CLS)],
              edgecolor='#0d0d0d',capsize=6,width=0.5,
              error_kw={'ecolor':'#aaa','linewidth':2})
for bar,val in zip(bars,cv_m):
    ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.01,
            f'{val:.4f}',ha='center',fontsize=10,color='#ccc',fontweight='bold')
ax.set_ylim(min(cv_m)*0.9,1.05)
ax.set_title('5-Fold Cross Validation',color='#f0f0f0',fontsize=13,fontweight='bold')
ax.set_ylabel('CV Accuracy'); ax.grid(axis='y',alpha=0.3)
plt.setp(ax.get_xticklabels(),rotation=15,ha='right')
fig.patch.set_facecolor(BG); plt.tight_layout(); save('cls_cross_validation.png')

# ── CLS 6: Feature Importance ─────────────────
for name,r in cls_results.items():
    m = r['model']
    if hasattr(m,'feature_importances_'):
        style()
        fi = pd.DataFrame({'Feature':feature_cols,'Importance':m.feature_importances_})\
               .sort_values('Importance',ascending=True).tail(15)
        fig, ax = plt.subplots(figsize=(9,6))
        clrs = plt.cm.RdYlGn(np.linspace(0.2,0.9,len(fi)))
        ax.barh(fi['Feature'],fi['Importance'],color=clrs)
        ax.set_title(f'Feature Importance — {name}',color='#f0f0f0',fontsize=12,fontweight='bold')
        ax.set_xlabel('Importance')
        fig.patch.set_facecolor(BG); plt.tight_layout()
        save(f'cls_feature_importance_{name.replace(" ","_")}.png')

# ── REG 1: Regression Metrics Bar ─────────────
style()
NAMES_REG = list(reg_results.keys())
fig, axes = plt.subplots(1,3,figsize=(16,5))
fig.suptitle('Regression Metrics Comparison',fontsize=14,fontweight='bold',color='#f0f0f0')

for ax, metric, title, fmt in zip(axes,
    ['r2','mae','rmse'],
    ['R² Score','MAE (AUD)','RMSE (AUD)'],
    ['{:.4f}','{:,.0f}','{:,.0f}']
):
    vals = [reg_results[n][metric] for n in NAMES_REG]
    bars = ax.bar(NAMES_REG,vals,color=PALETTE[:len(NAMES_REG)],edgecolor='#0d0d0d',width=0.5)
    for bar,val in zip(bars,vals):
        ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()*1.01,
                fmt.format(val),ha='center',fontsize=8,color='#ccc',fontweight='bold')
    ax.set_title(title,color='#f0f0f0',fontsize=12,fontweight='bold')
    ax.set_ylabel(title); ax.grid(axis='y',alpha=0.3)
    plt.setp(ax.get_xticklabels(),rotation=20,ha='right',fontsize=8)

fig.patch.set_facecolor(BG); plt.tight_layout(); save('reg_metrics_comparison.png')

# ── REG 2: Actual vs Predicted ────────────────
style()
n_reg = len(reg_results)
fig, axes = plt.subplots(1,n_reg,figsize=(5*n_reg,5))
if n_reg==1: axes=[axes]
fig.suptitle('Actual vs Predicted Price',fontsize=13,fontweight='bold',color='#f0f0f0')
for ax,(name,r) in zip(axes,reg_results.items()):
    actual = np.array(r['y_true'])/1e6
    pred   = np.array(r['y_pred'])/1e6
    ax.scatter(actual,pred,alpha=0.2,s=8,color=PALETTE[0],edgecolors='none')
    mn,mx = min(actual.min(),pred.min()), max(actual.max(),pred.max())
    ax.plot([mn,mx],[mn,mx],'--',color=PALETTE[3],lw=2,label='Perfect')
    ax.set_xlabel('Actual (M AUD)'); ax.set_ylabel('Predicted (M AUD)')
    ax.set_title(f'{name}\nR²={r["r2"]:.3f}',color='#f0f0f0',fontsize=10,fontweight='bold')
    ax.legend(fontsize=8,framealpha=0.3)
fig.patch.set_facecolor(BG); plt.tight_layout(); save('reg_actual_vs_predicted.png')

# ── REG 3: Residuals ──────────────────────────
style()
fig, axes = plt.subplots(1,n_reg,figsize=(5*n_reg,5))
if n_reg==1: axes=[axes]
fig.suptitle('Residual Plots',fontsize=13,fontweight='bold',color='#f0f0f0')
for ax,(name,r) in zip(axes,reg_results.items()):
    pred = np.array(r['y_pred'])/1e6
    res  = (np.array(r['y_true']) - np.array(r['y_pred']))/1e6
    ax.scatter(pred,res,alpha=0.2,s=8,color=PALETTE[2],edgecolors='none')
    ax.axhline(0,color=PALETTE[0],lw=2,ls='--')
    ax.set_xlabel('Predicted (M AUD)'); ax.set_ylabel('Residual (M AUD)')
    ax.set_title(name,color='#f0f0f0',fontsize=10,fontweight='bold')
fig.patch.set_facecolor(BG); plt.tight_layout(); save('reg_residuals.png')

# ── REG 4: R² Comparison ──────────────────────
style()
fig, ax = plt.subplots(figsize=(10,5))
r2_vals = [reg_results[n]['r2'] for n in NAMES_REG]
bars = ax.bar(NAMES_REG,r2_vals,color=PALETTE[:len(NAMES_REG)],edgecolor='#0d0d0d',width=0.5)
for bar,val in zip(bars,r2_vals):
    ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.005,
            f'{val:.4f}',ha='center',fontsize=11,color='#ccc',fontweight='bold')
ax.set_ylim(0,1.1)
ax.set_title('R² Score Comparison — Regression',color='#f0f0f0',fontsize=13,fontweight='bold')
ax.set_ylabel('R²'); ax.grid(axis='y',alpha=0.3)
ax.axhline(1.0,color=PALETTE[3],lw=1.5,ls='--',alpha=0.5,label='Perfect R²=1')
ax.legend(fontsize=9,framealpha=0.3)
plt.setp(ax.get_xticklabels(),rotation=15,ha='right')
fig.patch.set_facecolor(BG); plt.tight_layout(); save('reg_r2_comparison.png')

# ── REG 5: Feature Importance (RF) ────────────
rf_reg = reg_results.get('RF Regressor')
if rf_reg and hasattr(rf_reg['model'],'feature_importances_'):
    style()
    fi = pd.DataFrame({'Feature':feature_cols,
                       'Importance':rf_reg['model'].feature_importances_})\
           .sort_values('Importance',ascending=True)
    fig, ax = plt.subplots(figsize=(9,6))
    clrs = plt.cm.RdYlGn(np.linspace(0.2,0.9,len(fi)))
    ax.barh(fi['Feature'],fi['Importance'],color=clrs)
    ax.set_title('Feature Importance — RF Regressor',color='#f0f0f0',fontsize=12,fontweight='bold')
    ax.set_xlabel('Importance')
    fig.patch.set_facecolor(BG); plt.tight_layout(); save('reg_feature_importance.png')

# ─────────────────────────────────────────────
# STEP 9 · DONE
# ─────────────────────────────────────────────
best_cls_name = max(cls_results, key=lambda n: cls_results[n]['accuracy'])
best_reg_name = max(reg_results, key=lambda n: reg_results[n]['r2'])

print("\n"+"="*65)
print("  ✅ TRAINING COMPLETE!")
print(f"  🏆 Best Classifier : {best_cls_name}  ({cls_results[best_cls_name]['accuracy']:.4f} acc)")
print(f"  🏆 Best Regressor  : {best_reg_name}  (R²={reg_results[best_reg_name]['r2']:.4f})")
print("\n  💾 PKL Models:")
for f in os.listdir(OUTPUT_DIR):
    if f.endswith('.pkl'): print(f"     • outputs/{f}")
print("\n  📊 Graphs saved in outputs/:")
for f in sorted(os.listdir(OUTPUT_DIR)):
    if f.endswith('.png'): print(f"     • {f}")
print("\n  🚀 Run: streamlit run app.py")
print("="*65)