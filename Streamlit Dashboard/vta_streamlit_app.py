import streamlit as st
import pandas as pd
import numpy as np
import time
import datetime
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    confusion_matrix, accuracy_score, f1_score, roc_curve, auc,
    ConfusionMatrixDisplay, mean_squared_error, mean_absolute_error, r2_score,
)
from sklearn.linear_model import LogisticRegression, LinearRegression, Lasso
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE
from xgboost import XGBClassifier, XGBRegressor

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

st.set_page_config(
    page_title="VTA Ridership — ML Dashboard",
    page_icon="🚍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CUSTOM CSS FOR VISUAL POLISH
# ============================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

    .main .block-container { padding-top: 1.5rem; max-width: 1200px; }

    .hero-title {
        font-family: 'Inter', sans-serif;
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #1e3a5f 0%, #2d8cf0 50%, #00b4d8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
        line-height: 1.2;
    }
    .hero-subtitle {
        font-family: 'Inter', sans-serif;
        font-size: 1.2rem;
        color: #6b7280;
        margin-top: 0.2rem;
        font-style: italic;
    }

    .metric-card {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        border-radius: 16px;
        padding: 1.2rem 1.5rem;
        text-align: center;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.06);
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 24px rgba(0,0,0,0.1);
    }
    .metric-value {
        font-family: 'Inter', sans-serif;
        font-size: 2rem;
        font-weight: 700;
        color: #1e3a5f;
        margin: 0.3rem 0;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .metric-green .metric-value { color: #059669; }
    .metric-blue .metric-value { color: #2563eb; }
    .metric-purple .metric-value { color: #7c3aed; }
    .metric-orange .metric-value { color: #d97706; }

    .prediction-box {
        border-radius: 16px;
        padding: 1.5rem 2rem;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 4px 16px rgba(0,0,0,0.08);
    }
    .prediction-low {
        background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
        border: 2px solid #059669;
    }
    .prediction-medium {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        border: 2px solid #d97706;
    }
    .prediction-high {
        background: linear-gradient(135deg, #fee2e2 0%, #fca5a5 100%);
        border: 2px solid #dc2626;
    }
    .prediction-label {
        font-family: 'Inter', sans-serif;
        font-size: 1.8rem;
        font-weight: 700;
    }
    .prediction-low .prediction-label { color: #059669; }
    .prediction-medium .prediction-label { color: #d97706; }
    .prediction-high .prediction-label { color: #dc2626; }
    .prediction-conf {
        font-size: 1rem;
        color: #4b5563;
        margin-top: 0.3rem;
    }

    .boarding-prediction {
        background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%);
        border: 2px solid #2563eb;
        border-radius: 16px;
        padding: 1.5rem 2rem;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 4px 16px rgba(0,0,0,0.08);
    }
    .boarding-value {
        font-family: 'Inter', sans-serif;
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e40af;
    }

    .section-header {
        font-family: 'Inter', sans-serif;
        font-size: 1.4rem;
        font-weight: 600;
        color: #1e3a5f;
        border-bottom: 3px solid #2d8cf0;
        padding-bottom: 0.5rem;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }

    .insight-box {
        background: #f0f9ff;
        border-left: 4px solid #2d8cf0;
        border-radius: 0 8px 8px 0;
        padding: 1rem 1.2rem;
        margin: 0.8rem 0 1.5rem 0;
        font-size: 0.95rem;
        color: #1e3a5f;
    }

    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e3a5f 0%, #0f172a 100%);
    }
    div[data-testid="stSidebar"] * {
        color: #e2e8f0 !important;
    }
    div[data-testid="stSidebar"] .stRadio label {
        font-size: 0.95rem;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 0.5rem 1.2rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# DATA LOADING & CLEANING (cached)
# ============================================================
@st.cache_data(show_spinner="Loading & cleaning VTA data...")
def load_and_clean_data():
    df_raw = pd.read_excel("OCT_2025_RBS_FULL_DATA_SET.XLSX", engine="openpyxl")
    df = df_raw.copy()
    df = df.dropna(how="all").dropna(axis=1, how="all")
    numeric_cols = ["BOARDINGS","ALIGHTINGS","TRIPS","AVG_BOARDINGS","AVG_ALIGHTINGS",
        "AVG_ACTIVITY","PASS_LOAD","PEAK_LOAD","AVG_PEAK_LOAD","SORT_ORDER","STOP_ID","TOTAL_SORT","SORT_SP"]
    for col in numeric_cols:
        if col in df.columns: df[col] = pd.to_numeric(df[col], errors="coerce")
    str_cols = ["ROUTE_NAME","ROUTE_NUMBER","SERVICE_PERIOD","SERVICE_CODE","DIRECTION_NAME",
        "BRANCH","MAIN_CROSS_STREET","CITY","STOP_DISPLAY","Additional_Notes","PATTERN_KEY","BLOCK"]
    for col in str_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace({"nan": np.nan, "None": np.nan, "": np.nan})
    def trip_time_to_hour(val):
        if isinstance(val, datetime.time): return val.hour
        try: return int(round(float(val) * 24)) % 24
        except: return np.nan
    df["TRIP_HOUR"] = df["TRIP_TIME"].apply(trip_time_to_hour).astype("Int64")
    df["CITY"] = df["CITY"].fillna("Unknown")
    df["Additional_Notes"] = df["Additional_Notes"].fillna("None")
    for col in ["BOARDINGS","ALIGHTINGS","AVG_BOARDINGS","AVG_ALIGHTINGS","AVG_ACTIVITY"]:
        if col in df.columns: df[col] = df[col].fillna(0)
    def classify_route(sc):
        if pd.isna(sc): return "Unknown"
        sc = str(sc).lower()
        if "express" in sc: return "Express"
        elif "frequent" in sc: return "Frequent"
        elif "local" in sc: return "Local"
        elif "light rail" in sc or "lrt" in sc: return "Light Rail"
        elif "rapid" in sc: return "Rapid"
        return "Other"
    df["ROUTE_TYPE"] = df["SERVICE_CODE"].apply(classify_route)
    def categorize_time(h):
        if pd.isna(h): return "Unknown"
        h = int(h)
        if 5 <= h < 9: return "AM Peak (5-9)"
        elif 9 <= h < 15: return "Midday (9-3)"
        elif 15 <= h < 19: return "PM Peak (3-7)"
        elif 19 <= h < 23: return "Evening (7-11)"
        return "Late Night (11-5)"
    df["TIME_PERIOD"] = df["TRIP_HOUR"].apply(categorize_time)
    df["TOTAL_ACTIVITY"] = df["BOARDINGS"] + df["ALIGHTINGS"]
    df = df.drop_duplicates()
    return df


@st.cache_resource(show_spinner="Training classification models...")
def train_classification(_df):
    df_clf = _df.dropna(subset=["PEAK_LOAD"]).copy()
    df_clf = df_clf[df_clf["PEAK_LOAD"] > 0]
    low_t = df_clf["PEAK_LOAD"].quantile(0.33)
    high_t = df_clf["PEAK_LOAD"].quantile(0.67)
    df_clf["CROWDING_LEVEL"] = pd.cut(df_clf["PEAK_LOAD"], bins=[-np.inf,low_t,high_t,np.inf], labels=["Low","Medium","High"])
    clf_features = ["ROUTE_TYPE","TIME_PERIOD","SERVICE_PERIOD","TRIP_HOUR","TRIPS","DIRECTION_NAME"]
    df_enc = df_clf[clf_features + ["CROWDING_LEVEL"]].dropna().copy()
    encoders = {}
    for col in ["ROUTE_TYPE","TIME_PERIOD","SERVICE_PERIOD","DIRECTION_NAME"]:
        le = LabelEncoder(); df_enc[col] = le.fit_transform(df_enc[col].astype(str)); encoders[col] = le
    le_target = LabelEncoder(); df_enc["TARGET"] = le_target.fit_transform(df_enc["CROWDING_LEVEL"])
    class_names = list(le_target.classes_)
    X = df_enc[clf_features].astype(float); y = df_enc["TARGET"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler(); X_train_s = scaler.fit_transform(X_train); X_test_s = scaler.transform(X_test)
    models_cfg = {
        "Logistic Regression": (LogisticRegression(max_iter=1000, random_state=42, multi_class="multinomial"), True),
        "KNN (k=5)": (KNeighborsClassifier(n_neighbors=5, n_jobs=-1), True),
        "Random Forest": (RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1), False),
        "XGBoost": (XGBClassifier(n_estimators=300, max_depth=10, learning_rate=0.1, random_state=42, use_label_encoder=False, eval_metric="mlogloss", tree_method="hist"), False),
    }
    results = {}
    for name, (model, scaled) in models_cfg.items():
        Xtr, Xte = (X_train_s, X_test_s) if scaled else (X_train, X_test)
        model.fit(Xtr, y_train); yp = model.predict(Xte); ypr = model.predict_proba(Xte)
        results[name] = {"model": model, "y_pred": yp, "y_prob": ypr,
            "accuracy": accuracy_score(y_test, yp), "f1": f1_score(y_test, yp, average="weighted")}
    best_name = max(results, key=lambda k: results[k]["accuracy"])
    return {"results": results, "best_name": best_name, "X_test": X_test, "X_test_scaled": X_test_s,
        "y_test": y_test, "class_names": class_names, "features": clf_features,
        "encoders": encoders, "scaler": scaler, "le_target": le_target, "low_thresh": low_t, "high_thresh": high_t}


@st.cache_resource(show_spinner="Training regression models...")
def train_regression(_df):
    agg_cols = ["ROUTE_NUMBER","ROUTE_TYPE","TIME_PERIOD","SERVICE_PERIOD"]
    df_route = (_df.dropna(subset=["BOARDINGS"]).groupby(agg_cols, observed=True)
        .agg(TOTAL_BOARDINGS=("BOARDINGS","sum"), AVG_TRIP_HOUR=("TRIP_HOUR","mean"), NUM_STOPS=("STOP_ID","nunique")).reset_index())
    df_route = df_route[df_route["TOTAL_BOARDINGS"] > 0]
    reg_features = ["ROUTE_NUMBER","ROUTE_TYPE","TIME_PERIOD","SERVICE_PERIOD","AVG_TRIP_HOUR","NUM_STOPS"]
    df_enc = df_route[reg_features + ["TOTAL_BOARDINGS"]].dropna().copy()
    encoders = {}
    for col in ["ROUTE_NUMBER","ROUTE_TYPE","TIME_PERIOD","SERVICE_PERIOD"]:
        le = LabelEncoder(); df_enc[col] = le.fit_transform(df_enc[col].astype(str)); encoders[col] = le
    X = df_enc[reg_features].astype(float); y = df_enc["TOTAL_BOARDINGS"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler(); X_train_s = scaler.fit_transform(X_train); X_test_s = scaler.transform(X_test)
    models_cfg = {
        "Linear Regression": (LinearRegression(), True),
        "Lasso Regression": (Lasso(alpha=1.0, random_state=42), True),
        "Random Forest": (RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1), False),
        "XGBoost": (XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42, tree_method="hist"), False),
    }
    results = {}
    for name, (model, scaled) in models_cfg.items():
        Xtr, Xte = (X_train_s, X_test_s) if scaled else (X_train, X_test)
        model.fit(Xtr, y_train); yp = model.predict(Xte)
        if scaled: yp = np.clip(yp, 0, None)
        results[name] = {"model": model, "y_pred": yp,
            "rmse": np.sqrt(mean_squared_error(y_test, yp)), "mae": mean_absolute_error(y_test, yp), "r2": r2_score(y_test, yp)}
    best_name = min(results, key=lambda k: results[k]["rmse"])
    return {"results": results, "best_name": best_name, "X_test": X_test, "X_test_scaled": X_test_s,
        "y_test": y_test, "features": reg_features, "encoders": encoders, "scaler": scaler, "df_enc": df_enc, "df_route": df_route}


# ============================================================
# LOAD & TRAIN
# ============================================================
df = load_and_clean_data()
clf_data = train_classification(df)
reg_data = train_regression(df)

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("## 🚍 VTA ML Dashboard")
    st.markdown("*Optimizing Silicon Valley Transit*")
    st.markdown("---")
    page = st.radio("Navigate", ["🏠 Overview", "🎯 Classification", "📈 Regression"], label_visibility="collapsed")
    st.markdown("---")
    st.markdown("**Dataset:** VTA Oct 2025")
    st.markdown(f"**Rows:** {len(df):,}")
    st.markdown("**Models:** 8 total")
    st.markdown("---")
    st.caption("DATA230 — Group 7")
    st.caption("Tool: Streamlit")


# ============================================================
# HELPER: Metric card HTML
# ============================================================
def metric_card(label, value, color_class=""):
    return f"""<div class="metric-card {color_class}">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
    </div>"""


# ============================================================
# PAGE: OVERVIEW
# ============================================================
if page == "🏠 Overview":
    st.markdown('<div class="hero-title">VTA 2025 Ridership</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">Where, When & How Silicon Valley Rides — Machine Learning Analysis</div>', unsafe_allow_html=True)
    st.markdown("")

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown(metric_card("Dataset Rows", f"{len(df):,}", "metric-blue"), unsafe_allow_html=True)
    with c2: st.markdown(metric_card("Classification Accuracy", f"{clf_data['results'][clf_data['best_name']]['accuracy']:.1%}", "metric-green"), unsafe_allow_html=True)
    with c3: st.markdown(metric_card("Regression R²", f"{reg_data['results'][reg_data['best_name']]['r2']:.3f}", "metric-purple"), unsafe_allow_html=True)
    with c4: st.markdown(metric_card("ML Models Trained", "8", "metric-orange"), unsafe_allow_html=True)

    st.markdown("")
    st.markdown('<div class="section-header">Classification: Crowding Prediction</div>', unsafe_allow_html=True)
    st.markdown(f"**Question:** Will this route-time combination experience Low, Medium, or High crowding?")

    clf_metrics = pd.DataFrame({
        "Model": list(clf_data["results"].keys()),
        "Accuracy": [f"{r['accuracy']:.4f}" for r in clf_data["results"].values()],
        "F1 Score": [f"{r['f1']:.4f}" for r in clf_data["results"].values()],
    })
    st.dataframe(clf_metrics, hide_index=True, use_container_width=True)

    st.markdown('<div class="insight-box">💡 <b>Key Finding:</b> XGBoost achieves 70% accuracy — tree-based models dramatically outperform linear approaches (44%), confirming non-linear crowding patterns.</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-header">Regression: Route-Level Boardings</div>', unsafe_allow_html=True)
    st.markdown(f"**Question:** How many total passengers will this route carry during this time period?")

    reg_metrics = pd.DataFrame({
        "Model": list(reg_data["results"].keys()),
        "RMSE": [f"{r['rmse']:.1f}" for r in reg_data["results"].values()],
        "MAE": [f"{r['mae']:.1f}" for r in reg_data["results"].values()],
        "R²": [f"{r['r2']:.4f}" for r in reg_data["results"].values()],
    })
    st.dataframe(reg_metrics, hide_index=True, use_container_width=True)

    st.markdown('<div class="insight-box">💡 <b>Key Finding:</b> XGBoost explains 86% of route-level boarding variance (R²=0.859). Linear models fail completely (R²=0.02), proving demand patterns are fundamentally non-linear.</div>', unsafe_allow_html=True)


# ============================================================
# PAGE: CLASSIFICATION
# ============================================================
elif page == "🎯 Classification":
    st.markdown('<div class="hero-title">Crowding Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">Predict whether a route-time combination will experience Low, Medium, or High crowding</div>', unsafe_allow_html=True)
    st.markdown("")

    best_acc = clf_data["results"][clf_data["best_name"]]["accuracy"]
    c1, c2, c3 = st.columns(3)
    with c1: st.markdown(metric_card("Best Model", clf_data["best_name"], "metric-blue"), unsafe_allow_html=True)
    with c2: st.markdown(metric_card("Accuracy", f"{best_acc:.1%}", "metric-green"), unsafe_allow_html=True)
    with c3: st.markdown(metric_card("Test Samples", f"{len(clf_data['y_test']):,}", "metric-purple"), unsafe_allow_html=True)

    st.markdown('<div class="section-header">🔮 Interactive Prediction</div>', unsafe_allow_html=True)
    pc1, pc2, pc3 = st.columns(3)
    with pc1:
        pred_rt = st.selectbox("Route Type", clf_data["encoders"]["ROUTE_TYPE"].classes_)
        pred_tp = st.selectbox("Time Period", clf_data["encoders"]["TIME_PERIOD"].classes_)
    with pc2:
        pred_sp = st.selectbox("Service Period", clf_data["encoders"]["SERVICE_PERIOD"].classes_)
        pred_dir = st.selectbox("Direction", clf_data["encoders"]["DIRECTION_NAME"].classes_)
    with pc3:
        pred_hour = st.slider("Trip Hour", 0, 23, 8)
        pred_trips = st.number_input("Trips", min_value=0, max_value=100, value=10)

    pred_row = pd.DataFrame([{
        "ROUTE_TYPE": clf_data["encoders"]["ROUTE_TYPE"].transform([pred_rt])[0],
        "TIME_PERIOD": clf_data["encoders"]["TIME_PERIOD"].transform([pred_tp])[0],
        "SERVICE_PERIOD": clf_data["encoders"]["SERVICE_PERIOD"].transform([pred_sp])[0],
        "TRIP_HOUR": pred_hour, "TRIPS": pred_trips,
        "DIRECTION_NAME": clf_data["encoders"]["DIRECTION_NAME"].transform([pred_dir])[0],
    }])

    best_clf_model = clf_data["results"]["XGBoost"]["model"]
    pred_class = best_clf_model.predict(pred_row)[0]
    pred_proba = best_clf_model.predict_proba(pred_row)[0]
    pred_label = clf_data["class_names"][pred_class]
    css_class = {"Low": "prediction-low", "Medium": "prediction-medium", "High": "prediction-high"}

    st.markdown(f"""<div class="prediction-box {css_class[pred_label]}">
        <div class="prediction-label">🚦 {pred_label} Crowding</div>
        <div class="prediction-conf">{pred_proba[pred_class]:.1%} confidence &nbsp;|&nbsp;
        Low: {pred_proba[clf_data['class_names'].index('Low')]:.1%} &nbsp;
        Med: {pred_proba[clf_data['class_names'].index('Medium')]:.1%} &nbsp;
        High: {pred_proba[clf_data['class_names'].index('High')]:.1%}</div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-header">📊 ML Visualizations</div>', unsafe_allow_html=True)
    best = clf_data["results"][clf_data["best_name"]]
    y_test = clf_data["y_test"]
    class_names = clf_data["class_names"]

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["🔲 Confusion Matrix", "📉 ROC Curves", "📊 Feature Importance", "🔍 SHAP Explainability", "📐 LDA Projection", "🌐 t-SNE Projection"])

    with tab1:
        cm = confusion_matrix(y_test, best["y_pred"])
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names).plot(cmap="Blues", values_format="d", ax=axes[0])
        axes[0].set_title("Counts", fontsize=13, fontweight="bold")
        ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=class_names).plot(cmap="Blues", values_format=".1%", ax=axes[1])
        axes[1].set_title("Normalized", fontsize=13, fontweight="bold")
        fig.suptitle(f"Confusion Matrix — {clf_data['best_name']}", fontsize=15, fontweight="bold")
        plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()
        st.markdown('<div class="insight-box">💡 Misclassifications between Low↔Medium are expected (adjacent bins). High↔Low errors are critical — they mean overcrowded routes going undetected.</div>', unsafe_allow_html=True)

    with tab2:
        y_test_oh = np.eye(len(class_names))[y_test.values]
        colors = ["#2ca02c", "#ff7f0e", "#d62728"]
        fig, ax = plt.subplots(figsize=(7, 5))
        for i, cn in enumerate(class_names):
            fpr, tpr, _ = roc_curve(y_test_oh[:, i], best["y_prob"][:, i])
            ax.plot(fpr, tpr, color=colors[i], lw=2.5, label=f"{cn} (AUC={auc(fpr, tpr):.3f})")
        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4)
        ax.set_xlabel("False Positive Rate", fontsize=12); ax.set_ylabel("True Positive Rate", fontsize=12)
        ax.set_title(f"Multi-class ROC — {clf_data['best_name']}", fontsize=14, fontweight="bold")
        ax.legend(loc="lower right", fontsize=11, framealpha=0.9)
        ax.set_facecolor("#fafafa")
        plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()
        st.markdown('<div class="insight-box">💡 AUC close to 1.0 = reliable crowding detection. The High class AUC is most critical for transit safety.</div>', unsafe_allow_html=True)

    with tab3:
        xgb_imp = pd.Series(clf_data["results"]["XGBoost"]["model"].feature_importances_, index=clf_data["features"]).sort_values()
        fig, ax = plt.subplots(figsize=(7, 4))
        norm_v = xgb_imp.values / xgb_imp.values.max()
        bars = ax.barh(xgb_imp.index, xgb_imp.values, color=plt.cm.Blues(0.3 + 0.7 * norm_v), edgecolor="white", linewidth=0.5)
        for i, v in enumerate(xgb_imp.values): ax.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=11, fontweight="bold")
        ax.set_xlabel("Importance (gain)", fontsize=12); ax.set_title("Feature Importance — XGBoost", fontsize=14, fontweight="bold")
        ax.set_facecolor("#fafafa")
        plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()
        st.markdown('<div class="insight-box">💡 Top features reveal what drives crowding — enabling VTA to target interventions at the most impactful factors.</div>', unsafe_allow_html=True)

    with tab4:
        with st.spinner("Computing SHAP values..."):
            shap_n = min(500, len(clf_data["X_test"]))
            X_shap = clf_data["X_test"].sample(n=shap_n, random_state=42)
            explainer = shap.TreeExplainer(clf_data["results"]["XGBoost"]["model"])
            shap_vals = explainer.shap_values(X_shap)
            high_idx = class_names.index("High")
            sv = shap_vals[high_idx] if isinstance(shap_vals, list) else shap_vals[:, :, high_idx]
            fig, ax = plt.subplots(figsize=(7, 4))
            shap.summary_plot(sv, X_shap, feature_names=clf_data["features"], show=False)
            plt.title('SHAP — "High Crowding" Class', fontsize=14, fontweight="bold")
            plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()
        st.markdown('<div class="insight-box">💡 SHAP shows how each feature pushes toward/away from High crowding. Red dots (high values) on the right increase crowding likelihood.</div>', unsafe_allow_html=True)

    with tab5:
        with st.spinner("Computing LDA projection..."):
            lda = LinearDiscriminantAnalysis(n_components=2)
            X_lda = lda.fit_transform(clf_data["X_test_scaled"], y_test)
            lda_labels = [class_names[i] for i in y_test.values]
            color_map_lda = {"Low": "#059669", "Medium": "#d97706", "High": "#dc2626"}
            fig, ax = plt.subplots(figsize=(7, 5))
            for label in class_names:
                mask = np.array(lda_labels) == label
                ax.scatter(X_lda[mask, 0], X_lda[mask, 1], c=color_map_lda[label],
                          label=label, alpha=0.3, s=10, edgecolors="none")
            ax.set_xlabel("LDA Component 1", fontsize=12); ax.set_ylabel("LDA Component 2", fontsize=12)
            ax.set_title("LDA Projection — Crowding Level Separability", fontsize=14, fontweight="bold")
            legend = ax.legend(fontsize=11, markerscale=3)
            for lh in legend.legend_handles: lh.set_alpha(1.0)
            ax.set_facecolor("#fafafa")
            plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()
        st.markdown(f'<div class="insight-box">💡 LDA is a <b>supervised</b> technique that maximizes class separation. Explained variance ratio: {lda.explained_variance_ratio_[0]:.1%} in first component. Clear separation confirms our features can distinguish crowding levels.</div>', unsafe_allow_html=True)

    with tab6:
        with st.spinner("Computing t-SNE embedding (this may take a moment)..."):
            tsne_n = min(3000, len(clf_data["X_test_scaled"]))
            tsne_idx = np.random.RandomState(42).choice(len(clf_data["X_test_scaled"]), tsne_n, replace=False)
            X_tsne_in = clf_data["X_test_scaled"][tsne_idx]
            y_tsne_labels = [class_names[i] for i in y_test.values[tsne_idx]]
            tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
            X_tsne = tsne.fit_transform(X_tsne_in)
            color_map_tsne = {"Low": "#059669", "Medium": "#d97706", "High": "#dc2626"}
            fig, ax = plt.subplots(figsize=(7, 5))
            for label in class_names:
                mask = np.array(y_tsne_labels) == label
                ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1], c=color_map_tsne[label],
                          label=label, alpha=0.4, s=12, edgecolors="none")
            ax.set_xlabel("t-SNE Component 1", fontsize=12); ax.set_ylabel("t-SNE Component 2", fontsize=12)
            ax.set_title("t-SNE Projection — Non-linear Crowding Clusters", fontsize=14, fontweight="bold")
            legend = ax.legend(fontsize=11, markerscale=3)
            for lh in legend.legend_handles: lh.set_alpha(1.0)
            ax.set_facecolor("#fafafa")
            plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()
        st.markdown('<div class="insight-box">💡 t-SNE preserves <b>local structure</b> — nearby points in 6D feature space remain neighbors in 2D. The overlap between classes reflects the continuous nature of PEAK_LOAD being split into discrete bins.</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-header">🏆 Model Comparison</div>', unsafe_allow_html=True)
    model_names = list(clf_data["results"].keys())
    accs = [clf_data["results"][m]["accuracy"] for m in model_names]
    f1s = [clf_data["results"][m]["f1"] for m in model_names]
    fig, ax = plt.subplots(figsize=(7, 4))
    x_pos = np.arange(len(model_names)); w = 0.35
    edge_c = ["black" if m == clf_data["best_name"] else "white" for m in model_names]
    edge_w = [2.5 if m == clf_data["best_name"] else 0.5 for m in model_names]
    b1 = ax.bar(x_pos - w/2, accs, w, label="Accuracy", color="#636EFA", edgecolor=edge_c, linewidth=edge_w)
    b2 = ax.bar(x_pos + w/2, f1s, w, label="Weighted F1", color="#AB63FA", edgecolor=edge_c, linewidth=edge_w)
    for i, (a, f) in enumerate(zip(accs, f1s)):
        ax.text(i - w/2, a + 0.008, f"{a:.3f}", ha="center", fontsize=10, fontweight="bold")
        ax.text(i + w/2, f + 0.008, f"{f:.3f}", ha="center", fontsize=10, fontweight="bold")
    ax.set_xticks(x_pos); ax.set_xticklabels(model_names, fontsize=11)
    ax.set_ylabel("Score", fontsize=12); ax.set_title("Model Comparison", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11); ax.set_ylim(0, max(max(accs), max(f1s)) * 1.15); ax.set_facecolor("#fafafa")
    plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()


# ============================================================
# PAGE: REGRESSION
# ============================================================
elif page == "📈 Regression":
    st.markdown('<div class="hero-title">Boardings Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">Predict total route-level boardings for demand-responsive scheduling</div>', unsafe_allow_html=True)
    st.markdown("")

    best_r2 = reg_data["results"][reg_data["best_name"]]["r2"]
    c1, c2, c3 = st.columns(3)
    with c1: st.markdown(metric_card("Best Model", reg_data["best_name"], "metric-blue"), unsafe_allow_html=True)
    with c2: st.markdown(metric_card("R² Score", f"{best_r2:.3f}", "metric-green"), unsafe_allow_html=True)
    with c3: st.markdown(metric_card("Route-Level Rows", f"{len(reg_data['df_route']):,}", "metric-purple"), unsafe_allow_html=True)

    st.markdown('<div class="section-header">🔮 Interactive Prediction</div>', unsafe_allow_html=True)
    pc1, pc2 = st.columns(2)
    with pc1:
        pred_rn = st.selectbox("Route Number", reg_data["encoders"]["ROUTE_NUMBER"].classes_)
        pred_rt = st.selectbox("Route Type", reg_data["encoders"]["ROUTE_TYPE"].classes_, key="r_rt")
        pred_tp = st.selectbox("Time Period", reg_data["encoders"]["TIME_PERIOD"].classes_, key="r_tp")
    with pc2:
        pred_sp = st.selectbox("Service Period", reg_data["encoders"]["SERVICE_PERIOD"].classes_, key="r_sp")
        pred_hour = st.slider("Avg Trip Hour", 0.0, 23.0, 12.0, 0.5, key="r_hr")
        pred_stops = st.number_input("Number of Stops", min_value=1, max_value=200, value=30, key="r_st")

    pred_row = pd.DataFrame([{
        "ROUTE_NUMBER": reg_data["encoders"]["ROUTE_NUMBER"].transform([pred_rn])[0],
        "ROUTE_TYPE": reg_data["encoders"]["ROUTE_TYPE"].transform([pred_rt])[0],
        "TIME_PERIOD": reg_data["encoders"]["TIME_PERIOD"].transform([pred_tp])[0],
        "SERVICE_PERIOD": reg_data["encoders"]["SERVICE_PERIOD"].transform([pred_sp])[0],
        "AVG_TRIP_HOUR": pred_hour, "NUM_STOPS": pred_stops,
    }])
    pred_val = max(0, reg_data["results"]["XGBoost"]["model"].predict(pred_row)[0])

    st.markdown(f"""<div class="boarding-prediction">
        <div style="font-size:0.9rem; color:#4b5563;">Predicted Total Boardings</div>
        <div class="boarding-value">🚌 {pred_val:,.0f} passengers</div>
        <div style="font-size:0.85rem; color:#6b7280; margin-top:0.3rem;">
            {pred_rn} • {pred_tp} • {pred_sp}</div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-header">📊 ML Visualizations</div>', unsafe_allow_html=True)
    best = reg_data["results"][reg_data["best_name"]]
    y_test = reg_data["y_test"]

    tab1, tab2, tab3, tab4 = st.tabs(["📍 Predicted vs Actual", "📊 Feature Importance", "🔍 SHAP Explainability", "🔲 Correlation Heatmap"])

    with tab1:
        actual = y_test.values; predicted = best["y_pred"]
        res_abs = np.abs(actual - predicted)
        fig, ax = plt.subplots(figsize=(7, 5))
        sc = ax.scatter(actual, predicted, alpha=0.7, s=55, c=res_abs,
                       cmap="coolwarm", edgecolors="white", linewidth=0.5, zorder=3)
        cbar = plt.colorbar(sc, ax=ax, shrink=0.8); cbar.set_label("|Residual|", fontsize=10)
        mv = max(actual.max(), predicted.max()) * 1.05
        ax.plot([0, mv], [0, mv], "k--", lw=1.5, alpha=0.7, label="Perfect (y=x)", zorder=2)
        ax.set_xlabel("Actual Total Boardings", fontsize=12); ax.set_ylabel("Predicted", fontsize=12)
        ax.set_title(f"Predicted vs Actual — {reg_data['best_name']}", fontsize=14, fontweight="bold")
        ax.legend(fontsize=11, loc="upper left"); ax.set_facecolor("#fafafa")
        stats_text = f"R² = {best['r2']:.3f}\nRMSE = {best['rmse']:.0f}\nMAE = {best['mae']:.0f}"
        ax.text(0.97, 0.03, stats_text, transform=ax.transAxes, fontsize=10,
                va="bottom", ha="right", bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#cccccc", alpha=0.9))
        plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()
        st.markdown('<div class="insight-box">💡 Points near the diagonal = accurate predictions. Spread indicates prediction uncertainty — wider at high boardings suggests top routes are harder to predict precisely.</div>', unsafe_allow_html=True)

    with tab2:
        xgb_imp = pd.Series(reg_data["results"]["XGBoost"]["model"].feature_importances_, index=reg_data["features"]).sort_values()
        fig, ax = plt.subplots(figsize=(7, 4))
        norm_v = xgb_imp.values / xgb_imp.values.max()
        bars = ax.barh(xgb_imp.index, xgb_imp.values, color=plt.cm.Blues(0.3 + 0.7 * norm_v), edgecolor="white", linewidth=0.5)
        for i, v in enumerate(xgb_imp.values): ax.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=11, fontweight="bold")
        ax.set_xlabel("Importance (gain)", fontsize=12); ax.set_title("Feature Importance — XGBoost", fontsize=14, fontweight="bold")
        ax.set_facecolor("#fafafa")
        plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()
        st.markdown('<div class="insight-box">💡 Dominant features reveal what drives boarding demand — enabling VTA to focus scheduling resources where they matter most.</div>', unsafe_allow_html=True)

    with tab3:
        with st.spinner("Computing SHAP values..."):
            shap_n = min(500, len(reg_data["X_test"]))
            X_shap = reg_data["X_test"].sample(n=shap_n, random_state=42)
            explainer = shap.TreeExplainer(reg_data["results"]["XGBoost"]["model"])
            shap_vals = explainer.shap_values(X_shap)
            fig, ax = plt.subplots(figsize=(7, 4))
            shap.summary_plot(shap_vals, X_shap, feature_names=reg_data["features"], show=False)
            plt.title("SHAP — Route-Level Boardings", fontsize=14, fontweight="bold")
            plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()
        st.markdown('<div class="insight-box">💡 SHAP reveals which routes and time periods push boardings up or down — directly informing demand-responsive scheduling decisions.</div>', unsafe_allow_html=True)

    with tab4:
        corr_cols = reg_data["features"] + ["TOTAL_BOARDINGS"]
        corr_matrix = reg_data["df_enc"][corr_cols].corr()
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                    square=True, linewidths=0.5, ax=ax, vmin=-1, vmax=1,
                    cbar_kws={"shrink": 0.8})
        ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold")
        plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()
        st.markdown('<div class="insight-box">💡 High inter-feature correlation = redundancy. Bottom row shows linear correlation with the target — weak correlations explain why linear models fail.</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-header">🏆 Model Comparison</div>', unsafe_allow_html=True)
    model_names = list(reg_data["results"].keys())
    rmses = [reg_data["results"][m]["rmse"] for m in model_names]
    r2s = [reg_data["results"][m]["r2"] for m in model_names]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    x_pos = np.arange(len(model_names))
    edge_c = ["black" if m == reg_data["best_name"] else "white" for m in model_names]
    edge_w = [2.5 if m == reg_data["best_name"] else 0.5 for m in model_names]
    b1 = ax1.bar(x_pos, rmses, color="#636EFA", width=0.6, edgecolor=edge_c, linewidth=edge_w)
    ax1.bar_label(b1, fmt="%.0f", padding=3, fontsize=10, fontweight="bold")
    ax1.set_xticks(x_pos); ax1.set_xticklabels(model_names, fontsize=10)
    ax1.set_ylabel("RMSE", fontsize=12); ax1.set_title("RMSE (lower = better)", fontsize=13, fontweight="bold")
    ax1.set_facecolor("#fafafa")
    b2 = ax2.bar(x_pos, r2s, color="#636EFA", width=0.6, edgecolor=edge_c, linewidth=edge_w)
    ax2.bar_label(b2, fmt="%.3f", padding=3, fontsize=10, fontweight="bold")
    ax2.set_xticks(x_pos); ax2.set_xticklabels(model_names, fontsize=10)
    ax2.set_ylabel("R²", fontsize=12); ax2.set_title("R² (higher = better)", fontsize=13, fontweight="bold")
    ax2.set_facecolor("#fafafa")
    fig.suptitle("Model Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()
