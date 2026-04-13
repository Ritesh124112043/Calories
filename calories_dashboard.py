"""
Calories Burn Prediction Dashboard
===================================
ML Project Dashboard built with Streamlit + XGBoost

HOW TO RUN:
  1. Install dependencies:
       pip install streamlit xgboost scikit-learn pandas numpy plotly
  2. Place calories.csv in the same folder as this script
     (or adjust the path in CALORIES_PATH below)
  3. Run:
       streamlit run calories_dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor

# ─── CONFIG ────────────────────────────────────────────────────────────────────
CALORIES_PATH = "calories.csv"   # <-- adjust path if needed

st.set_page_config(
    page_title="Calories Burn Predictor",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CUSTOM CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.4rem;
        font-weight: 700;
        background: linear-gradient(135deg, #FF6B35, #F7931E);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #6B7280;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        padding: 1rem 1.2rem;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(102,126,234,0.3);
    }
    .metric-card h2 { font-size: 2rem; margin: 0; }
    .metric-card p  { font-size: 0.85rem; margin: 0; opacity: 0.85; }
    .prediction-box {
        background: linear-gradient(135deg, #FF6B35, #F7931E);
        border-radius: 16px;
        padding: 1.5rem 2rem;
        color: white;
        text-align: center;
        box-shadow: 0 6px 20px rgba(255,107,53,0.35);
    }
    .prediction-box h1 { font-size: 3.5rem; margin: 0; }
    .prediction-box p  { font-size: 1rem; opacity: 0.9; margin: 0; }
    .section-title {
        font-size: 1.25rem;
        font-weight: 600;
        color: #1F2937;
        border-left: 4px solid #FF6B35;
        padding-left: 0.75rem;
        margin: 1.5rem 0 1rem;
    }
</style>
""", unsafe_allow_html=True)


# ─── DATA & MODEL ──────────────────────────────────────────────────────────────
@st.cache_data
def load_and_prepare_data(path):
    """Generate synthetic exercise data merged with real calories."""
    calories = pd.read_csv(path)

    np.random.seed(42)
    n = len(calories)
    exercise = pd.DataFrame({
        "User_ID": calories["User_ID"],
        "Gender":  np.random.choice(["male", "female"], size=n),
        "Age":     np.random.randint(20, 60, size=n),
        "Height":  np.random.uniform(155, 190, size=n).round(1),
        "Weight":  np.random.uniform(45, 100, size=n).round(1),
        "Duration": np.random.randint(5, 60, size=n),
        "Heart_Rate": np.random.randint(60, 150, size=n),
        "Body_Temp": np.random.uniform(37, 41, size=n).round(1),
    })

    energy = pd.concat([exercise, calories["Calories"]], axis=1)
    energy = energy.drop(columns="User_ID")

    enc = LabelEncoder()
    energy["Gender"] = enc.fit_transform(energy["Gender"])   # male=1, female=0
    return energy, enc


@st.cache_resource
def train_model(data):
    X = data.drop(columns="Calories")
    Y = data["Calories"]
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=2
    )
    model = XGBRegressor(n_estimators=200, learning_rate=0.05,
                         max_depth=5, random_state=42)
    model.fit(X_train, Y_train)

    train_pred = model.predict(X_train)
    test_pred  = model.predict(X_test)

    metrics = {
        "Train MAE": mean_absolute_error(Y_train, train_pred),
        "Test MAE":  mean_absolute_error(Y_test,  test_pred),
        "Train R²":  r2_score(Y_train, train_pred),
        "Test R²":   r2_score(Y_test,  test_pred),
    }
    return model, X_train, X_test, Y_train, Y_test, train_pred, test_pred, metrics


# ─── LOAD ──────────────────────────────────────────────────────────────────────
try:
    data, encoder = load_and_prepare_data(CALORIES_PATH)
except FileNotFoundError:
    st.error(f"⚠️  Could not find `{CALORIES_PATH}`. Place the file in the same "
             "directory as this script, or update the CALORIES_PATH variable.")
    st.stop()

(model, X_train, X_test, Y_train, Y_test,
 train_pred, test_pred, model_metrics) = train_model(data)


# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/fire-element.png", width=72)
    st.markdown("## 🔥 Calories Predictor")
    st.markdown("---")
    st.markdown("### 🎛️ Predict for a Person")

    gender_label = st.selectbox("Gender", ["Male", "Female"])
    gender_val   = 1 if gender_label == "Male" else 0

    age     = st.slider("Age (years)",      15, 80, 25)
    height  = st.slider("Height (cm)",     140, 200, 170)
    weight  = st.slider("Weight (kg)",      40, 120, 70)
    duration= st.slider("Exercise Duration (min)", 5, 120, 30)
    hr      = st.slider("Heart Rate (bpm)",  60, 180, 110)
    btemp   = st.slider("Body Temperature (°C)", 36.0, 42.0, 38.5, 0.1)

    predict_btn = st.button("🔥 Predict Calories", use_container_width=True)

    st.markdown("---")
    page = st.radio("📊 Navigate",
                    ["Overview", "Data Explorer", "Model Performance", "Feature Insights"])


# ─── HEADER ───────────────────────────────────────────────────────────────────
st.markdown('<p class="main-header">🔥 Calories Burn Prediction Dashboard</p>',
            unsafe_allow_html=True)
st.markdown('<p class="sub-header">XGBoost-powered ML model · Exercise Analytics · 15,000 records</p>',
            unsafe_allow_html=True)

# Prediction result (always shown at top when button clicked)
if predict_btn:
    input_arr = np.array([[gender_val, age, height, weight, duration, hr, btemp]])
    pred_cal  = model.predict(input_arr)[0]
    col_p1, col_p2 = st.columns([1, 2])
    with col_p1:
        st.markdown(f"""
        <div class="prediction-box">
            <p>Estimated Calories Burned</p>
            <h1>{pred_cal:.1f}</h1>
            <p>kcal 🔥</p>
        </div>""", unsafe_allow_html=True)
    with col_p2:
        # Gauge chart
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=pred_cal,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": "Calorie Burn Level", "font": {"size": 18}},
            delta={"reference": float(data["Calories"].mean()),
                   "increasing": {"color": "#FF6B35"}},
            gauge={
                "axis": {"range": [0, 320], "tickwidth": 1},
                "bar":  {"color": "#FF6B35"},
                "steps": [
                    {"range": [0,   80],  "color": "#D1FAE5"},
                    {"range": [80,  160], "color": "#FEF3C7"},
                    {"range": [160, 320], "color": "#FEE2E2"},
                ],
                "threshold": {
                    "line": {"color": "red", "width": 3},
                    "thickness": 0.75,
                    "value": float(data["Calories"].mean()),
                },
            }
        ))
        fig_gauge.update_layout(height=260, margin=dict(t=30, b=0, l=20, r=20))
        st.plotly_chart(fig_gauge, use_container_width=True)
    st.success(f"🏃 A **{gender_label}**, age **{age}**, exercising for **{duration} min** with "
               f"HR **{hr} bpm** is estimated to burn **{pred_cal:.1f} kcal**.")

st.markdown("---")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "Overview":
    # KPI row
    c1, c2, c3, c4 = st.columns(4)
    kpis = [
        ("Total Records", f"{len(data):,}"),
        ("Avg Calories",  f"{data['Calories'].mean():.1f}"),
        ("Max Calories",  f"{data['Calories'].max():.0f}"),
        ("Model R² (Test)", f"{model_metrics['Test R²']:.4f}"),
    ]
    colors = [
        "linear-gradient(135deg,#667eea,#764ba2)",
        "linear-gradient(135deg,#f093fb,#f5576c)",
        "linear-gradient(135deg,#4facfe,#00f2fe)",
        "linear-gradient(135deg,#43e97b,#38f9d7)",
    ]
    for col, (label, val), color in zip([c1,c2,c3,c4], kpis, colors):
        col.markdown(f"""
        <div style="background:{color};border-radius:12px;padding:1rem 1.2rem;
                    color:white;text-align:center;box-shadow:0 4px 15px rgba(0,0,0,.15)">
            <h2 style="font-size:1.9rem;margin:0">{val}</h2>
            <p style="font-size:.85rem;margin:0;opacity:.85">{label}</p>
        </div>""", unsafe_allow_html=True)

    st.markdown('<p class="section-title">Calories Distribution</p>', unsafe_allow_html=True)
    col_a, col_b = st.columns(2)

    with col_a:
        fig_hist = px.histogram(
            data, x="Calories", nbins=50, color_discrete_sequence=["#FF6B35"],
            title="Distribution of Calories Burned",
            labels={"Calories": "Calories (kcal)"}
        )
        fig_hist.update_layout(bargap=0.05, height=320)
        st.plotly_chart(fig_hist, use_container_width=True)

    with col_b:
        fig_box = px.box(
            data, y="Calories", color_discrete_sequence=["#764ba2"],
            title="Calorie Spread (Box Plot)",
            points="outliers"
        )
        fig_box.update_layout(height=320)
        st.plotly_chart(fig_box, use_container_width=True)

    st.markdown('<p class="section-title">Key Correlations</p>', unsafe_allow_html=True)
    corr = data.corr(numeric_only=True)
    fig_corr = px.imshow(
        corr, text_auto=".2f",
        color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
        title="Feature Correlation Matrix"
    )
    fig_corr.update_layout(height=420)
    st.plotly_chart(fig_corr, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: DATA EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Data Explorer":
    st.markdown('<p class="section-title">Raw Dataset (first 500 rows)</p>',
                unsafe_allow_html=True)
    display_df = data.copy()
    display_df["Gender"] = display_df["Gender"].map({1: "Male", 0: "Female"})
    st.dataframe(display_df.head(500), use_container_width=True, height=300)

    st.markdown('<p class="section-title">Univariate Analysis</p>', unsafe_allow_html=True)
    feat = st.selectbox("Select a feature to explore", data.columns.tolist())
    col_u1, col_u2 = st.columns(2)
    with col_u1:
        fig_u = px.histogram(data, x=feat, nbins=40,
                             color_discrete_sequence=["#667eea"],
                             title=f"Distribution of {feat}")
        st.plotly_chart(fig_u, use_container_width=True)
    with col_u2:
        fig_vs = px.scatter(data, x=feat, y="Calories",
                            color="Calories",
                            color_continuous_scale="Viridis",
                            opacity=0.4,
                            title=f"{feat} vs Calories")
        st.plotly_chart(fig_vs, use_container_width=True)

    st.markdown('<p class="section-title">Statistical Summary</p>', unsafe_allow_html=True)
    st.dataframe(data.describe().T.style.background_gradient(cmap="Blues"),
                 use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Model Performance":
    st.markdown('<p class="section-title">Model Metrics</p>', unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    metric_items = [
        ("Train MAE", f"{model_metrics['Train MAE']:.3f} kcal"),
        ("Test MAE",  f"{model_metrics['Test MAE']:.3f} kcal"),
        ("Train R²",  f"{model_metrics['Train R²']:.4f}"),
        ("Test R²",   f"{model_metrics['Test R²']:.4f}"),
    ]
    colors2 = [
        "linear-gradient(135deg,#a18cd1,#fbc2eb)",
        "linear-gradient(135deg,#ffecd2,#fcb69f)",
        "linear-gradient(135deg,#a1c4fd,#c2e9fb)",
        "linear-gradient(135deg,#d4fc79,#96e6a1)",
    ]
    for col, (lbl, val), clr in zip([m1,m2,m3,m4], metric_items, colors2):
        col.markdown(f"""
        <div style="background:{clr};border-radius:12px;padding:1rem;
                    text-align:center;box-shadow:0 3px 10px rgba(0,0,0,.1)">
            <h2 style="font-size:1.7rem;margin:0;color:#1F2937">{val}</h2>
            <p style="font-size:.85rem;margin:0;color:#374151">{lbl}</p>
        </div>""", unsafe_allow_html=True)

    st.markdown('<p class="section-title">Actual vs Predicted (Test Set)</p>',
                unsafe_allow_html=True)
    results_df = pd.DataFrame({
        "Actual":    Y_test.values,
        "Predicted": test_pred,
        "Error":     np.abs(Y_test.values - test_pred),
    })
    fig_av = px.scatter(
        results_df, x="Actual", y="Predicted", color="Error",
        color_continuous_scale="Inferno",
        title="Actual vs Predicted Calories (Test Set)",
        labels={"Actual": "Actual Calories", "Predicted": "Predicted Calories"},
        opacity=0.6,
    )
    # Perfect prediction line
    mn, mx = results_df["Actual"].min(), results_df["Actual"].max()
    fig_av.add_shape(type="line", x0=mn, y0=mn, x1=mx, y1=mx,
                     line=dict(color="red", width=2, dash="dash"))
    fig_av.update_layout(height=420)
    st.plotly_chart(fig_av, use_container_width=True)

    st.markdown('<p class="section-title">Residual Distribution</p>', unsafe_allow_html=True)
    residuals = Y_test.values - test_pred
    fig_res = px.histogram(
        pd.DataFrame({"Residual": residuals}),
        x="Residual", nbins=60,
        color_discrete_sequence=["#F7931E"],
        title="Prediction Error Distribution (Residuals)",
    )
    fig_res.add_vline(x=0, line_dash="dash", line_color="red", line_width=2)
    fig_res.update_layout(height=320)
    st.plotly_chart(fig_res, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: FEATURE INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Feature Insights":
    st.markdown('<p class="section-title">Feature Importances (XGBoost)</p>',
                unsafe_allow_html=True)
    feature_names = X_train.columns.tolist()
    importances   = model.feature_importances_
    fi_df = (pd.DataFrame({"Feature": feature_names, "Importance": importances})
               .sort_values("Importance", ascending=True))

    fig_fi = px.bar(
        fi_df, x="Importance", y="Feature", orientation="h",
        color="Importance", color_continuous_scale="Plasma",
        title="XGBoost Feature Importance Scores",
    )
    fig_fi.update_layout(height=380, coloraxis_showscale=False)
    st.plotly_chart(fig_fi, use_container_width=True)

    st.markdown('<p class="section-title">Feature vs Calories (Scatter Matrix)</p>',
                unsafe_allow_html=True)
    top_features = fi_df.sort_values("Importance", ascending=False)["Feature"].head(4).tolist()
    sample = data.sample(min(2000, len(data)), random_state=42)
    fig_sp = px.scatter_matrix(
        sample,
        dimensions=top_features + ["Calories"],
        color="Calories",
        color_continuous_scale="Viridis",
        opacity=0.4,
        title="Top Features × Calories",
    )
    fig_sp.update_traces(diagonal_visible=False, marker_size=3)
    fig_sp.update_layout(height=560)
    st.plotly_chart(fig_sp, use_container_width=True)

    st.markdown('<p class="section-title">Duration & Heart Rate vs Calories</p>',
                unsafe_allow_html=True)
    fig_3d = px.scatter_3d(
        sample,
        x="Duration", y="Heart_Rate", z="Calories",
        color="Calories", color_continuous_scale="Inferno",
        opacity=0.5,
        title="Duration × Heart Rate × Calories (3D)",
        labels={"Duration": "Duration (min)", "Heart_Rate": "Heart Rate (bpm)"},
    )
    fig_3d.update_layout(height=480)
    st.plotly_chart(fig_3d, use_container_width=True)


# ─── FOOTER ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#9CA3AF;font-size:.85rem'>"
    "🔥 Calories Burn Prediction · XGBoost Model · Built with Streamlit"
    "</p>",
    unsafe_allow_html=True,
)
