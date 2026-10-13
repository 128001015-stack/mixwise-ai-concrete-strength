import os
import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.inspection import permutation_importance

st.set_page_config(page_title="MixWise AI | Concrete Lab", page_icon="🧱", layout="wide")

# ---------- Visual identity ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
html, body, [class*="css"] {font-family: 'DM Sans', sans-serif;}
.stApp {background: linear-gradient(135deg,#f4f7fb 0%,#eef3f7 55%,#f8f5ee 100%);}
.block-container {padding-top: 1.2rem; max-width: 1400px;}
.hero {background: linear-gradient(120deg,#12263a,#1c4966); color:white; padding:28px 32px; border-radius:22px; margin-bottom:18px; box-shadow:0 12px 30px rgba(18,38,58,.14);}
.hero h1 {font-family:'Space Grotesk',sans-serif; font-size:2.2rem; margin:0;}
.hero p {color:#d9e8f1; margin:.45rem 0 0 0; font-size:1rem;}
.eyebrow {color:#8ee0c0; font-weight:700; letter-spacing:2px; font-size:.72rem; text-transform:uppercase;}
.panel {background:white; padding:20px 22px; border:1px solid #e1e8ee; border-radius:18px; box-shadow:0 4px 16px rgba(28,54,75,.045); margin-bottom:14px;}
.tag {display:inline-block; padding:5px 10px; border-radius:99px; background:#e1f5ed; color:#17654d; font-size:.78rem; font-weight:700;}
.warning {background:#fff5df; border-left:5px solid #e4a32b; padding:14px 16px; border-radius:10px; color:#654b18;}
.small-muted {font-size:.84rem;color:#607184;}
div[data-testid="stMetric"] {background:white; border:1px solid #e1e8ee; padding:14px 16px; border-radius:14px;}
.stButton>button {border-radius:10px; font-weight:700;}
</style>
""", unsafe_allow_html=True)

DATA_PATH = os.path.join(os.path.dirname(__file__), "Concrete_Data.xls")
COLS = ["Cement", "Blast Furnace Slag", "Fly Ash", "Water", "Superplasticizer",
        "Coarse Aggregate", "Fine Aggregate", "Age", "Concrete Compressive Strength"]
FEATURES = COLS[:-1]
TARGET = COLS[-1]
UNITS = {
    "Cement":"kg/m³", "Blast Furnace Slag":"kg/m³", "Fly Ash":"kg/m³",
    "Water":"kg/m³", "Superplasticizer":"kg/m³", "Coarse Aggregate":"kg/m³",
    "Fine Aggregate":"kg/m³", "Age":"days"
}

@st.cache_data
def load_data():
    df = pd.read_excel(DATA_PATH, engine="xlrd")
    df.columns = COLS
    return df

@st.cache_resource
def train_model(data):
    X = data[FEATURES]
    y = data[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    model = RandomForestRegressor(
        n_estimators=350, max_features=0.9, min_samples_leaf=1,
        random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    metrics = {
        "R²": r2_score(y_test, pred),
        "MAE (MPa)": mean_absolute_error(y_test, pred),
        "RMSE (MPa)": np.sqrt(mean_squared_error(y_test, pred)),
    }
    return model, X_test, y_test, pred, metrics

df = load_data()
model, X_test, y_test, test_pred, metrics = train_model(df)

with st.sidebar:
    st.markdown("## 🧱 MixWise AI")
    st.caption("Concrete Intelligence Studio")
    page = st.radio("NAVIGATION", ["Overview", "Strength Predictor", "Mix Design Explorer", "Dataset & Insights", "Model Performance", "AI Engineering Assistant"])
    st.divider()
    st.markdown("**Model**")
    st.write("Random Forest Regression")
    st.write("80/20 train-test split · seed 42")
    st.caption("Dataset: UCI Concrete Compressive Strength (Yeh, 1998)")
    st.divider()
    st.caption("Academic demonstration • Not for structural approval")

def header(title, subtitle):
    st.markdown(f'<div class="hero"><div class="eyebrow">MIXWISE AI · CIVIL ENGINEERING</div><h1>{title}</h1><p>{subtitle}</p></div>', unsafe_allow_html=True)

def feature_inputs(prefix="mix"):
    defaults = {
        "Cement": 300.0, "Blast Furnace Slag": 0.0, "Fly Ash": 0.0,
        "Water": 180.0, "Superplasticizer": 6.0, "Coarse Aggregate": 800.0,
        "Fine Aggregate": 600.0, "Age": 28
    }
    ranges = {
        "Cement": (0.0, 600.0), "Blast Furnace Slag": (0.0, 400.0),
        "Fly Ash": (0.0, 250.0), "Water": (80.0, 300.0),
        "Superplasticizer": (0.0, 40.0), "Coarse Aggregate": (500.0, 1300.0),
        "Fine Aggregate": (400.0, 1100.0), "Age": (1, 365)
    }
    vals = {}
    cols = st.columns(2)
    for i, name in enumerate(FEATURES):
        lo, hi = ranges[name]
        step = 1.0 if name == "Age" else 1.0
        with cols[i % 2]:
            vals[name] = st.number_input(
                f"{name} ({UNITS[name]})", min_value=float(lo), max_value=float(hi),
                value=float(defaults[name]), step=step, key=f"{prefix}_{name}"
            )
    vals["Age"] = int(vals["Age"])
    return pd.DataFrame([vals], columns=FEATURES)

def mix_flags(row):
    flags = []
    wc = row["Water"] / row["Cement"] if row["Cement"] > 0 else None
    if wc is None:
        flags.append("Cement is zero, so the water/cement ratio cannot be calculated.")
    elif wc > 0.60:
        flags.append(f"Water/cement ratio is {wc:.2f}; review durability and strength requirements.")
    elif wc < 0.25:
        flags.append(f"Water/cement ratio is {wc:.2f}; verify workability and compaction feasibility.")
    total = sum(float(row[x]) for x in FEATURES[:7])
    if not 1800 <= total <= 2800:
        flags.append(f"Approximate ingredient total is {total:.0f} kg/m³; check units and mix proportions.")
    if row["Age"] < 1 or row["Age"] > 365:
        flags.append("Age is outside the dataset's 1–365 day range.")
    return wc, total, flags

if page == "Overview":
    header("Concrete strength, made clearer.", "A data-driven workspace to explore mix ingredients, estimate compressive strength, and understand model behaviour.")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Lab records", f"{len(df):,}")
    c2.metric("Input variables", "8")
    c3.metric("Average strength", f"{df[TARGET].mean():.2f} MPa")
    c4.metric("Strength range", f"{df[TARGET].min():.1f}–{df[TARGET].max():.1f} MPa")
    left,right = st.columns([1.25,1])
    with left:
        st.markdown('<div class="panel"><h3>🧭 What you can do</h3><p>• Predict strength from ingredient proportions and curing age.<br>• Explore cement, water, aggregate and age relationships.<br>• Inspect correlations and feature importance.<br>• Review test-set performance using R², MAE and RMSE.<br>• Compare candidate water/admixture combinations for further investigation.</p></div>', unsafe_allow_html=True)
    with right:
        st.markdown('<div class="panel"><h3>🔬 How it works</h3><p>1. Load the UCI laboratory dataset.<br>2. Train a Random Forest regression model.<br>3. Evaluate on a held-out 20% test set.<br>4. Use the model for interactive estimates.</p><span class="tag">Educational decision support</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="warning"><b>Engineering disclaimer:</b> Predictions are statistical estimates, not certified mix designs. Validate all proportions through applicable standards, trial mixes, workability and durability checks, and laboratory testing before any real-world use. Never rely on this app alone for structural safety decisions.</div>', unsafe_allow_html=True)
    st.caption("Dataset attribution: Yeh, I. (1998), Concrete Compressive Strength, UCI Machine Learning Repository. CC BY 4.0.")

elif page == "Strength Predictor":
    header("Strength Predictor", "Enter a concrete mix and estimate its compressive strength at the specified age.")
    st.markdown('<div class="panel"><b>Mix parameters</b><p class="small-muted">Use ingredient quantities per cubic metre. Values are prefilled with an editable example.</p>', unsafe_allow_html=True)
    sample = feature_inputs("predict")
    st.markdown("</div>", unsafe_allow_html=True)
    if st.button("🔮 Predict compressive strength", type="primary", use_container_width=True):
        pred = float(model.predict(sample)[0])
        wc, total, flags = mix_flags(sample.iloc[0])
        st.markdown("### Prediction result")
        a,b,c = st.columns(3)
        a.metric("Estimated strength", f"{pred:.2f} MPa")
        b.metric("Water / cement ratio", f"{wc:.3f}" if wc is not None else "N/A")
        c.metric("Approx. ingredient total", f"{total:.0f} kg/m³")
        st.progress(min(max(pred / 80, 0), 1), text="Prediction position relative to the dataset's approximate strength span")
        if flags:
            for flag in flags:
                st.warning(flag)
        else:
            st.info("No simple range warning was triggered. This does not confirm code compliance.")
        st.markdown('<div class="warning"><b>Do not use as final mix approval.</b> Model predictions can be wrong, especially for unusual mixes or conditions not represented in the training data. Confirm with a qualified engineer and laboratory trial mixes.</div>', unsafe_allow_html=True)
        st.dataframe(sample, use_container_width=True, hide_index=True)

elif page == "Mix Design Explorer":
    header("Mix Design Explorer", "Screen candidate water and superplasticizer values while holding the other ingredients fixed.")
    st.info("This is an AI-based screening aid, not a standards-compliant mix-design solver. The closest candidate may not be workable, durable, economical, or code-compliant.")
    base = feature_inputs("explore")
    r1,r2 = st.columns(2)
    with r1:
        target = st.number_input("Target compressive strength (MPa)", min_value=5.0, max_value=80.0, value=40.0, step=1.0)
        water_min = st.slider("Water search range (kg/m³)", 100, 260, (140, 220))
    with r2:
        age = st.number_input("Target age (days)", min_value=1, max_value=365, value=28)
        sp_min = st.slider("Superplasticizer search range (kg/m³)", 0, 30, (0, 12))
    if st.button("🧪 Screen candidate combinations", type="primary", use_container_width=True):
        rows = []
        fixed = base.iloc[0].to_dict()
        for water in np.arange(water_min[0], water_min[1] + 0.1, 5):
            for sp in np.arange(sp_min[0], sp_min[1] + 0.1, 1):
                item = fixed.copy()
                item["Water"] = float(water)
                item["Superplasticizer"] = float(sp)
                item["Age"] = int(age)
                rows.append(item)
        candidates = pd.DataFrame(rows, columns=FEATURES)
        candidates["Predicted strength (MPa)"] = model.predict(candidates[FEATURES])
        candidates["Absolute target gap (MPa)"] = abs(candidates["Predicted strength (MPa)"] - target)
        candidates["Water/cement ratio"] = candidates["Water"] / candidates["Cement"].replace(0, np.nan)
        candidates = candidates.sort_values("Absolute target gap (MPa)")
        st.markdown("### Closest model-estimated candidates")
        st.dataframe(candidates.head(10).round(3), use_container_width=True, hide_index=True)
        best = candidates.iloc[0]
        st.success(f"Closest candidate in the searched grid: {best['Predicted strength (MPa)']:.2f} MPa; target gap {best['Absolute target gap (MPa)']:.2f} MPa.")
        st.warning("A close prediction does not mean the target is achieved in practice. Check binder limits, water/binder ratio, aggregate grading, exposure class, workability, admixture compatibility and relevant IS 10262 / IS 456 requirements, then validate with trial batches.")
        fig, ax = plt.subplots()
        ax.scatter(candidates["Water"], candidates["Predicted strength (MPa)"], c=candidates["Superplasticizer"], alpha=.8)
        ax.axhline(target, linestyle="--")
        ax.set_xlabel("Water (kg/m³)"); ax.set_ylabel("Predicted strength (MPa)")
        ax.set_title("Candidate screening — colour represents superplasticizer dosage")
        st.pyplot(fig, use_container_width=True)

elif page == "Dataset & Insights":
    header("Dataset & Insights", "Explore the supplied laboratory dataset and the relationships between variables.")
    a,b,c = st.columns(3)
    a.metric("Rows", f"{df.shape[0]:,}")
    b.metric("Columns", df.shape[1])
    c.metric("Missing cells", int(df.isna().sum().sum()))
    with st.expander("View dataset", expanded=True):
        st.dataframe(df, use_container_width=True, height=300)
        st.download_button("Download dataset as CSV", df.to_csv(index=False).encode("utf-8"), "concrete_strength_dataset.csv", "text/csv")
    st.markdown("### Variable distributions")
    chosen = st.selectbox("Choose a variable", COLS)
    fig, ax = plt.subplots()
    ax.hist(df[chosen], bins=28, edgecolor="white")
    ax.set_xlabel(chosen); ax.set_ylabel("Frequency"); ax.set_title(f"Distribution of {chosen}")
    st.pyplot(fig, use_container_width=True)
    st.markdown("### Ingredient vs compressive strength")
    chosen_x = st.selectbox("Choose an input variable", FEATURES, index=0)
    fig, ax = plt.subplots()
    ax.scatter(df[chosen_x], df[TARGET], alpha=.45)
    ax.set_xlabel(f"{chosen_x} ({UNITS[chosen_x]})"); ax.set_ylabel("Compressive strength (MPa)")
    ax.set_title(f"{chosen_x} vs compressive strength")
    st.pyplot(fig, use_container_width=True)
    st.markdown("### Correlation matrix")
    corr = df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(10,7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0, ax=ax)
    st.pyplot(fig, use_container_width=True)
    st.caption("Correlation describes association, not causation.")

elif page == "Model Performance":
    header("Model Performance", "Transparent evaluation on a held-out test set that was not used to fit the model.")
    a,b,c = st.columns(3)
    a.metric("R²", f"{metrics['R²']:.3f}")
    b.metric("MAE", f"{metrics['MAE (MPa)']:.2f} MPa")
    c.metric("RMSE", f"{metrics['RMSE (MPa)']:.2f} MPa")
    st.caption("The displayed metrics are calculated for this run and split; they are not guaranteed performance on new concrete mixes.")
    fig, ax = plt.subplots()
    ax.scatter(y_test, test_pred, alpha=.6)
    low = min(float(y_test.min()), float(test_pred.min()))
    high = max(float(y_test.max()), float(test_pred.max()))
    ax.plot([low, high], [low, high], linestyle="--")
    ax.set_xlabel("Actual strength (MPa)"); ax.set_ylabel("Predicted strength (MPa)")
    ax.set_title("Actual vs predicted strength")
    st.pyplot(fig, use_container_width=True)
    residuals = y_test.to_numpy() - test_pred
    fig, ax = plt.subplots()
    ax.scatter(test_pred, residuals, alpha=.6)
    ax.axhline(0, linestyle="--")
    ax.set_xlabel("Predicted strength (MPa)"); ax.set_ylabel("Residual (actual − predicted)")
    ax.set_title("Residual analysis")
    st.pyplot(fig, use_container_width=True)
    imp = pd.Series(model.feature_importances_, index=FEATURES).sort_values()
    fig, ax = plt.subplots()
    imp.plot(kind="barh", ax=ax)
    ax.set_xlabel("Random Forest feature importance"); ax.set_title("Relative feature importance")
    st.pyplot(fig, use_container_width=True)
    st.info("Feature importance is model-specific and should not be interpreted as proof that changing an ingredient will cause a particular strength change.")

elif page == "AI Engineering Assistant":
    header("AI Engineering Assistant", "Quick, grounded answers about this dataset, model and responsible use.")
    question = st.selectbox("Choose a question", [
        "What does the app predict?",
        "What inputs does the model use?",
        "What is Random Forest regression?",
        "How should I interpret R², MAE and RMSE?",
        "Can this app approve a concrete mix for construction?",
        "How should a 40 MPa target mix be developed?"
    ])
    answers = {
        "What does the app predict?": "It estimates concrete compressive strength in MPa from seven ingredient quantities and the concrete age in days. It learns patterns from historical laboratory records.",
        "What inputs does the model use?": "Cement, blast furnace slag, fly ash, water, superplasticizer, coarse aggregate, fine aggregate, and age. Ingredient quantities are expressed in kg/m³; age is in days.",
        "What is Random Forest regression?": "Random Forest combines predictions from many decision trees trained on varied samples/features. Averaging their predictions can model nonlinear relationships and reduce the variance of a single tree.",
        "How should I interpret R², MAE and RMSE?": "R² measures how much variation is explained on the test set (higher is generally better). MAE is the average absolute error in MPa. RMSE is the square root of mean squared error and penalizes larger errors more strongly. Always consider all metrics and the test-set context.",
        "Can this app approve a concrete mix for construction?": "No. This is an educational prediction tool, not a certified mix-design or structural-safety system. A qualified engineer must check relevant standards, exposure and durability requirements, workability, material properties and trial-mix laboratory results.",
        "How should a 40 MPa target mix be developed?": "Use the applicable mix-design standard and project requirements to establish target mean strength, water/binder ratio, cementitious content, aggregate proportions and admixture compatibility. Prepare trial batches, test fresh properties and compressive strength at specified ages, and adjust based on lab evidence. The AI explorer can only help screen candidate values."
    }
    st.markdown(f'<div class="panel"><h3>💬 Answer</h3><p>{answers[question]}</p></div>', unsafe_allow_html=True)
    st.markdown("### Ask a custom question")
    custom = st.text_input("Type a question about the app or machine-learning concepts")
    if custom:
        st.info("This offline assistant is limited to the project topics above. For project-specific or engineering-critical questions, consult your instructor and a qualified concrete/mix-design engineer. Do not treat generated advice as approval for construction.")
    st.markdown('<div class="warning"><b>Safety reminder:</b> AI-generated recommendations must be checked through laboratory testing and engineering judgement before implementation.</div>', unsafe_allow_html=True)

st.markdown("---")
st.caption("MixWise AI · Student project for Data Analytics for Civil Engineering · Dataset: UCI Concrete Compressive Strength (Yeh, 1998), CC BY 4.0.")



st.sidebar.markdown("---")
st.sidebar.markdown("### Developed by")
st.sidebar.markdown("**Jeevitha B.** \nB.Tech Civil Engineering Student")
