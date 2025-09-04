import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from io import StringIO

st.set_page_config(page_title="Cohort Intelligence + Feedback Loop", layout="wide")

st.title("AI Agent 1 + Agent 2: Cohort Intelligence & Feedback Loop")

st.markdown(
    "This app demonstrates how **Agent 1 (Cohort Intelligence)** identifies high-potential cohorts and "
    "how **Agent 2 (Feedback Loop)** improves the model after observing actual outcomes. "
    "You can upload your own CSV or use the bundled mock dataset."
)

# ----------------------
# Data Loading / Fallback
# ----------------------
@st.cache_data
def load_default_data():
    try:
        df = pd.read_csv("mock_customer_data_extended.csv")
        return df
    except Exception:
        # Fallback: generate a synthetic dataset with similar columns
        np.random.seed(42)
        n = 500
        user_ids = [f"user_{i}" for i in range(1, n+1)]
        categories = np.random.choice(["Groceries", "Travel", "Electronics", "Fashion"], n)
        payment_types = np.random.choice(["UPI", "Wallet", "OtherCard", "NeuCard"], n)
        spend_amount = np.random.randint(200, 20000, n)
        neucoins_redeemed = np.random.randint(0, 500, n)
        applied = np.random.choice([0, 1], n, p=[0.7, 0.3])

        spend_7d = np.random.randint(0, 5000, n)
        spend_30d = spend_7d + np.random.randint(0, 8000, n)
        spend_60d = spend_30d + np.random.randint(0, 10000, n)
        spend_6m = spend_60d + np.random.randint(0, 30000, n)
        spend_1y = spend_6m + np.random.randint(0, 50000, n)

        tx_7d = np.random.randint(0, 10, n)
        tx_30d = tx_7d + np.random.randint(0, 20, n)
        tx_60d = tx_30d + np.random.randint(0, 40, n)
        tx_6m = tx_60d + np.random.randint(0, 80, n)
        tx_1y = tx_6m + np.random.randint(0, 150, n)

        days_since_last_tx = np.random.randint(0, 60, n)
        spend_share = np.random.dirichlet(np.ones(4), n)
        spend_share_grocery = spend_share[:, 0]
        spend_share_travel = spend_share[:, 1]
        spend_share_electronics = spend_share[:, 2]
        spend_share_fashion = spend_share[:, 3]

        # top 3 categories string
        top_categories = []
        for row in spend_share:
            cats = ["Groceries", "Travel", "Electronics", "Fashion"]
            cats_sorted = [x for _, x in sorted(zip(row, cats), reverse=True)]
            top_categories.append(",".join(cats_sorted[:3]))

        pct_credit_card_spend = np.random.uniform(0, 1, n)
        is_credit_carded = (pct_credit_card_spend > 0.2).astype(int)
        recency_score = np.exp(-days_since_last_tx / 30)
        bureau_score = np.random.randint(300, 901, n)
        income_range = np.random.choice(["<3L", "3L-6L", "6L-10L", "10L-20L", "20L+"], n, p=[0.1,0.25,0.3,0.25,0.1])

        df = pd.DataFrame({
            "user_id": user_ids,
            "spend_category": categories,
            "payment_type": payment_types,
            "spend_amount": spend_amount,
            "neucoins_redeemed": neucoins_redeemed,
            "applied": applied,
            "spend_7d": spend_7d,
            "spend_30d": spend_30d,
            "spend_60d": spend_60d,
            "spend_6m": spend_6m,
            "spend_1y": spend_1y,
            "tx_7d": tx_7d,
            "tx_30d": tx_30d,
            "tx_60d": tx_60d,
            "tx_6m": tx_6m,
            "tx_1y": tx_1y,
            "days_since_last_tx": days_since_last_tx,
            "spend_share_grocery": spend_share_grocery,
            "spend_share_travel": spend_share_travel,
            "spend_share_electronics": spend_share_electronics,
            "spend_share_fashion": spend_share_fashion,
            "top_3_categories": top_categories,
            "pct_credit_card_spend": pct_credit_card_spend,
            "is_credit_carded": is_credit_carded,
            "recency_score": recency_score,
            "bureau_score": bureau_score,
            "income_range": income_range
        })
        return df

uploaded = st.file_uploader("Upload your CSV (optional). If empty, we'll use the default mock dataset.", type=["csv"])
if uploaded is not None:
    df = pd.read_csv(uploaded)
else:
    df = load_default_data()

with st.expander("Preview Data", expanded=False):
    st.dataframe(df.head(20))

# ----------------------
# Train Baseline Model (Agent 1)
# ----------------------
st.header("Agent 1: Cohort Intelligence (Baseline Model)")

feature_cols = [
    "spend_30d", "spend_60d", "spend_6m", "spend_1y",
    "tx_30d", "tx_60d", "tx_6m", "tx_1y",
    "days_since_last_tx", "pct_credit_card_spend",
    "is_credit_carded", "recency_score", "bureau_score"
]
# ensure missing are filled
df_model = df.copy()
for col in feature_cols + ["applied"]:
    if col not in df_model.columns:
        st.error(f"Missing required column: {col}")
        st.stop()
df_model = df_model[feature_cols + ["applied","user_id","spend_category","payment_type","top_3_categories"]].dropna()

X = df_model[feature_cols]
y = df_model["applied"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
model = GradientBoostingClassifier(random_state=42)
model.fit(X_train, y_train)

y_prob = model.predict_proba(X_test)[:,1]
auc_roc = roc_auc_score(y_test, y_prob)
prec, rec, thr = precision_recall_curve(y_test, y_prob)
auc_pr = auc(rec, prec)

col1, col2 = st.columns(2)
with col1:
    st.metric("Baseline AUC-ROC", f"{auc_roc:.2f}")
with col2:
    st.metric("Baseline AUC-PR", f"{auc_pr:.2f}")

df_model["score_before"] = model.predict_proba(df_model[feature_cols])[:,1]

# Cohort summary
cohort_by_category = df_model.groupby("spend_category")["score_before"].mean().sort_values(ascending=False).to_frame("avg_score_before")
cohort_by_payment = df_model.groupby("payment_type")["score_before"].mean().sort_values(ascending=False).to_frame("avg_score_before")

st.subheader("Top Cohorts by Predicted Score (Before Feedback)")
c1, c2 = st.columns(2)
with c1:
    st.write("**By Spend Category**")
    st.dataframe(cohort_by_category)
with c2:
    st.write("**By Payment Type**")
    st.dataframe(cohort_by_payment)

st.write("**Top 20 Users by Predicted Score (Before Feedback)**")
st.dataframe(df_model.sort_values("score_before", ascending=False)[["user_id","spend_category","payment_type","top_3_categories","score_before"]].head(20))

# ----------------------
# Feedback Loop Simulation (Agent 2)
# ----------------------
st.header("Agent 2: Feedback Loop (Retraining with Outcomes)")
st.markdown("We simulate that high-score users saw the pitch and a subset converted. Retrain the model with these updated outcomes.")

threshold = st.slider("Assume users above this score converted in the next cycle", min_value=0.50, max_value=0.95, value=0.60, step=0.01)

df_feedback = df_model.copy()
df_feedback["applied_after"] = df_feedback["applied"].copy()
df_feedback.loc[df_feedback["score_before"] >= threshold, "applied_after"] = 1

# Retrain
X_fb = df_feedback[feature_cols]
y_fb = df_feedback["applied_after"]
X_train_fb, X_test_fb, y_train_fb, y_test_fb = train_test_split(X_fb, y_fb, test_size=0.3, random_state=42, stratify=y_fb)

model_fb = GradientBoostingClassifier(random_state=42)
model_fb.fit(X_train_fb, y_train_fb)

y_prob_fb = model_fb.predict_proba(X_test_fb)[:,1]
auc_roc_fb = roc_auc_score(y_test_fb, y_prob_fb)
prec_fb, rec_fb, thr_fb = precision_recall_curve(y_test_fb, y_prob_fb)
auc_pr_fb = auc(rec_fb, prec_fb)

col1, col2 = st.columns(2)
with col1:
    st.metric("AUC-ROC After Feedback", f"{auc_roc_fb:.2f}", delta=f"{auc_roc_fb-auc_roc:.2f}")
with col2:
    st.metric("AUC-PR After Feedback", f"{auc_pr_fb:.2f}", delta=f"{auc_pr_fb-auc_pr:.2f}")

df_feedback["score_after"] = model_fb.predict_proba(df_feedback[feature_cols])[:,1]

# Cohort deltas
delta_by_category = (df_feedback.groupby("spend_category")[["score_before","score_after"]].mean()
                     .assign(delta=lambda x: x["score_after"] - x["score_before"])
                     .sort_values("delta", ascending=False))
st.subheader("Cohort Movement After Feedback (Avg Score Δ)")
st.dataframe(delta_by_category)

st.write("**Top 20 Users by Predicted Score (After Feedback)**")
st.dataframe(df_feedback.sort_values("score_after", ascending=False)[["user_id","spend_category","payment_type","top_3_categories","score_after"]].head(20))

# ----------------------
# Exports
# ----------------------
st.subheader("Export Scored Data")
csv_buf = StringIO()
export_cols = ["user_id","spend_category","payment_type","top_3_categories","score_before","score_after","applied","applied_after"]
df_export = df_feedback[export_cols].copy()
df_export.to_csv(csv_buf, index=False)
st.download_button("Download Scored Cohorts CSV", data=csv_buf.getvalue(), file_name="scored_cohorts.csv", mime="text/csv")
