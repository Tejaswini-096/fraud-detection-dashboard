import streamlit as st
import pandas as pd

st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")

st.title("🚨 Fraud Detection Alert Dashboard")

# Load autoencoder alert data
df = pd.read_csv('fraud_alerts.csv')

# Top metrics
col1, col2, col3 = st.columns(3)
col1.metric("Total Transactions", len(df))
col2.metric("Total Alerts Raised", int(df['alert'].sum()))
col3.metric("Actual Fraud Cases", int(df['true_class'].sum()))

st.subheader("🔍 Flagged Transactions (Alerts)")
alerts = df[df['alert'] == 1].sort_values('reconstruction_error', ascending=False)
st.dataframe(alerts)

st.subheader("📊 Reconstruction Error Distribution")
st.bar_chart(df['reconstruction_error'])

st.subheader("⚙️ Filter Alerts")
min_error = st.slider("Minimum reconstruction error", 0.0, float(df['reconstruction_error'].max()), 1.0)
filtered = df[df['reconstruction_error'] > min_error]
st.write(f"Transactions above threshold {min_error}: {len(filtered)}")
st.dataframe(filtered)

# Model comparison section
st.subheader("🏆 Model Comparison")
comparison_df = pd.read_csv('model_comparison.csv')
st.dataframe(comparison_df)
st.bar_chart(comparison_df.set_index('Model')[['Precision', 'Recall', 'F1-Score']])

st.caption("Random Forest + SMOTE performed best overall, since it's a supervised model trained on labeled, balanced data. Isolation Forest and Autoencoder are unsupervised — they detect anomalies without ever seeing fraud labels, which is more realistic for catching new/unknown fraud patterns in production.")
