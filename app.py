import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# ==============================================================================
# Page Configuration
# ==============================================================================
st.set_page_config(
    page_title="Vendor Invoice Intelligence Portal",
    page_icon="🤖",
    layout="wide"
)

# ==============================================================================
# Header Section
# ==============================================================================
st.markdown("""
# 🛡️ Vendor Invoice Intelligence Portal
### AI-Driven Freight Cost Prediction & Invoice Risk Flagging

This internal analytics portal leverages machine learning to
- **Forecast freight costs accurately**
- **Detect risky or abnormal vendor invoices**
- **Reduce financial leakage and manual workload**
""")

st.divider()

# ==============================================================================
# Sidebar
# ==============================================================================
st.sidebar.title("🔍 Model Selection")
selected_model = st.sidebar.radio(
    "Choose Prediction Module",
    [
        "Freight Cost Prediction",
        "Invoice Manual Approval Flag"
    ]
)

st.sidebar.markdown("""
---
**Business Impact**
- 📉 Improved cost forecasting
- 🛡️ Reduced invoice fraud & anomalies
- ⚙️ Faster finance operations
""")

# ==============================================================================
# Module 1: Freight Cost Prediction
# ==============================================================================
if selected_model == "Freight Cost Prediction":
    st.subheader("🚚 Freight Cost Prediction")

    st.markdown("""
    **Objective:**
    Predict freight cost for a vendor invoice using **Quantity** and **Invoice Dollars**
    to support budgeting, forecasting, and vendor negotiations.
    """)

    with st.form("freight_form"):
        coll, col2 = st.columns(2)

        with coll:
            quantity = st.number_input("📦 Quantity", min_value=1, value=1200)

        with col2:
            dollars = st.number_input("💰 Invoice Dollars", min_value=1.0, value=18500.0)

        submit_freight = st.form_submit_button("🤖 Predict Freight Cost")

    if submit_freight:
        # Load local data file and train regression instantly
        try:
            df = pd.read_csv("vendor_invoice.csv")
            reg_model = LinearRegression()
            reg_model.fit(df[['Dollars']], df['Freight'])

            sample_data = pd.DataFrame({"Dollars": [dollars]})
            prediction = reg_model.predict(sample_data)[0]

            st.success("Prediction completed successfully.")
            st.metric(label="📊 Estimated Freight Cost", value=f"${prediction:,.2f}")
        except Exception as e:
            st.error(f"Error loading vendor_invoice.csv: {e}")

# ==============================================================================
# Module 2: Invoice Flag Prediction
# ==============================================================================
else:
    st.subheader("📥 Invoice Manual Approval Prediction")

    st.markdown("""
    **Objective:**
    Predict whether a vendor invoice should be **flagged for manual approval**
    based on abnormal cost, freight, or delivery patterns.
    """)

    with st.form("invoice_flag_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            invoice_quantity = st.number_input("Invoice Quantity", min_value=1, value=50)
            freight = st.number_input("Freight Cost", min_value=0.0, value=1.73)

        with col2:
            invoice_dollars = st.number_input("Invoice Dollars", min_value=1.0, value=352.95)
            total_item_quantity = st.number_input("Total Item Quantity", min_value=1, value=162)

        with col3:
            total_item_dollars = st.number_input("Total Item Dollars", min_value=1.0, value=2476.0)

        submit_flag = st.form_submit_button("🧠 Evaluate Invoice Risk")

    if submit_flag:
        try:
            # Rebuild features layout matching data frame
            df = pd.read_csv("vendor_invoice.csv")
            df['PODate'] = pd.to_datetime(df['PODate'])
            df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
            df['PayDate'] = pd.to_datetime(df['PayDate'])

            df['days_po_to_invoice'] = (df['InvoiceDate'] - df['PODate']).dt.days
            df['days_to_pay'] = (df['PayDate'] - df['InvoiceDate']).dt.days
            df['receiving_delay'] = (df['PayDate'] - df['PODate']).dt.days

            pa = df.groupby('PONumber').agg(
                total_brands=('VendorName', 'nunique'),
                total_item_quantity=('Quantity', 'sum'),
                total_item_dollars=('Dollars', 'sum'),
                avg_receiving_delay=('receiving_delay', 'mean')
            ).reset_index()

            vi = df[['PONumber', 'Quantity', 'Dollars', 'Freight', 'days_po_to_invoice', 'days_to_pay']].copy()
            vi = vi.rename(columns={'Quantity': 'invoice_quantity', 'Dollars': 'invoice_dollars'})
            final_flagging_df = pd.merge(vi, pa, on='PONumber', how='left')

            median_delay = final_flagging_df["avg_receiving_delay"].median()
            final_flagging_df["flag_invoice"] = final_flagging_df.apply(
                lambda r: 1 if (abs(r["invoice_dollars"] - r["total_item_dollars"]) > 5 or r["avg_receiving_delay"] > median_delay) else 0, 
                axis=1
            )

            features = ['invoice_quantity', 'invoice_dollars', 'Freight', 'total_brands', 'total_item_quantity', 'days_po_to_invoice', 'total_item_dollars']
            X = final_flagging_df[features]
            y = final_flagging_df['flag_invoice']

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            clf = RandomForestClassifier(max_depth=6, n_estimators=50, random_state=42, n_jobs=-1)
            clf.fit(X_scaled, y)

            # Process individual inputs
            input_data = pd.DataFrame([{
                "invoice_quantity": invoice_quantity,
                "invoice_dollars": invoice_dollars,
                "Freight": freight,
                "total_brands": 1, 
                "total_item_quantity": total_item_quantity,
                "days_po_to_invoice": 10,
                "total_item_dollars": total_item_dollars
            }])

            input_scaled = scaler.transform(input_data[features])
            flag_prediction = clf.predict(input_scaled)[0]

            if bool(flag_prediction):
                st.error("📥 Invoice requires **MANUAL APPROVAL**")
            else:
                st.success("✅ Invoice is **SAFE for Auto-Approval**")
        except Exception as e:
            st.error(f"Error evaluating metrics: {e}")
