# Vendor Invoice Intelligence System

## 🛄 Freight Cost Prediction & Invoice Risk Flagging

This project implements an **end-to-end machine learning system** designed to support Finance teams by:
* **Predicting expected freight costs** for vendor invoices.
* **Flagging high-risk invoices** that require manual review due to abnormal cost, freight, or operational patterns.

---

## 📖 Table of Contents
1. [Project Overview](#project-overview)
2. [Business Objectives](#business-objectives)
3. [Data Sources](#data-sources)
4. [Exploratory Data Analysis](#exploratory-data-analysis)
5. [Models Used](#models-used)
6. [Evaluation Metrics](#evaluation-metrics)
7. [Project Structure](#project-structure)
8. [Application](#application)
9. [How to Run This Project](#how-to-run-this-project)
10. [Author & Contact](#author--contact)

---

## <a class="anchor" id="project-overview"></a>📌 Project Overview
Manual verification of logistics invoices is time-consuming and prone to errors. This repository provides an automated machine learning workflow that analyzes historical invoice data, estimates fair shipping prices, and detects operational anomalies. By isolating standard low-risk transactions, finance teams can focus their attention solely on high-variance discrepancies.

---

## <a class="anchor" id="business-objectives"></a>💼 Business Objectives
* **Optimize Logistics Spends:** Provide real-time estimates of shipping and freight costs to identify overcharges.
* **Automate Invoice Auditing:** Automate the approval pipeline by systematically separating low-risk and high-risk invoices.
* **Reduce Financial Leakage:** Prevent erroneous payments and manual data-entry mismatch leakages.

---

## <a class="anchor" id="data-sources"></a>📊 Data Sources
Describe the datasets used in this project. For example:
* `invoices.csv`: Historical vendor bills containing line-item costs, vendor details, and dates.
* `shipment_logs.csv`: Operational logistics data including shipping distances, weight, origin, and destination codes.

---

## <a class="anchor" id="eda"></a>🔍 Exploratory Data Analysis (EDA)
Key insights uncovered during the data exploration phase:
* Distribution analysis of freight costs across different shipping zones.
* Identification of missing values, anomalies, and structural outliers in vendor formatting.
* Correlation analysis between package volume/weight and final shipping fees.

---

## <a class="anchor" id="models-used"></a>🤖 Models Used
The system leverages two distinct modeling approaches:
1. **Regression Models (Cost Prediction):** Used to predict continuous freight costs (e.g., XGBoost, LightGBM, or Linear Regression).
2. **Classification/Anomaly Models (Risk Flagging):** Used to identify high-risk outliers (e.g., Isolation Forests, Random Forests, or threshold-based heuristic flagging).

---

## <a class="anchor" id="metrics"></a>📐 Evaluation Metrics
The models are evaluated using the following performance indicators:
* **Regression:** Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) to gauge cost prediction accuracy.
* **Classification:** Precision, Recall, and F1-Score to ensure high detection rates of actual billing errors while keeping false alarms low.

---

## <a class="anchor" id="project-structure"></a>📂 Project Structure
```text
├── data/                  # Raw and processed datasets
├── notebooks/             # Jupyter notebooks for EDA and model prototyping
├── src/                   # Production-ready Python source code
│   ├── data_prep.py       # Data cleaning and feature engineering
│   ├── train.py           # Model training scripts
│   └── inference.py       # Prediction and risk flagging logic
├── app/                   # Application or deployment configuration files
├── requirements.txt       # Project dependencies
└── README.md              # Project documentation
```

---

## <a class="anchor" id="application"></a>🚀 Application
Detail how the final system is served or delivered. (e.g., "The model is deployed as a Streamlit web dashboard where users can upload invoice PDFs/CSVs to instantly receive risk assessments and flag indicators.")

---

## <a class="anchor" id="how-to-run-this-project"></a>💻 How to Run This Project

### 1. Clone the repository
```bash
git clone https://github.com
cd vendor-invoice-intelligence
```

### 2. Set up a virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Execute the pipeline
```bash
python src/train.py
```

---

## <a class="anchor" id="author"></a>✍️ Author & Contact
* **Name:** Harsh Khelani
* **Email:** harshkhelani2005@gmail.com
* **GitHub:** [harsh-49maker](https://github.com)
* **LinkedIn:** [harsh khelani](https://linkedin.com)