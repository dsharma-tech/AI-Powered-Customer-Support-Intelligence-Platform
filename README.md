# AI-Powered Customer Support Intelligence Platform

Welcome to the **Customer Support Intelligence Platform**, an end-to-end analytical and machine learning application that decodes customer support tickets, extracts insights using Natural Language Processing (NLP), explains model predictions using Explainable AI (SHAP), and leverages Google Gemini to serve as an interactive AI Support Consultant.

This platform features a beautiful, glassmorphic dark-themed Streamlit dashboard with interactive Plotly visualizations and a real-time CSAT prediction tool.

For a deep dive into the technical details, mathematical formulas, and system design architecture, see the [PROJECT_DESCRIPTION.md](file:///Users/dishasharma/Documents/2026/Data%20Analytics/Projects/Customer%20Support%20Intelligence%20Platform/PROJECT_DESCRIPTION.md) file.

---

## 📂 Project Directory Structure

```
Customer Support Intelligence Platform/
├── PROJECT_DESCRIPTION.md           # Detailed technical concepts & problem description
├── README.md                        # Setup and run instructions (this file)
└── customer_support_intelligence/   # Main application source directory
    ├── requirements.txt             # Project Python dependencies
    ├── README.md                    # Setup and run instructions copy
    ├── data/
    │   ├── raw/
    │   │   └── customer_support_tickets.csv
    │   └── processed/
    │       └── customer_support_tickets_processed.csv
    ├── notebooks/
    │   └── modeling.ipynb           # Data preprocessing and ML model training pipeline
    ├── models/
    │   ├── best_model.pkl           # Trained Random Forest/XGBoost Regressor
    │   ├── preprocessor.pkl         # Fitted categorical/numerical encoder pipeline
    │   └── shap_summary.png         # Exported explainability SHAP plot
    └── app/
        ├── app.py                   # Streamlit multi-tab application file
        └── styles.css               # Premium CSS layout & styling definitions
```

---

## ⚡ Key Features
https://www.kaggle.com/datasets/suraj520/customer-support-ticket-dataset?resource=download
1. **Clean Data Preprocessing & Feature Extraction:**
   - Cleans ticket datasets and computes duration features (First Response Time, Resolution Duration).
2. **AI-Driven NLP (Sentiment & Topic Clustering):**
   - Integrates **NLTK VADER** to analyze customer sentiments, categorizing text as Positive, Neutral, or Negative.
   - Performs topic modeling to classify issues into core categories (Product Setup, Billing, Account Access, Hardware, Software).
3. **Explainable Machine Learning Modeling:**
   - Fits **Random Forest Regressor** and **XGBoost Regressor** to predict Customer Satisfaction Ratings.
   - Applies **SHAP** values to identify global features driving customer satisfaction ratings.
4. **Google Gemini AI Support Consultant:**
   - Features five distinct sub-tools: a ticket summarizer, a troubleshooting steps planner, an insights generator, an operations recommendations list, and a semantic metrics Q&A search.
5. **Interactive Glassmorphic Dashboard:**
   - Multi-tab dashboard with interactive filters, custom dark-mode styling, hover transitions, and Plotly charts.

---

## 🛠️ Installation & Setup Guide

Execute all commands from the root directory of the project (`/Users/dishasharma/Documents/2026/Data Analytics/Projects/Customer Support Intelligence Platform`).

### 1. Initialize Virtual Environment
Set up a clean Python virtual environment to manage dependencies:
```bash
# Create the virtual environment
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate
```

### 2. Install Project Dependencies
Install all required libraries, including plotting engines, machine learning estimators, and the Google GenAI SDK:
```bash
pip install --upgrade pip
pip install -r customer_support_intelligence/requirements.txt
```

### 3. Set Up API Key (Optional but Recommended)
To enable the **AI Support Consultant** features powered by Gemini, export your Google Gemini API Key as an environment variable before launching the dashboard:
```bash
export GEMINI_API_KEY="your-api-key-here"
```
> [!NOTE]
> If this environment variable is not set, the application will fallback to a default API key. You can also paste your Gemini API Key directly into the secure input field in the sidebar of the running dashboard.

### 4. Run Preprocessing & Modeling Pipeline
Ensure that the processed data and machine learning artifacts (`best_model.pkl`, `preprocessor.pkl`, and `shap_summary.png`) are generated before starting the dashboard.

Choose one of the two methods below to run the pipeline:

* **Method A: Interactive execution (Recommended if using IDEs)**
  Open the notebook file `customer_support_intelligence/notebooks/modeling.ipynb` in VS Code or Jupyter Lab, select your virtual environment kernel (`venv`), and click **Run All**.
  
* **Method B: Programmatic execution via terminal**
  Execute the notebook headlessly directly from your terminal:
  ```bash
  pip install jupyter
  jupyter nbconvert --to notebook --execute --inplace customer_support_intelligence/notebooks/modeling.ipynb
  ```

### 5. Launch the Streamlit Dashboard
Start the real-time AI-Powered Customer Support Intelligence Platform locally:
```bash
streamlit run customer_support_intelligence/app/app.py
```
Once launched, the dashboard will automatically open in a new tab in your default browser (usually at `http://localhost:8501`).

---

## 🖥️ Using the Dashboard

The dashboard is structured into three main tabs:
1. **Executive Metrics**: Operational distributions (tickets by status, channel, priority), product lines performance (CSAT vs. resolution duration), and NLTK VADER sentiment analysis.
2. **AI Support Consultant**: Access Gemini-powered diagnostics, summaries, custom recommendations checklists, and text-based metrics analysis.
3. **CSAT Prediction Tool**: Input customer parameters (e.g. ticket priority, sentiment score, product line, channel) and get a real-time forecasted CSAT rating from the trained machine learning models, along with SHAP explainability insights.

---

## ❓ Troubleshooting

* **Error: `Processed dataset not found! Please run the modeling notebook first.`**
  * *Solution*: Make sure you run the preprocessing and modeling notebook as explained in Step 4. Check that `customer_support_tickets_processed.csv` exists under `customer_support_intelligence/data/processed/`.
* **Error: `Port 8501 already in use`**
  * *Solution*: Streamlit is already running on that port. You can run on a different port by appending the port flag: `streamlit run customer_support_intelligence/app/app.py --server.port 8502`.
* **Error: `No API key provided`**
  * *Solution*: Supply your API key in the terminal with `export GEMINI_API_KEY="key"`, or enter it directly in the text box inside the sidebar of the dashboard.
