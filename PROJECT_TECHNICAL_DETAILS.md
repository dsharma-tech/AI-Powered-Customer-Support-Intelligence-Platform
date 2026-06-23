# Technical Reference Manual: AI-Powered Customer Support Intelligence Platform

This document serves as the single, consolidated technical reference file for the **Customer Support Intelligence Platform**. It outlines the project's data schema, mathematical equations, machine learning architecture, explainable AI formulations, generative AI configuration, front-end visual specifications, and complete setup and execution commands.

---

## 1. Directory Structure & File Mapping

The following structure illustrates the layout of the project workspace:

```
Customer Support Intelligence Platform/
├── README.md                           # Main installation and setup guide
├── PROJECT_DESCRIPTION.md              # High-level business overview
├── PROJECT_TECHNICAL_DETAILS.md        # Consolidated technical reference (this file)
└── customer_support_intelligence/      # Source directory
    ├── requirements.txt                # Python libraries and dependencies
    ├── data/
    │   ├── raw/
    │   │   └── customer_support_tickets.csv              # Source tickets dataset
    │   └── processed/
    │       └── customer_support_tickets_processed.csv    # Cleansed and engineered dataset
    ├── notebooks/
    │   └── modeling.ipynb              # Preprocessing, training, and SHAP pipeline
    ├── models/
    │   ├── best_model.pkl              # Serialized trained regressor model
    │   ├── preprocessor.pkl            # Serialized sklearn preprocessing pipeline
    │   └── shap_summary.png            # Saved feature importance summary chart
    └── app/
        ├── app.py                      # Multi-tab Streamlit dashboard
        └── styles.css                  # Custom CSS styles (glassmorphic theme)
```

---

## 2. Data Engineering & Preprocessing Pipeline

### A. Raw Data Schema
The source dataset (`customer_support_tickets.csv`) contains support logs with the following columns:
*   `Ticket ID`: Unique ticket identifier.
*   `Customer Name`, `Customer Email`: Demographics (omitted during model training to preserve privacy).
*   `Customer Age`: Continuous numeric value.
*   `Customer Gender`: Categorical value (`Male`, `Female`, `Other`).
*   `Product Purchased`: Categorical value representing the company's product line.
*   `Date of Purchase`: Raw string representations of dates.
*   `Ticket Type`: Categorical classification of the issue (e.g., `Technical Issue`, `Billing Inquiry`).
*   `Ticket Subject`: Short text description.
*   `Ticket Description`: Long unstructured textual complaint.
*   `Ticket Status`: Categorical value (`Open`, `Pending`, `Closed`).
*   `Ticket Priority`: Categorical value (`Low`, `Medium`, `High`, `Critical`).
*   `Ticket Channel`: Categorical ingestion channel (`Email`, `Chat`, `Phone`, `Web`).
*   `First Response Time`: Text representing timestamp or duration of first response.
*   `Time to Resolution`: Text representing timestamp or duration of ticket resolution.

### B. Preprocessing & Feature Engineering Operations
The preprocessing pipeline is executed inside `modeling.ipynb` and transforms raw variables into model features:

1.  **Duration Calculations**:
    *   `First Response Time` and `Time to Resolution` are parsed and converted to continuous values in hours (e.g. `First Response Time (Hours)` and `Resolution Duration (Hours)`).
2.  **Datetime Feature Engineering**:
    *   `Date of Purchase` is converted to standard timestamps to extract cyclic calendar components:
        *   `Purchase Year`, `Purchase Month`, `Purchase Quarter`, `Purchase Weekday`.
3.  **NLP Sentiment Quantification**:
    *   The `Ticket Description` text is analyzed using the **NLTK VADER** sentiment engine.
    *   Each text generates positive, negative, and neutral percentages, which are condensed into a single continuous target feature: `Sentiment Score` ranging from `-1.0` (highly negative) to `+1.0` (highly positive).
4.  **Handling Missing Data**:
    *   Missing age or duration metrics are filled with median values.
    *   Categorical missing values are filled with a standard placeholder (e.g., `"Unknown"`).

---

## 3. Supervised Machine Learning Pipeline

### A. Feature Transformations
To prepare the dataset for scikit-learn and XGBoost, a custom preprocessing pipeline (`preprocessor.pkl`) is defined:
*   **Numerical Features**: `Customer Age`, `First Response Time (Hours)`, `Resolution Duration (Hours)`, `Sentiment Score`.
    *   *Transformation*: Standardized using a `StandardScaler` to have a mean of `0.0` and a standard deviation of `1.0`:
        $$z = \frac{x - \mu}{\sigma}$$
*   **Categorical Features**: `Customer Gender`, `Product Purchased`, `Ticket Type`, `Ticket Subject`, `Ticket Priority`, `Ticket Channel`, `Ticket Topic Label`.
    *   *Transformation*: Encoded using `OneHotEncoder(handle_unknown='ignore')` to convert categories into binary sparse vectors.

### B. Models & Optimizers
The project fits two core regressors to predict the continuous customer satisfaction target (CSAT rating, scaled `1.0` to `5.0` stars):

#### 1. Random Forest Regressor
*   **Methodology**: Bagging ensemble algorithm. Constructs a multitude of uncorrelated decision trees during training.
*   **Objective**: Reduces variance via bootstrap aggregating (bagging) and random feature selection, yielding high robustness against noise.
*   **Hyperparameters**: Fitted with `n_estimators=100`, `max_depth=15`, and `random_state=42`.

#### 2. XGBoost Regressor (Extreme Gradient Boosting)
*   **Methodology**: Boosting ensemble algorithm. Fits trees sequentially, with each new tree minimizing the residuals of the preceding model.
*   **Objective**: Optimizes a regularized objective function:
    $$\mathcal{L}^{(t)} = \sum_{i=1}^n l(y_i, \hat{y}_i^{(t-1)} + f_t(x_i)) + \Omega(f_t)$$
    where $\Omega(f_t) = \gamma T + \frac{1}{2}\lambda \sum_{j=1}^T w_j^2$ acts as a penalty for tree complexity (number of leaves $T$ and weights $w$).
*   **Hyperparameters**: Fitted with `n_estimators=100`, `learning_rate=0.05`, `max_depth=6`, and `random_state=42`.

### C. Model Evaluation Metrics
Models are validated on a 20% holdout split using:
*   **Mean Absolute Error (MAE)**: Measures the average magnitude of prediction errors:
    $$\text{MAE} = \frac{1}{n} \sum_{i=1}^n |y_i - \hat{y}_i|$$
*   **Root Mean Squared Error (RMSE)**: Penalizes larger errors heavily:
    $$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2}$$
*   **Coefficient of Determination ($R^2$)**: Represents the proportion of variance explained by features:
    $$R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$$

---

## 4. Explainable AI (SHAP) Formulations

The platform integrates **SHAP (Shapley Additive exPlanations)** values to establish global and local feature transparency.

### A. Mathematical Theory
SHAP values represent the additive feature contributions to a model's prediction. The value $\phi_i$ assigned to feature $i$ is calculated as:
$$\phi_i(x) = \sum_{S \subseteq F \setminus \{i\}} \frac{|S|!(|F| - |S| - 1)!}{|F|!} \left[ f(S \cup \{i\}) - f(S) \right]$$
Where:
*   $F$ is the complete set of features.
*   $S$ is a subset of features excluding feature $i$.
*   $f(S)$ is the model's predicted outcome when only features in $S$ are present.
*   $|S|! (|F| - |S| - 1)! / |F|!$ represents the probability that subset $S$ is selected.

### B. Artifact Generation
*   A `shap.TreeExplainer` is initialized on the trained Random Forest model.
*   SHAP values are computed across all test records.
*   A global SHAP summary plot (`shap_summary.png`) is exported. This plot aligns features vertically by importance and colors individual points based on feature values (red indicating high values, blue indicating low values) to show their directional impact on satisfaction.

---

## 5. Generative AI (Gemini) Integration

The platform uses **Google Gemini 2.5 Flash** via the official `google-genai` SDK to run context-aware operations analysis.

### A. In-Memory Context Serialization
Instead of sending large ticket databases to the LLM API, the app computes highly condensed JSON summary strings and injects them directly into the LLM system prompt. 

Example payload format:
```json
{
  "total_records": 8450,
  "global_averages": {
    "csat": 3.42,
    "resolution_hours": 8.7
  },
  "channel_metrics": [
    {"channel": "Email", "avg_csat": 2.9, "avg_resolution": 14.2},
    {"channel": "Chat", "avg_csat": 4.1, "avg_resolution": 2.5}
  ],
  "product_friction": [
    {"product": "Smart Thermostat", "tickets": 1205, "avg_csat": 2.4}
  ]
}
```

### B. Gemini Prompts & Sub-Tools

1.  **AI Recommendations Generator**:
    *   *System Role*: Customer Experience Consultant.
    *   *Task*: Review the serialized context. Output five immediate operational recommendations including staffing adjustments, channels needing SLA checks, and product friction resolutions.
2.  **AI Ticket Summarizer & Classifier**:
    *   *Task*: Analyze a single raw unstructured ticket description.
    *   *Output*: Return a JSON or markdown block indicating Issue Category, Severity (Low, Medium, High, Critical), Ticket Summary, Recommended Support Team, and suggested Resolution Steps.
3.  **Insights Generator**:
    *   *Task*: Create short observations from dashboard graphs (e.g. comparing CSAT scores across Response Time and Resolution Time buckets).
4.  **Semantic Metrics Q&A**:
    *   *Task*: A conversational chat interface. Gemini uses the serialized metrics JSON context to answer natural language user queries like *"Which product line should we fix first and why?"*

---

## 6. Streamlit Visual System & Custom CSS

The front-end is styled using a custom **glassmorphic dark SaaS theme** (`styles.css`):

### A. Card Containers
```css
.chart-card {
    background: #09162F;
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 28px;
    padding: 24px;
}
```
*   **Double Borders Overrides**: Direct target rules in Streamlit prevent double-border nesting inside vertical columns:
    ```css
    div[data-testid="stVerticalBlock"]:has(> .element-container .chart-card-marker) {
        background: #09162F;
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 28px;
        padding: 24px;
    }
    ```

### B. Visual Layout Safeguards
*   **No Scrollbars**: Explicit heights and `overflow` rules are avoided. Charts resize responsively relative to parent divs.
*   **Donut Chart Configurations**: Sized to fill 70–75% of card space by explicitly overriding margins and domains:
    *   `hole=0.65`
    *   `domain=dict(x=[0.05, 0.95], y=[0.05, 0.95])`
    *   Horizontal legend placed at `y=-0.15` to maximize plot area.
*   **Bubble Scatter Coordinates**: The Satisfaction vs. Resolution duration bubble plot is configured with an expanded right margin to prevent overlaps:
    *   `coloraxis_colorbar=dict(x=1.02, y=0.5, len=0.65, thickness=18)`
    *   `margin=dict(l=60, r=120, t=50, b=70)`

---

## 7. Setup & Run Instructions

Execute these commands from the root directory of the workspace.

### 1. Initialize Python Environment
Ensure a clean Python environment is established:
```bash
# Initialize virtual environment
python3 -m venv venv

# Activate environment
source venv/bin/activate
```

### 2. Install Package Requirements
```bash
pip install --upgrade pip
pip install -r customer_support_intelligence/requirements.txt
```

### 3. Provide Gemini Credentials
Export your Google Gemini API key:
```bash
export GEMINI_API_KEY="your-api-key-here"
```
*(Alternatively, copy and paste the API key directly into the secure key field in the sidebar of the Streamlit dashboard).*

### 4. Execute the Data Pipeline & ML Training
To generate processed CSV files and model weights (`best_model.pkl`, `preprocessor.pkl`, `shap_summary.png`), execute the Jupyter modeling notebook:
```bash
# Install Jupyter command-line tool
pip install jupyter

# Execute the notebook headlessly and update in-place
jupyter nbconvert --to notebook --execute --inplace customer_support_intelligence/notebooks/modeling.ipynb
```

### 5. Launch the Streamlit Web Application
Run the dashboard server locally:
```bash
streamlit run customer_support_intelligence/app/app.py
```
Open `http://localhost:8501` in your browser.
