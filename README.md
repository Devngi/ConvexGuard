# ConvexGuard

Real-time credit card fraud detection using a **from-scratch Support Vector Machine (SVM)**. Inference runs entirely inside the client browser via pure mathematical formulas — **no backend server, no cloud API, zero network latency**.

---

## 📊 Dataset

The model is trained on the benchmark **Credit Card Fraud Detection** dataset:
- **Source**: [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Overview**: 284,807 transactions made by European cardholders in September 2013, with 492 fraud cases (0.172% extreme class imbalance).
- **Features**:
  - `V1` to `V28`: Anonymized principal components (PCA) protecting sensitive cardholder identity and location.
  - `Time`: Elapsed seconds since the first recorded transaction.
  - `Amount`: Transaction amount in USD.
  - `Class`: Ground-truth label (`1` for fraud, `0` for legitimate).
- **Engineered Features**:
  - `LogAmount`: $\ln(1 + \text{Amount})$ to stabilize the right-skewed amount distribution.
  - `Hour`: $\lfloor \text{Time} / 3600 \rfloor \pmod{24}$ capturing diurnal fraud patterns.

> **Note**: `data/creditcard.csv` (~150 MB) exceeds GitHub's 100 MB file size limit and is excluded via `.gitignore`. You must download it separately from Kaggle before training.

---

## 🚀 Running Steps

### 1. Clone the Repository
```bash
git clone https://github.com/Devngi/ConvexGuard.git
cd ConvexGuard
```

### 2. Install Dependencies
Ensure you have Python 3.10+ installed:
```bash
pip install -r train/requirements.txt
```

### 3. Place the Dataset
1. Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).
2. Place the file into the `data/` folder:
   ```text
   ConvexGuard/
   └── data/
       └── creditcard.csv
   ```

### 4. Train the Model & Export Web Assets
Run the end-to-end training and export script:
```bash
python3 train/train_and_export.py
```
This script will:
- Deduplicate and stratify the dataset into train, validation, and test splits.
- Fit a standard scaler on the 30 features ($V_1$–$V_{28}$, $\text{LogAmount}$, $\text{Hour}$).
- Train the weighted **Primal Subgradient SVM** ($C_{\text{ratio}} = 10$).
- Select the optimal decision threshold optimizing $F_2$-score (prioritizing recall).
- Verify optimization optimality via **KKT complementary slackness** and **duality gap**.
- Export everything into `site/data.js` (~940 KB self-contained JavaScript payload).

### 5. Launch the Website

You can run the web app in either of two ways:

#### Option A: Local HTTP Server (Recommended)
```bash
python3 -m http.server 8080 --directory site
```
Then open your browser and navigate to:
```
http://localhost:8080
```

#### Option B: Direct Browser Open
Because `site/data.js` assigns data directly to `window.DATA` without making dynamic `fetch()` requests, you can open the file directly in any modern browser without a web server:
```bash
open site/index.html
```

---

## 🖥️ Website Features

The web interface is organized into 5 interactive tabs:

1. **Check a Transaction**:
   - Interactive transaction scoring form with real-time feedback.
   - Adjust key signals (Amount, Hour, $V_4$, $V_9$, $V_{10}$, $V_{12}$, $V_{14}$, $V_{17}$) or expand all 30 features.
   - One-click buttons to load confirmed **real FRAUD** and **real LEGIT** test cases.
   - Real-time score indicator, threshold comparison, decision axis, and top-8 feature contribution breakdown ($w_j \cdot z_j$).
2. **Dataset & Features**:
   - Detailed schema documentation for source PCA components and engineered features.
3. **Exploratory Data Analysis (EDA)**:
   - Interactive distributions of transaction amounts, fraud occurrence rates by hour of day, and feature correlation heatmaps.
4. **Model Architecture & Convex Optimization**:
   - Full mathematical derivation rendered with KaTeX.
   - Primal objective, subgradient updates, and soft-margin loss formulation.
   - Live numerical validation of **KKT complementary slackness** ($\max \text{violation} < 0.01$) and **duality gap** bounds.
5. **Results & Benchmarks**:
   - Interactive Precision-Recall curve with an interactive decision threshold slider.
   - Comparative benchmark table comparing ConvexGuard's SVM against `sklearn.svm.LinearSVC` and Logistic Regression.

---

## 📁 Repository Structure

```text
ConvexGuard/
├── data/
│   └── creditcard.csv               # Kaggle dataset (gitignored)
├── site/                            # Standalone frontend application
│   ├── index.html                   # Semantic HTML5 layout
│   ├── style.css                    # Responsive modern styling
│   ├── app.js                       # Browser inference & Chart.js logic
│   ├── data.js                      # Trained model weights & precomputed metrics
│   └── lib/                         # Bundled KaTeX and Chart.js libraries
├── src/
│   └── convexguard/                 # Core Python package
│       ├── data.py                  # Dataset loading & deduplication
│       ├── preprocess.py            # Feature engineering & scaling
│       ├── solvers/
│       │   └── primal_subgradient.py # NumPy SVM primal subgradient solver
│       ├── duality.py               # Primal & dual objective computations
│       ├── kkt.py                   # KKT violation check functions
│       ├── threshold.py             # F-beta threshold optimization
│       └── metrics.py               # Precision, Recall, PR-AUC evaluation
├── train/
│   ├── requirements.txt             # Python dependencies
│   └── train_and_export.py          # Unified training, evaluation & export script
└── tests/                           # Unit tests for solvers, KKT, and duality
```

---

## 🧪 Testing

To run the full verification test suite:
```bash
PYTHONPATH=src pytest
```

