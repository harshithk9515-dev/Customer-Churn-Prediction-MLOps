# Customer Churn MLOps Pipeline - CLI Execution Guide

## Directory Structure
```
Customer-Churn-Prediction-MLOps/
├── .github/
│   └── workflows/
│       └── ml.yml
├── data/                       # Directory where generated data resides
├── artifacts/                  # Directory where model and metrics are saved
├── tests/
│   └── test_pipeline.py
├── .gitignore
├── pytest.ini
├── requirements.txt
├── generate_data.py
├── validation.py
├── train.py
└── api.py
```

## 1. Environment Setup

### PowerShell
```powershell
cd "C:\vs code proj\mlops\Customer-Churn-Prediction-MLOps"
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Bash
```bash
cd /path/to/Customer-Churn-Prediction-MLOps
python -m pip install --upgrade pip
pip3 install -r requirements.txt
```

**Pinned dependencies** (from `requirements.txt`):
- numpy==1.26.4
- pandas==2.2.3
- scikit-learn==1.5.2
- joblib==1.4.2
- mlflow==2.17.2
- pytest==8.3.3
- dvc==3.56.0
- fastapi==0.115.4
- uvicorn==0.32.0

---

## 2. Run Pipeline Step-by-Step

### Generate Synthetic Data
```powershell
python generate_data.py
```
```bash
python3 generate_data.py
```
Output: Generates 2000 rows with SEED=42, saves to `data/customers.csv`, prints summary stats.

### Validate Data
```powershell
python validation.py
```
```bash
python3 validation.py
```
Output: Validates `data/customers.csv` against schema, raises `ValueError` if checks fail.

### Train Model
```powershell
python train.py --check-quality-gate
```
```bash
python3 train.py --check-quality-gate
```
Output: 
- Trains LogisticRegression pipeline with ColumnTransformer (StandardScaler + OneHotEncoder)
- Logs experiment "Customer Churn Prediction" to MLflow
- Logs parameters, accuracy, and ROC-AUC metrics
- Saves model to `artifacts/model.joblib`
- Saves metrics to `artifacts/metrics.json`
- Throws `RuntimeError` if ROC-AUC < 0.60

### Run pytest
```powershell
python -m pytest -v tests/test_pipeline.py
```
```bash
python3 -m pytest -v tests/test_pipeline.py
```
Output: Runs 8 unit/integration tests (all pass).

---

## 3. Launch MLflow UI
```powershell
mlflow ui --host 127.0.0.1 --port 5000
```
Access: http://127.0.0.1:5000

MLflow tracks runs locally in the `mlruns/` directory (excluded by `.gitignore`).

---

## 4. Configure Local DVC Tracking
```powershell
dvc init
dvc remote add -d myremote ./dvc-data
```
Or use existing MLflow tracking:
```powershell
dvc remote add -d myremote http://127.0.0.1:5000/api/v2/mlflow/repo
```

Data managed by DVC (`data/customers.csv`) is NOT excluded by `.gitignore` (critical for DVC to work).

---

## 5. Run FastAPI Server
```powershell
python api.py
```
Access: http://127.0.0.1:8000

### Endpoints:
- `GET /` -> `{"message": "Customer Churn Prediction MLOps API", "status": "running"}`
- `GET /health` -> `{"status": "healthy", "model_loaded": true}` (503 if model not loaded)
- `POST /predict` -> `{"churn_prediction": 0/1, "churn_probability": float, "churn_risk": "High/Low"}`

**Example POST request:**
```powershell
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"tenure": 30, "monthly_charges": 75.5, "support_calls": 2, "contract": "One Year", "internet_service": "Fiber"}'
```

### Bash equivalent:
```bash
uvicorn api:app --host 127.0.0.1 --port 8000
```

---

## 6. CI/CD Workflow
The `.github/workflows/ml.yml` runs on `push` and `pull_request` on `ubuntu-latest` with Python 3.11:
- Checkout -> Setup Python -> Install dependencies -> generate_data.py -> validation.py -> train.py --check-quality-gate -> pytest -v

---

## 7. Quick Test Commands

### Verify data validity
```powershell
python -c "from validation import validate_data; validate_data('data/customers.csv'); print('Validation passed')"
```

### Verify model artifacts
```powershell
python -c "import os; print('Model exists:', os.path.exists('artifacts/model.joblib')); print('Metrics exist:', os.path.exists('artifacts/metrics.json'))"
```

### Verify prediction
```powershell
python -c "
import joblib, pandas as pd
clf = joblib.load('artifacts/model.joblib')
df = pd.read_csv('data/customers.csv', keep_default_na=False)
X = df.drop('churn', axis=1)
proba = clf.predict_proba(X)[:, 1]
print(f'ROC-AUC (train): {round(roc_auc_score(df[\"churn\"].values, clf.predict_proba(X)[:,1]), 4)}')
"
```