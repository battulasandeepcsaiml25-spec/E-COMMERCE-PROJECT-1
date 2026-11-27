# Online Purchase Conversion Advisor

A full-stack machine learning project that predicts whether an online shopping session will end in a purchase, and turns those predictions into actionable insights for an e-commerce business.

> **Note:** Items marked `TODO` need to be confirmed or filled in against the actual code before you publish this README.

---

## Overview

E-commerce sites lose most visitors before checkout. This project uses session-level behavioural data (pages viewed, time spent, bounce/exit rates, visitor type, seasonality, etc.) to estimate the probability that a session converts to revenue, and presents the results through an interactive web app.

- **Problem type:** Binary classification (target: `Revenue`)
- **Dataset:** Online Shoppers Purchasing Intention Dataset (UCI) — TODO: confirm
- **Models:** Logistic Regression, Decision Tree, Random Forest — TODO: confirm final model(s) used in the app

## Features

- Purchase prediction form: enter session details, get a conversion probability
- Exploratory data analysis (EDA) visualisations
- Model comparison with real evaluation metrics (accuracy, precision, recall, F1, ROC-AUC)
- Conversion advisor: suggestions based on model output
- Documentation page explaining the methodology
- REST API connecting the frontend to the trained models

## Project Structure

```
E-COMMERCE-PROJECT-1/
├── backend/        # API server that loads trained models and serves predictions
├── frontend/       # Web UI (dashboard, prediction form, charts)
├── ml_models/      # Trained/serialized models and preprocessing artifacts
├── experiments/    # Notebooks and scripts for EDA, preprocessing and training
└── .gitignore
```

## Tech Stack

| Layer | Technology (TODO: confirm) |
| --- | --- |
| Frontend | React, Vite, Tailwind CSS, React Router, Recharts |
| Backend | Python, FastAPI (or Flask) |
| ML | scikit-learn, pandas, NumPy, joblib |
| Experiments | Jupyter Notebook |

## Getting Started

### Prerequisites

- Python 3.9+
- Node.js 18+ and npm
- Git

### 1. Clone the repository

```bash
git clone https://github.com/battulasandeepcsaiml25-spec/E-COMMERCE-PROJECT-1.git
cd E-COMMERCE-PROJECT-1
```

### 2. Run the backend

```bash
cd backend
python -m venv venv
# Windows: venv\Scripts\activate
source venv/bin/activate
pip install -r requirements.txt        # TODO: confirm file exists
uvicorn main:app --reload              # TODO: adjust module name / command
```

The API should be available at `http://localhost:8000` (interactive docs at `/docs` if using FastAPI).

### 3. Run the frontend

```bash
cd frontend
npm install
npm run dev
```

The app should be available at `http://localhost:5173`.

### 4. (Optional) Retrain the models

Open the notebooks/scripts in `experiments/` to reproduce preprocessing, training and evaluation. Save the resulting model files into `ml_models/`.

## API Reference

TODO: replace with the real endpoints. Example:

| Method | Endpoint | Description |
| --- | --- | --- |
| `GET` | `/health` | Service health check |
| `POST` | `/predict` | Returns conversion prediction and probability for a session |
| `GET` | `/metrics` | Returns model evaluation metrics |

**Example request**

```json
{
  "PageValues": 12.5,
  "BounceRates": 0.02,
  "ExitRates": 0.04,
  "ProductRelated": 30,
  "VisitorType": "Returning_Visitor",
  "Month": "Nov"
}
```

## Model Details

- **Preprocessing:** missing-value handling, categorical encoding, feature scaling — TODO: confirm
- **Class imbalance:** only a minority of sessions convert; metrics such as recall, F1 and ROC-AUC are reported alongside accuracy — TODO: confirm technique used
- **Evaluation:** train/test split with metrics reported from real runs (add your actual results here)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | --- | --- | --- | --- | --- |
| Logistic Regression | TODO | TODO | TODO | TODO | TODO |
| Decision Tree | TODO | TODO | TODO | TODO | TODO |
| Random Forest | TODO | TODO | TODO | TODO | TODO |

## Screenshots

Add screenshots of the dashboard, prediction page and model performance page here.

## Roadmap

- [ ] Hyperparameter tuning and cross-validation
- [ ] Model explainability (feature importance / SHAP)
- [ ] Docker setup and cloud deployment
- [ ] Automated tests and CI

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes and push the branch
4. Open a pull request

## License

TODO: add a license (e.g. MIT) or remove this section.

## Author

**battulasandeepcsaiml25-spec** — [GitHub profile](https://github.com/battulasandeepcsaiml25-spec)
