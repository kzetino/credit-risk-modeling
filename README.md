# Credit Risk Modeling

A machine learning app that predicts credit risk (good or bad) based on applicant information, built with an Extra Trees classifier and a Streamlit web interface.

## Features

- Trained on the German Credit Dataset
- Uses an Extra Trees classifier (`sklearn`)
- Label encodes categorical features (Sex, Housing, Saving accounts, Checking account)
- Interactive web UI built with Streamlit

## Usage

Install dependencies:

```bash
pip install scikit-learn xgboost joblib streamlit pandas
```

Run the app:

```bash
streamlit run app.py
```

## Inputs

| Field | Type | Description |
|---|---|---|
| Age | Number | Applicant age (18–80) |
| Sex | Select | male / female |
| Job | Number | Job type (0–3) |
| Housing | Select | own / rent / free |
| Saving accounts | Select | little / moderate / rich / quite rich |
| Checking account | Select | little / moderate / rich |
| Credit amount | Number | Loan amount (100–100,000) |
| Duration | Number | Loan duration in months (1–60) |

## Output

- **LOW risk (Good)** — applicant is likely creditworthy
- **HIGH risk (Bad)** — applicant may default

## Files

- `app.py` — Streamlit app
- `analysis_model.ipynb` — Model training notebook
- `extra_trees_credit_model.pkl` — Trained model
- `*_encoder.pkl` — Label encoders for categorical features
- `german_credit_data.csv` — Training dataset
