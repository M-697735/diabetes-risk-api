# Diabetes Risk Prediction API (simple version)

Predicts whether a patient's diabetes risk is **Low, Moderate or High** from 6 health values.

## Files

| File | What it does |
|---|---|
| `diabetes_risk_prediction_dataset.csv` | the data (50,000 patients) |
| `main.py` | EDA (explore data, make pictures) + trains the model + saves `diabetes_model.pkl` |
| `app.py` | Flask API that loads the saved model and answers requests |
| `requirement.txt` | libraries to install |
| `Dockerfile` | packages everything into a Docker container |
| `postman/` | Postman collection to test all endpoints |

## How to run

```
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirement.txt

python main.py     # step 1: EDA + train + save the model
python app.py       # step 2: start the API  ->  http://127.0.0.1:5000
```

Then import `postman/simple-diabetes.postman_collection.json` into Postman and run the requests.

## Endpoints

| Method | URL | What it does |
|---|---|---|
| GET | `/` | API is running |
| GET | `/health` | health check |
| GET | `/model-info` | features, accuracy, recall per class |
| GET | `/eda` | summary of the data (rows, classes, averages) |
| POST | `/predict` | risk for one patient |
| GET | `/history` | last 10 predictions |

Example body for `POST /predict` (Body -> raw -> JSON):

```json
{
  "Age": 65,
  "BMI": 35,
  "Fasting_Blood_Sugar": 190,
  "HbA1c": 10,
  "Blood_Pressure_Systolic": 150,
  "Family_History_Diabetes": 1
}
```

Family_History_Diabetes: 1 = Yes, 0 = No. Answer:

```json
{ "risk": "High", "probabilities": { "High": 1.0, "Low": 0.0, "Moderate": 0.0 } }
```

If a field is missing or is text, the API answers with an error (status 400).

## Docker

```
docker build -t diabetes-simple .
docker run -p 5000:5000 diabetes-simple
```

Then use Postman on `http://127.0.0.1:5000` as before.

## Workflow

```
CSV  ->  EDA  ->  clean  ->  split 80/20  ->  train RandomForest  ->  check accuracy
     ->  save model (.pkl)  ->  Flask API loads it  ->  Postman tests  ->  Docker
```

## Results

- Accuracy: about **80 %** on 20 % of patients the model never saw (always guessing "High" gives 73 %).
- Recall: Low 0.80, Moderate 0.78, High 0.81 (it catches about 80 % of every class).

## Things to mention in your report

- Three columns (`Diabetes_Risk_Score`, `AI_Health_Recommendation`, `Doctor_Consultation_Needed`) were NOT used because they are made from the answer (data leakage).
- "Low" risk is only 1 % of the data, so the model uses `class_weight="balanced"`. Without it accuracy is higher (85 %) but the model never predicts "Low".
- The data looks artificially generated, so the results are for learning, not for real patients.
- Not a medical tool.

## Deploy on Render

1. Push this folder to GitHub (the Actions tab should show a green tick).
2. Render -> New -> Web Service -> pick the repo.
3. Build Command: `pip install -r requirement.txt && python main.py`
4. Start Command: `gunicorn app:app`
5. Open `https://<your-app>.onrender.com/health`.

The model file is not stored in git; Render trains it during the build.
Free Render services go to sleep when unused, so the first request can be slow.
