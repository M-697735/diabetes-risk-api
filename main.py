# main.py  -  EDA + training. Run this first:  python main.py
# Step A: EDA (look at the data)   Step B: train the model   Step C: save it

import os
import pickle

import matplotlib
matplotlib.use("Agg")            # draw pictures without opening a window
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------- 1. Load
df = pd.read_csv("diabetes_risk_prediction_dataset.csv")
total_rows, total_columns = df.shape

features = ["Age", "BMI", "Fasting_Blood_Sugar", "HbA1c",
            "Blood_Pressure_Systolic", "Family_History_Diabetes"]
target = "Diabetes_Risk"          # Low / Moderate / High
classes = ["Low", "Moderate", "High"]

# NOTE: Diabetes_Risk_Score, AI_Health_Recommendation and Doctor_Consultation_Needed
# are made FROM the answer, so we do NOT use them (that would be cheating / data leakage).

# ---------------------------------------------------------------- 2. EDA
print("=" * 50)
print("EDA - EXPLORATORY DATA ANALYSIS")
print("=" * 50)

# EDA 1: size of the data
print("\n[1] Shape (rows, columns):", df.shape)

# EDA 2: first rows and data types
print("\n[2] First 3 rows of the chosen columns:")
print(df[features + [target]].head(3).to_string(index=False))
print("\n    Data types:")
print(df[features + [target]].dtypes.to_string())

# EDA 3: missing values
missing = df[features + [target]].isnull().sum()
print("\n[3] Missing values per column:")
print(missing.to_string())

# EDA 4: how many patients in each class (class balance)
class_counts = df[target].value_counts().reindex(classes)
print("\n[4] Patients per risk class:")
for c in classes:
    print(f"    {c:<9}{class_counts[c]:>7}  ({class_counts[c] / len(df) * 100:.1f} %)")

# EDA 5: summary statistics (mean, min, max ...)
print("\n[5] Summary statistics:")
print(df[features[:-1]].describe().round(2).to_string())

# EDA 6: average value of each feature for each risk class
clean = df[features + [target]].dropna().copy()
clean["Family_History_Diabetes"] = clean["Family_History_Diabetes"].map({"No": 0, "Yes": 1})
means = clean.groupby(target)[features].mean().reindex(classes).round(2)
print("\n[6] Average value per risk class:")
print(means.to_string())

# EDA 7: pictures (saved in the eda_outputs folder)
os.makedirs("eda_outputs", exist_ok=True)

plt.figure(figsize=(5, 4))
class_counts.plot(kind="bar", color=["#2a9d8f", "#e9c46a", "#e76f51"])
plt.title("Patients per risk class")
plt.ylabel("Number of patients")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("eda_outputs/class_balance.png", dpi=120)
plt.close()

fig, axes = plt.subplots(2, 3, figsize=(12, 7))
for ax, col in zip(axes.flat, features[:-1] + ["Family_History_Diabetes"]):
    ax.hist(clean[col], bins=25, color="#457b9d")
    ax.set_title(col)
plt.tight_layout()
plt.savefig("eda_outputs/histograms.png", dpi=120)
plt.close()

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for ax, col in zip(axes, ["HbA1c", "Fasting_Blood_Sugar", "BMI"]):
    data = [clean.loc[clean[target] == c, col] for c in classes]
    ax.boxplot(data, tick_labels=classes)
    ax.set_title(col + " by risk class")
plt.tight_layout()
plt.savefig("eda_outputs/boxplots.png", dpi=120)
plt.close()

corr = clean[features].corr()
plt.figure(figsize=(6, 5))
plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
plt.colorbar()
plt.xticks(range(len(features)), features, rotation=45, ha="right")
plt.yticks(range(len(features)), features)
plt.title("Correlation between features")
plt.tight_layout()
plt.savefig("eda_outputs/correlation.png", dpi=120)
plt.close()
print("\n[7] Pictures saved in the eda_outputs/ folder")

# EDA 8: what the EDA tells us
print("\n[8] What we learn:")
print("    - 'Low' risk is very rare, so the model needs class_weight='balanced'.")
print("    - HbA1c and Fasting_Blood_Sugar change most between classes.")
print("    - No strong correlation between the chosen features (good, not repeating info).")

# ---------------------------------------------------------------- 3. Clean
print("\n" + "=" * 50)
print("TRAINING")
print("=" * 50)
df = clean          # rows with empty values are already removed, Yes/No already 1/0
X = df[features]    # inputs
y = df[target]      # answer

# ---------------------------------------------------------------- 4. Split 80/20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------------------------------------------------------------- 5. Train
model = RandomForestClassifier(n_estimators=100, max_depth=8,
                               class_weight="balanced", random_state=42)
model.fit(X_train, y_train)

# ---------------------------------------------------------------- 6. Check
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
recall = recall_score(y_test, predictions, labels=classes, average=None)
recall_per_class = {c: round(float(r), 2) for c, r in zip(classes, recall)}

print("Rows used:", len(df))
print("Accuracy:", round(accuracy * 100, 2), "%")
print("Recall per class:", recall_per_class)

# ---------------------------------------------------------------- 7. Save
eda = {
    "rows": int(total_rows),
    "columns_in_file": int(total_columns),
    "rows_used_for_training": int(len(df)),
    "missing_values": {k: int(v) for k, v in missing.items()},
    "class_counts": {c: int(class_counts[c]) for c in classes},
    "average_per_class": means.to_dict(orient="index"),
}
saved = {"model": model, "features": features, "accuracy": accuracy,
         "recall": recall_per_class, "eda": eda}
with open("diabetes_model.pkl", "wb") as f:
    pickle.dump(saved, f)
print("Model saved as diabetes_model.pkl")
