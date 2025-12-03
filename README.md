# EarlyDiabetesML

Machine learning project for early diabetes risk prediction based on a medical questionnaire.  
This repository contains the code, dataset and thesis materials from my graduation project.

---

## Project overview

The goal of this project is to explore whether basic questionnaire data (symptoms, demographics)
is enough to build a model that predicts the presence of diabetes at an early stage.

The repository includes:

- `early_diabetes_ml.py` – main script with data analysis and ML models  
- `diabetes_data.csv` – source dataset (semicolon-separated)  
- `EarlyDiabetesML_Thesis.pdf` – thesis (public version)  
- `presentation.pptx` – defense presentation  
- `requirements.txt` – Python dependencies  
- `describe.csv` – generated descriptive statistics (created automatically by the script)

---

## Models

The script trains and evaluates several classification models:

- **Logistic Regression**
- **k-Nearest Neighbors (kNN)**
- **Decision Tree**
- **Random Forest**

For each model the following metrics are printed:

- Accuracy
- Recall
- Precision
- F1-score

Additionally, the script generates several visualizations:

- Target distribution (diabetes / no diabetes)
- Distributions by gender, age and symptoms
- Correlation heatmap for all features

---

## Dataset

- File: `diabetes_data.csv`  
- Separator: `;` (semicolon)  
- Target column: `class`  
- Example feature columns: `gender`, `age`, `polyuria`, `polydipsia`, `sudden_weight_loss`, `weakness`, etc.

In the script, the `gender` column is converted to numeric:

```python
df["gender"] = df["gender"].replace(["Male", "Female"], [0, 1])
