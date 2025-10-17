"""
Credit Risk Prediction - Inference Script
-----------------------------------------
Loads the trained GradientBoosting Classifier model from MLflow Model Registry,
performs predictions on unseen (test) data, and exports the results to a CSV file.
"""

import mlflow
import pandas as pd
import os



MODEL_NAME = "Credit Risk Prediction GradientBoosting Classifier"
MODEL_VERSION = "4"   
TEST_DATA_PATH = "notebook/data/test.csv"   
OUTPUT_PATH = "notebook/data/submission.csv"

# Load the trained model from MLflow Model Registry
model_uri = "models:/{MODEL_NAME}/{MODEL_VERSION}"
print(f"Loading model from: {model_uri}")



try:
    model = mlflow.pyfunc.load_model(model_uri)
    print("Model successfully loaded from MLflow registry.")
except Exception as e:
    raise RuntimeError(f"Failed to load model from MLflow: {e}")


# Load the test data
test_data = pd.read_csv(TEST_DATA_PATH)

df_test_data =test_data.drop('SeriousDlqin2yrs', axis=1)

df_test_data = df_test_data.fillna(df_test_data.median())

# Perform predictions on the test data
predictions = model.predict(df_test_data)

df_test_data_with_predictions = df_test_data.copy()

df_test_data_with_predictions['SRC_DLQ_TWO_YRS_PREDICTED'] = predictions

df_test_data_with_predictions.to_csv(OUTPUT_PATH, index=False)


print(f"Predictions saved to: {OUTPUT_PATH}")