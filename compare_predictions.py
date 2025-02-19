"""
We are testing multiple models on the same training and test data. We want a way to compare their outputs, to see if they agree
"""

import pandas as pd

bart = False

# Load outputs
directory = "/home/daraghhollman/Main/Work/mercury/DataSets/"

if bart:
    bart_predictions = pd.read_csv(directory + "/bart_predictions.csv")
    random_forest_predictions = pd.read_csv(directory + "random_forest_predictions_reduced_dropna.csv")
    gradient_boosting_predictions = pd.read_csv(directory + "gradient_boosting_predictions_reduced_dropna.csv")

else:
    random_forest_predictions_1 = pd.read_csv(directory + "random_forest_predictions_test002.csv")
    random_forest_predictions_2 = pd.read_csv(directory + "random_forest_predictions_test001.csv")
    gradient_boosting_predictions = pd.read_csv(directory + "gradient_boosting_predictions_reduced.csv")

pair = [random_forest_predictions_1, random_forest_predictions_2]
# matching_rows = random_forest_predictions["Prediction"] == gradient_boosting_predictions["Prediction"]
matching_rows = pair[0]["Prediction"] == pair[1]["Prediction"]

print(f"Matching Predictions: {matching_rows.sum() / len(matching_rows):.2f}")

# Of the misclassified:
misclassified_rows = [model["Truth"] != model["Prediction"] for model in pair]
misclassification_agreement = (misclassified_rows[0] & misclassified_rows[1]).sum() / (misclassified_rows[0].sum() + misclassified_rows[1].sum() / 2)
print(misclassification_agreement)
