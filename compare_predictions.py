"""
We are testing multiple models on the same training and test data. We want a way to compare their outputs, to see if they agree
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

colours = ["black", "#DC267F", "#648FFF", "#FFB000"]

bart = False

# Load outputs
directory = "/home/daraghhollman/Main/Work/mercury/DataSets/bow_shock/"

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

non_matching_rows = pair[0]["Prediction"] != pair[1]["Prediction"]

print(pair[0][non_matching_rows]["P(Solar Wind)"])
hist_data = pair[0][non_matching_rows]["P(Solar Wind)"].tolist() + pair[1][non_matching_rows]["P(Solar Wind)"].tolist()
plt.hist(hist_data, color="black", label=f"Differing Predictions, N={len(pair[0][non_matching_rows]['P(Solar Wind)'].tolist() + pair[1][non_matching_rows]['P(Solar Wind)'].tolist())}")

plt.axvline(np.mean(hist_data), color=colours[1], label=f"Mean = {np.mean(hist_data):.3f}")
plt.axvline(np.median(hist_data), color=colours[2], ls="dashed", label=f"Median = {np.median(hist_data)}")

plt.title("Distribution of Probabilities of unmatching classifications\n for two independent-seed Random Forest Models")
plt.xlabel("P(Solar Wind)")
plt.ylabel("# Samples Classified")

plt.legend()

plt.show()

# print(f"Matching Predictions: {matching_rows.sum() / len(matching_rows):.2f}")

# Of the misclassified:
misclassified_rows = [model["Truth"] != model["Prediction"] for model in pair]

misclassification_agreement = (misclassified_rows[0] & misclassified_rows[1]).sum() / (misclassified_rows[0].sum() + misclassified_rows[1].sum() / 2)
# print(misclassification_agreement)
