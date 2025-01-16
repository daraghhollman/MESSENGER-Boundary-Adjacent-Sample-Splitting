"""
We are testing multiple models on the same training and test data. We want a way to compare their outputs, to see if they agree
"""

import pandas as pd


# Load outputs
directory = "/home/daraghhollman/Main/Work/mercury/DataSets/"
random_forest_predictions = pd.read_csv(directory + "random_forest_predictions_reduced.csv")
gradient_boosting_predictions = pd.read_csv(directory + "gradient_boosting_predictions_reduced.csv")

matching_rows = random_forest_predictions["Prediction"] == gradient_boosting_predictions["Prediction"]

print(matching_rows.sum())
print((~matching_rows).sum())
