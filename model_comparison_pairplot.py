"""
MODEL COMPARISON PARIPLOT
A script to compare the output predictions of two models for any two variables.
"""

import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

# SETTINGS
# Choose which features to compare, (x, y)
plotted_features = "X MSM' (radii)", "Mean |B|"
boundary = "bow_shock"
colours = ["black", "#DC267F", "#648FFF", "#FFB000"]

# Load models
models = {
    "Random Forest": {
        "Name": "Random Forest",
        "Path": f"/home/daraghhollman/Main/Work/mercury/DataSets/{boundary}/random_forest_model",
    },
    "Gradient Boosting": {
        "Name": "Gradient Boosting",
        "Path": f"/home/daraghhollman/Main/Work/mercury/DataSets/{boundary}/gradient_boosting_model",
    },
}

# Load data
data_to_predict: pd.DataFrame = pd.read_csv(
    f"/home/daraghhollman/Main/Work/mercury/DataSets/{boundary}/combined_features.csv"
).dropna()
correct_predictions = data_to_predict["label"]

# Choose models to be compared
selected_models = [models["Random Forest"], models["Gradient Boosting"]]

for model in selected_models:
    with open(model["Path"], "rb") as file:
        current_model = pickle.load(file)

    # Only input features that were used to make the model
    model_features = sorted(current_model.feature_names_in_)
    data_to_predict = data_to_predict[model_features]

    # Make predictions on data using models
    predictions = current_model.predict(data_to_predict)
    probabilities = current_model.predict_proba(data_to_predict)

    model["Predictions"] = predictions

# Plot
# We want three types of scatter:
#   1. Both models are correct for this sample
#   2. Both models disagree for this sample
#   3. Both models are incorrect for this sample

# Get the indices for each of these cases
models_agree_correctly = np.where(
    (selected_models[0]["Predictions"] == selected_models[1]["Predictions"])
    & (selected_models[0]["Predictions"] == correct_predictions)
)[0]
models_agree_incorrectly = np.where(
    (selected_models[0]["Predictions"] == selected_models[1]["Predictions"])
    & (selected_models[0]["Predictions"] != correct_predictions)
)[0]
models_disagree = np.where(
    selected_models[0]["Predictions"] != selected_models[1]["Predictions"]
)[0]


fig, ax = plt.subplots()

# Loop through each case and plot
# PREDICTION A == PREDICTION B == LABEL
x = data_to_predict[plotted_features[0]].iloc[models_agree_correctly]
y = data_to_predict[plotted_features[1]].iloc[models_agree_correctly]

xy = np.vstack([x, y])
kde = gaussian_kde(xy)(xy)

plt.tricontourf(x, y, kde, levels=10, cmap="binary")  # Triangular interpolation
plt.colorbar(
    label=f"Kernel Density Estimate:\nPrediction A == Predicition B == Label (N={len(x)})"
)

# PREDICTION A == PREDICTION B != LABEL
x = data_to_predict[plotted_features[0]].iloc[models_agree_incorrectly]
y = data_to_predict[plotted_features[1]].iloc[models_agree_incorrectly]

ax.scatter(
    x,
    y,
    marker="^",
    color=colours[1],
    alpha=1,
    label=f"Prediction A == Prediction B != Label (N={len(x)})",
)

# PREDICTION A != PREDICTION B
x = data_to_predict[plotted_features[0]].iloc[models_disagree]
y = data_to_predict[plotted_features[1]].iloc[models_disagree]

ax.scatter(
    x,
    y,
    marker="x",
    color=colours[3],
    alpha=1,
    label=f"Prediction A != Prediction B (N={len(x)})",
)

ax.set_title(
    f"Model comparison between {selected_models[0]['Name']} (A) and {selected_models[1]['Name']} (B) models\n"
    + f"for region predictions surrounding the {boundary.replace('_', ' ').title()}."
)

ax.set_xlabel(plotted_features[0])
ax.set_ylabel(plotted_features[1])

ax.legend()

plt.show()
