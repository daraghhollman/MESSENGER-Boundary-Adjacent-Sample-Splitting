"""
Script to compare the distribution of correctly classified and misclasified for any feature variable
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

combined_features = pd.read_csv(
    "/home/daraghhollman/Main/Work/mercury/DataSets/combined_features.csv"
)

# We want to plot the misclassified samples' features atop the pairplot
random_forest_results = pd.read_csv(
    "/home/daraghhollman/Main/Work/mercury/DataSets/random_forest_predictions.csv"
)
correctly_classified_results = random_forest_results.loc[
    random_forest_results["Truth"] == random_forest_results["Prediction"]
]
misclassified_results = random_forest_results.loc[
    random_forest_results["Truth"] != random_forest_results["Prediction"]
]

print(f"N (Correctly Classified) = {len(correctly_classified_results)}")
print(f"N (Misclassified) = {len(misclassified_results)}")

# Map these back to their features
correctly_classified_samples = combined_features.iloc[
    correctly_classified_results[correctly_classified_results.columns[0]]
]

misclassified_samples = combined_features.iloc[
    misclassified_results[misclassified_results.columns[0]]
]

# Select a variable
# comparison_variable = "X MSM' (radii)"
comparison_variables = combined_features.columns
comparison_variables = comparison_variables.drop(
    labels=[
        "Unnamed: 0",
        "Sample Start",
        "Sample End"
    ]
)
print(comparison_variables)

for comparison_variable in comparison_variables:

    solar_wind = {
        "Correctly Classified": correctly_classified_samples.loc[
            correctly_classified_samples["label"] == "Solar Wind"
        ][comparison_variable].dropna(),
        "Misclassified": misclassified_samples.loc[
            misclassified_samples["label"] == "Solar Wind"
        ][comparison_variable].dropna(),
    }

    magnetosheath = {
        "Correctly Classified": correctly_classified_samples.loc[
            correctly_classified_samples["label"] == "Magnetosheath"
        ][comparison_variable].dropna(),
        "Misclassified": misclassified_samples.loc[
            misclassified_samples["label"] == "Magnetosheath"
        ][comparison_variable].dropna(),
    }

    fig, axes = plt.subplots(1, 2, sharey=True)
    fig.set_size_inches(10, 6)

    sw_ax, msh_ax = axes
    sw_ax.set_title("Solar Wind")
    msh_ax.set_title("Magnetosheath")

    x_labels = ["Correctly Classified", "Misclassified"]
    box_width = 0.5
    sw_ax.boxplot(solar_wind.values(), tick_labels=x_labels, widths=box_width)
    msh_ax.boxplot(magnetosheath.values(), tick_labels=x_labels, widths=box_width)

    sw_ax.set_ylabel(comparison_variable)

    # Check if either of the distributions cross y=0
    if (
        (
            np.max(solar_wind["Correctly Classified"]) > 0
            and np.min(solar_wind["Correctly Classified"]) < 0
        )
        or (
            np.max(solar_wind["Misclassified"]) > 0
            and np.min(solar_wind["Misclassified"]) < 0
        )
    ) or (
        (
            np.max(magnetosheath["Correctly Classified"]) > 0
            and np.min(magnetosheath["Correctly Classified"]) < 0
        )
        or (
            np.max(magnetosheath["Misclassified"]) > 0
            and np.min(magnetosheath["Misclassified"]) < 0
        )
    ):
        sw_ax.axhline(0, ls="dashed", color="lightgrey")
        msh_ax.axhline(0, ls="dashed", color="lightgrey")

    plt.show()
