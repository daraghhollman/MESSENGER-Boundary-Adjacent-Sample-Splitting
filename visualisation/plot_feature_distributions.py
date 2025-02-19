"""
Script to plot the distribution of individual features from the combined features data set.

The aim is to plot the features of the time series to get an image of how often the data is steady vs dynamic. We do this splitting between solar wind and magnetosheath
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats

boundary = input("Boundary: (bow_shock, magnetopause)\n> ")

# Load data set
combined_features = pd.read_csv(
    f"/home/daraghhollman/Main/Work/mercury/DataSets/{boundary}/combined_features.csv"
)

if boundary == "bow_shock":
    features_a = combined_features.loc[combined_features["label"] == "Solar Wind"]
    features_b = combined_features.loc[
        combined_features["label"] == "Magnetosheath"
    ]

else:
    features_a = combined_features.loc[combined_features["label"] == "Magnetosphere"]
    features_b = combined_features.loc[
        combined_features["label"] == "Magnetosheath"
    ]

features_to_plot = [
    "Mean",
    "Median",
    "Standard Deviation",
    "Skew",
    "Kurtosis",
]
bin_sizes = [5, 5, 0.5, 0.5, 0.5]
variables_to_plot = ["|B|", "Bx", "By", "Bz"]

# Perform the code twice
for features_data in [features_a, features_b]:

    for feature, bin_size in zip(features_to_plot, bin_sizes):

        fig, axes = plt.subplots(len(variables_to_plot), 1, sharex=True, sharey=True)

        for ax, variable in zip(axes, variables_to_plot):

            # Remove outliers
            """
            features_data = features_data[
                (np.abs(scipy.stats.zscore(features_data[f"{feature} {variable}"])) < 3)
            ]
            """
            hist_data = features_data[f"{feature} {variable}"]

            bins = np.arange(np.min(hist_data), np.nanmax(hist_data) + bin_size, bin_size)

            ax.hist(hist_data, color="black", bins=bins)

            ax.set_xlabel(f"{feature} {variable}")
            ax.set_ylabel("# Events")

            ax.set_yscale("log")

            ax.margins(0)

        fig.suptitle(features_data.iloc[0]["label"])

        plt.show()
