import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
import seaborn as sns


def main():
    combined_features = pd.read_csv(
        "/home/daraghhollman/Main/Work/mercury/DataSets/combined_features.csv"
    )

    # We want to plot the misclassified samples' features atop the pairplot
    random_forest_results = pd.read_csv(
        "/home/daraghhollman/Main/Work/mercury/DataSets/random_forest_predictions.csv"
    )
    misclassified_results = random_forest_results.loc[
        random_forest_results["Truth"] != random_forest_results["Prediction"]
    ]

    misclassified_samples = combined_features.iloc[misclassified_results.iloc[:, 0]]
    # print(misclassified_samples["label"].tolist())

    variables = [
        "Median |B|",
        "Standard Deviation |B|",
        "Mean |B|",
        "Standard Deviation By",
        "X MSM' (radii)",
    ]

    # Get all possible pairs of variables
    pairs = list(itertools.combinations(variables, 2))

    for pair in pairs:

        var_1, var_2 = pair

        ax = sns.kdeplot(
            data=combined_features, x=var_1, y=var_2, hue="label", log_scale=True
        )

        ax.scatter(
            misclassified_samples.loc[
                misclassified_samples["label"] == "Magnetosheath"
            ][var_1],
            misclassified_samples.loc[
                misclassified_samples["label"] == "Magnetosheath"
            ][var_2],
            color="indianred",
            marker="+",
            zorder=10,
        )
        ax.scatter(
            misclassified_samples.loc[misclassified_samples["label"] == "Solar Wind"][
                var_1
            ],
            misclassified_samples.loc[misclassified_samples["label"] == "Solar Wind"][
                var_2
            ],
            color="cornflowerblue",
            marker="+",
            zorder=10,
        )

        # pairplot = sns.pairplot(combined_features, vars=variables, hue="label", corner=True, kind="kde", plot_kws=dict(alpha=0.5))

        plt.show()


if __name__ == "__main__":
    main()
