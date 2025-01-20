"""
Perform random forest classification for all samples
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import random
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def main():
    reduced_features = True
    show_plots = True
    save_model = False

    # Load data
    combined_features = pd.read_csv(
        "/home/daraghhollman/Main/Work/mercury/DataSets/combined_features.csv"
    )

    X = combined_features.drop(
        columns=["label", "Sample Start", "Sample End"]
    )  # Features

    if reduced_features:
        X = X.drop(
            columns=[
                "Grazing Angle (deg.)",
                "Is Inbound?",
                "Dip Statistic |B|",
                "Dip Statistic Bx",
                "Dip Statistic By",
                "Dip Statistic Bz",
                "Dip P-Value |B|",
                "Dip P-Value Bx",
                "Dip P-Value By",
                "Dip P-Value Bz",
            ]
        )

    indices = X.iloc[:, 0]
    X = X.iloc[:, 1:]  # Remove the index column

    column_names = list(X.columns.values)
    column_names.sort()
    X = X[column_names]

    y = combined_features["label"]  # Target

    # Instead of a standard train test split, we create N groups
    random.shuffle(indices)
    groups = np.array_split(indices, 5)

    outputs = []
    for group in groups:
        X_test = X.iloc[group]
        y_test = y.iloc[group]

        X_train = X.drop(group)
        y_train = y.drop(group)

        random_forest = RandomForestClassifier(n_estimators=100, random_state=0)
        random_forest.fit(X_train, y_train)

        # Assign to dataframe
        truths = y_test  # What the correct label is
        predictions = random_forest.predict(X_test)  # What the random forest predicted
        magnetosheath_probabilities, solar_wind_probabilities = (
            random_forest.predict_proba(X_test).T
        )

        prediction_data = pd.DataFrame(
            {
                "Truth": truths,
                "Prediction": predictions,
                "P(Magnetosheath)": magnetosheath_probabilities,
                "P(Solar Wind)": solar_wind_probabilities,
            }
        )
        outputs.append(prediction_data)

    all_predictions = pd.concat(outputs)

    accuracy = (all_predictions["Truth"] == all_predictions["Prediction"]).sum() / len(all_predictions)
    print(f"Accuracy: {accuracy}")
        
    if input("Save predictions to csv? [Y/n]\n > ") != "n":
        all_predictions.to_csv(
            "/home/daraghhollman/Main/Work/mercury/DataSets/random_forest_predictions_all_data.csv"
        )


# Visualisation functions


def Show_Training_Spread(training_data):
    """A function to check if the training data is disperse spatially.

    This is done by plotting distributions of spatial features.

    """

    features_to_test = [
        "Heliocentric Distance (AU)",
        "Local Time (hrs)",
        "Latitude (deg.)",
        "Magnetic Latitude (deg.)",
        "X MSM' (radii)",
        "Y MSM' (radii)",
        "Z MSM' (radii)",
    ]

    for feature in features_to_test:
        _, ax = plt.subplots()

        ax.hist(training_data[feature], color="black")

        ax.set_xlabel(feature)
        ax.set_ylabel("# Events")

        ax.margins(0)

        plt.show()


if __name__ == "__main__":
    main()
