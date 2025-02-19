import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import pickle
import arviz as az
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
import pymc as pm
import pymc_bart as pmb
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def main():
    reduced_features = True

    # Load data
    combined_features = pd.read_csv(
        "/home/daraghhollman/Main/Work/mercury/DataSets/combined_features.csv"
    )

    # BART can't handle missing values.
    combined_features = combined_features.dropna()

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

    X = X.iloc[:, 1:]  # Remove the index column

    column_names = list(X.columns.values)
    column_names.sort()
    X = X[column_names]

    _, _, _, y_test_with_labels = train_test_split(
        X, combined_features["label"], test_size=0.2, random_state=1
    )
    y = pd.Categorical(combined_features["label"]).codes  # Target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1
    )
    _, regions = pd.factorize(y, sort=True)

    coords = {"n_obs": np.arange(len(X_train)), "regions": regions}
    with pm.Model(coords=coords) as model:

        X_data = pm.Data("X_data", X_train)
        y_data = pm.Data("y_data", y_train)

        mu = pmb.BART("mu", X_data, y_train, m=20, dims=["regions", "n_obs"])
        softmax = pm.Deterministic(
            "softmax", pm.math.softmax(mu, axis=0)
        )  # use softmax to restrict mu between 0 and 1
        y = pm.Categorical("y", p=softmax.T, observed=y_data)

        idata = pm.sample(chains=4, draws=1000)
        pp_train = pm.sample_posterior_predictive(idata)

    with model:
        # Update the data to perform test
        pm.set_data(
            new_data={"X_data": X_test, "y_data": np.zeros(len(y_test), dtype=int)},
            coords={"n_obs": np.arange(len(X_test)), "regions": regions},
        )
        pp_test = pm.sample_posterior_predictive(idata, return_inferencedata=True)

    # Average accross chains and draws
    # The result is an array with elements between 0 to 1. With 0 corresponding to magnetosheath,
    # and 1 to solar wind. These can be remapped to a 'probability score' between 0 and 1 for each
    # region.
    predictions = np.mean(
        np.mean(pp_test.posterior_predictive.y.to_numpy(), axis=0), axis=0
    )
    solar_wind_probability = predictions
    magnetosheath_probability = 1 - predictions

    # Get Accuracy
    accuracy = (np.round(predictions) == y_test).sum() / len(y_test)

    print(f"Accuracy: {accuracy}")

    # Save to csv
    if input("Save predictions to csv? [Y/n]\n > ") != "n":
        prediction_labels = np.where(
            np.round(predictions) == 1, "Solar Wind", "Magnetosheath"
        )

        prediction_data = pd.DataFrame(
            {
                "Truth": y_test_with_labels,
                "Prediction": prediction_labels,
                "P(Magnetosheath)": magnetosheath_probability,
                "P(Solar Wind)": solar_wind_probability,
            }
        )
        prediction_data.to_csv(
            "/home/daraghhollman/Main/Work/mercury/DataSets/bart_predictions.csv"
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
