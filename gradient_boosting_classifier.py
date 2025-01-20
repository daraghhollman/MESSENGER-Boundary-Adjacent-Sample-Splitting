import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import pickle
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def main():
    show_plots = True
    reduced_features = True
    save_model = True
    drop_nan = True

    # Load data
    combined_features = pd.read_csv(
        "/home/daraghhollman/Main/Work/mercury/DataSets/combined_features.csv"
    )

    if drop_nan:
        combined_features = combined_features.dropna()

    X = combined_features.drop(columns=["label", "Sample Start", "Sample End"])  # Features

    if reduced_features:
        X = X.drop(columns=[
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
        ])


    X = X.iloc[:, 1:]  # Remove the index column

    column_names = list(X.columns.values)
    column_names.sort()
    X = X[column_names]

    y = combined_features["label"]  # Target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    if input("Show training spread? [Y/n]\n > ") != "n":
        Show_Training_Spread(X_train)

    model = HistGradientBoostingClassifier(random_state=0)
    model.fit(X_train, y_train)
    
    if save_model:
        with open("/home/daraghhollman/Main/Work/mercury/DataSets/bow_shock_gradient_boosting", "wb") as file:
            pickle.dump(model, file)

    if show_plots:

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy:.4f}")

        cm = confusion_matrix(y_test, y_pred)

        print("\nConfusion Matrix\n")
        print(cm)

        print("\nClassification Report\n")
        print(classification_report(y_test, y_pred))

        cm_display = ConfusionMatrixDisplay(
            cm, display_labels=["Magnetosheath", "Solar Wind"]
        )
        cm_display.plot()
        plt.show()


    if input("Save predictions to csv? [Y/n]\n > ") != "n":
        truths = y_test  # What the correct label is
        predictions = []  # What the random forest predicted
        sheath_probability = []  # probability it is sheath
        solar_wind_probability = []  # probability it is solar wind

        # Create dataframe of how well the random forest performed
        for i in tqdm(range(len(X_test))):
            sample = X_test.iloc[i].to_frame().T

            prediction = model.predict(sample)[0]
            probabilities = model.predict_proba(sample)[0]

            predictions.append(prediction)
            sheath_probability.append(probabilities[0])
            solar_wind_probability.append(probabilities[1])

        prediction_data = pd.DataFrame(
            {
                "Truth": truths,
                "Prediction": predictions,
                "P(Magnetosheath)": sheath_probability,
                "P(Solar Wind)": solar_wind_probability,
            }
        )
        prediction_data.to_csv(
            "/home/daraghhollman/Main/Work/mercury/DataSets/gradient_boosting_predictions_reduced_dropna.csv"
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
