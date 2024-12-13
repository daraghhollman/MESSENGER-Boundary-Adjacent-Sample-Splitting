"""
Script to overplot the time series of correctly classified samples, and compare to the time series of misclassified samples.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load samples
solar_wind_samples = pd.read_csv(
    "/home/daraghhollman/Main/Work/mercury/DataSets/solar_wind_sample_10_mins.csv"
)
magnetosheath_samples = pd.read_csv(
    "/home/daraghhollman/Main/Work/mercury/DataSets/magnetosheath_sample_10_mins.csv"
)

# We want to stack these on top of each other (with the addition of a label column) so
# that we can compare to indices output from our random forest
solar_wind_samples["Label"] = "Solar Wind"
magnetosheath_samples["Label"] = "Magnetosheath"

all_samples = pd.concat([solar_wind_samples, magnetosheath_samples]).reset_index()

# Fix formatting for MAG
components = ["|B|", "Bx", "By", "Bz"]
for component in components:
    all_samples[component] = all_samples[component].apply(
        lambda x: list(map(float, x.strip("[]").split(",")))
    )

# Load random forest predictions
predictions = pd.read_csv(
    "/home/daraghhollman/Main/Work/mercury/DataSets/random_forest_predictions.csv"
)

correct_predictions = predictions.loc[predictions["Truth"] == predictions["Prediction"]]
incorrect_predictions = predictions.loc[
    predictions["Truth"] != predictions["Prediction"]
]


alpha = 0.1
variables = ["|B|", "Bx", "By", "Bz"]

for variable in variables:
    # We want two panels, one for solar wind, one for magnetosheath
    fig, (sw_axis, ms_axis) = plt.subplots(1, 2, sharey=True)

    sw_correct_samples = {
        variable: [],
    }
    ms_correct_samples = {
        variable: [],
    }
    sw_incorrect_samples = {
        variable: [],
    }
    ms_incorrect_samples = {
        variable: [],
    }

    for _, correct_prediction in correct_predictions.iterrows():

        index = correct_prediction[correct_predictions.columns[0]]

        corresponding_sample = all_samples.iloc[index].copy()
        seconds = np.arange(0, 600)

        if len(corresponding_sample[variable]) < 600:
            corresponding_sample[variable] = np.append(
                corresponding_sample[variable],
                np.full(600 - len(corresponding_sample[variable]), np.nan),
            )

        if correct_prediction["Truth"] == "Solar Wind":

            sw_axis.plot(
                seconds, corresponding_sample[variable][:600], color="grey", alpha=alpha
            )

            sw_correct_samples[variable].append(corresponding_sample[variable][:600])

        elif correct_prediction["Truth"] == "Magnetosheath":

            ms_axis.plot(
                seconds, corresponding_sample[variable][:600], color="grey", alpha=alpha
            )

            ms_correct_samples[variable].append(corresponding_sample[variable][:600])


    for _, incorrect_prediction in incorrect_predictions.iterrows():

        index = incorrect_prediction[incorrect_predictions.columns[0]]

        corresponding_sample = all_samples.iloc[index].copy()
        seconds = np.arange(0, 600)

        if len(corresponding_sample[variable]) < 600:
            corresponding_sample[variable] = np.append(
                corresponding_sample[variable],
                np.full(600 - len(corresponding_sample[variable]), np.nan),
            )

        if incorrect_prediction["Truth"] == "Solar Wind":

            sw_axis.plot(
                seconds,
                corresponding_sample[variable][:600],
                color="crimson",
                alpha=alpha,
                zorder=2,
            )

            sw_incorrect_samples[variable].append(corresponding_sample[variable][:600])

        elif incorrect_prediction["Truth"] == "Magnetosheath":

            ms_axis.plot(
                seconds,
                corresponding_sample[variable][:600],
                color="crimson",
                alpha=alpha,
                zorder=2,
            )

            ms_incorrect_samples[variable].append(corresponding_sample[variable][:600])


    # Plot average correct and average incorrect
    sw_correct_average = np.nanmean(sw_correct_samples[variable], axis=0)
    sw_axis.plot(
        np.arange(0, len(sw_correct_average)),
        sw_correct_average,
        color="black",
        lw=3,
        label="Correctly Classified Average",
    )

    ms_correct_average = np.nanmean(ms_correct_samples[variable], axis=0)
    ms_axis.plot(
        np.arange(0, len(ms_correct_average)),
        ms_correct_average,
        color="black",
        lw=3,
        label="Correctly Classified Average",
    )

    sw_incorrect_average = np.nanmean(sw_incorrect_samples[variable], axis=0)
    sw_axis.plot(
        np.arange(0, len(sw_incorrect_average)),
        sw_incorrect_average,
        color="firebrick",
        lw=3,
        label="Misclassified Average",
    )

    ms_incorrect_average = np.nanmean(ms_incorrect_samples[variable], axis=0)
    ms_axis.plot(
        np.arange(0, len(ms_incorrect_average)),
        ms_incorrect_average,
        color="firebrick",
        lw=3,
        label="Misclassified Average",
    )

    sw_axis.set_ylabel(f"{variable} [nT]")
    sw_axis.legend()

    sw_axis.set_xlabel("Time [seconds]")
    ms_axis.set_xlabel("Time [seconds]")

    sw_axis.set_title("Solar Wind Samples")
    ms_axis.set_title("Magnetosheath Samples")

    for ax in [sw_axis, ms_axis]:
        ax.margins(0)


    plt.show()
