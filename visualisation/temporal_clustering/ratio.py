"""
A script to investigate the temporal spread of misclassified samples.
Looking at the time series of probabilities for correct and incorrect samples.
"""

import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.widgets import Slider
import pandas as pd
import numpy as np

# Import sample features
combined_features = pd.read_csv(
    "/home/daraghhollman/Main/Work/mercury/DataSets/combined_features.csv",
)

# Load random forest predictions
predictions = pd.read_csv(
    "/home/daraghhollman/Main/Work/mercury/DataSets/random_forest_predictions.csv"
)

correctness = []

for _, row in predictions.iterrows():

    if row["Truth"] == "Solar Wind":
        correctness.append(row["P(Solar Wind)"])

    else:
        correctness.append(row["P(Magnetosheath)"])

predictions["Correctness"] = correctness

correct_predictions = predictions.loc[predictions["Truth"] == predictions["Prediction"]]
incorrect_predictions = predictions.loc[
    predictions["Truth"] != predictions["Prediction"]
]

# Limit to inbound or outbound only
direction = "none"
match direction:
    case "inbound":
        correct_predictions = correct_predictions.loc[
            combined_features["Is Inbound?"] == 1
        ]
        incorrect_predictions = incorrect_predictions.loc[
            combined_features["Is Inbound?"] == 1
        ]

    case "outbound":
        correct_predictions = correct_predictions.loc[
            combined_features["Is Inbound?"] == 0
        ]
        incorrect_predictions = incorrect_predictions.loc[
            combined_features["Is Inbound?"] == 0
        ]

    case _:
        pass

correctly_classified_indices = correct_predictions[correct_predictions.columns[0]]
misclassified_indices = incorrect_predictions[incorrect_predictions.columns[0]]

misclassified_start_times = combined_features["Sample Start"].iloc[
    misclassified_indices
]
correctly_classified_start_times = combined_features["Sample Start"].iloc[
    correctly_classified_indices
]

misclassified_start_times = pd.to_datetime(
    misclassified_start_times, format="ISO8601"
).tolist()
correctly_classified_start_times = pd.to_datetime(
    correctly_classified_start_times, format="ISO8601"
).tolist()


# Create arbitrary bins in time
number_of_bins = 15

start = dt.datetime(2011, 3, 1)
end = dt.datetime(2015, 5, 1)
bin_edges = [
    start + i * (end - start) / number_of_bins for i in range(number_of_bins + 1)
]
bin_centers = (mdates.date2num(bin_edges[:-1]) + mdates.date2num(bin_edges[1:])) / 2

correct_histogram, _ = np.histogram(mdates.date2num(correctly_classified_start_times), bins=mdates.date2num(bin_edges))
incorrect_histogram, _ = np.histogram(mdates.date2num(misclassified_start_times), bins=mdates.date2num(bin_edges))

fig, ax = plt.subplots()

ax.scatter(
    correctly_classified_start_times,
    correct_predictions["Correctness"],
    color="black",
    marker=".",
)

ax.scatter(
    misclassified_start_times,
    incorrect_predictions["Correctness"],
    color="indianred",
    marker=".",
)

for edge in bin_edges:
    ax.axvline(edge, color="lightgrey")

line = ax.plot(bin_centers, incorrect_histogram / correct_histogram)

plt.ylim(0, 1)
plt.margins(0)
plt.xlabel("Date")
plt.ylabel("P ( Truth )")

plt.show()
