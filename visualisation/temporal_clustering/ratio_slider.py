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

# Initialize the figure and axis
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.25)

# Plot placeholders
sc1 = ax.scatter([], [], color="black", marker=".", label="Correct")
sc2 = ax.scatter([], [], color="indianred", marker=".", label="Incorrect")
(line,) = ax.plot([], [], label="P ( Truth )", color="cornflowerblue")
ax.legend()

# Add slider axis
ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03])
slider = Slider(ax_slider, "Bins", 5, 50, valinit=20, valstep=1)


# Function to update the plot based on slider
def update(val):
    number_of_bins = int(slider.val)

    start = dt.datetime(2011, 3, 22)
    end = dt.datetime(2015, 5, 1)
    bin_edges = [
        start + i * (end - start) / number_of_bins for i in range(number_of_bins + 1)
    ]
    bin_centers = (mdates.date2num(bin_edges[:-1]) + mdates.date2num(bin_edges[1:])) / 2

    correct_histogram, _ = np.histogram(
        mdates.date2num(correctly_classified_start_times),
        bins=mdates.date2num(bin_edges),
    )
    incorrect_histogram, _ = np.histogram(
        mdates.date2num(misclassified_start_times), bins=mdates.date2num(bin_edges)
    )


    # Update line data
    line.set_xdata(bin_centers)
    line.set_ydata(
        incorrect_histogram / np.maximum(correct_histogram, 1)
    )  # Avoid division by zero

    # Update axvlines
    ax.clear()
    for edge in bin_edges:
        ax.axvline(edge, color="lightgrey")

    ax.scatter(
        correctly_classified_start_times,
        correct_predictions["Correctness"],
        color="black",
        marker=".",
        label="Correctly Classified Samples",
    )
    ax.scatter(
        misclassified_start_times,
        incorrect_predictions["Correctness"],
        color="indianred",
        marker=".",
        label="Misclassified Samples",
    )

    ax.bar(
        bin_edges[:-1],
        incorrect_histogram / np.maximum(correct_histogram, 1),
        width=np.diff(bin_edges),
        color="cornflowerblue",
        align="edge",
        label="(# Misclassified / Correctly Classified) per bin",
        alpha=0.7,
    )

    ax.set_ylim(0, 1)
    ax.margins(0)
    ax.set_xlabel("Date")

    # Manual ylabels
    ax.text(
        -0.1,
        0.5,
        "P(Truth)",
        fontdict={"size": "large"},
        ha="center",
        va="center",
        rotation="vertical",
        transform=ax.transAxes,
    )
    ax.text(
        -0.05,
        0.5,
        "(# Misclassified / Correctly Classified) per bin",
        color="cornflowerblue",
        fontdict={"size": "large"},
        ha="center",
        va="center",
        rotation="vertical",
        transform=ax.transAxes,
    )

    plt.draw()

    ax.legend(loc="center left")


# Attach the update function to the slider
slider.on_changed(update)

# Initial plot
update(slider.val)

plt.show()
