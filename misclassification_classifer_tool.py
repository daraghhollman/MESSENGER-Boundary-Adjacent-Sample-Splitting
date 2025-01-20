"""
A tool to classify misclassifications in the random forest data set
"""

import csv
import os
import datetime as dt
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import pandas as pd
from hermpy import mag, boundaries, utils, plotting

output_csv = "/home/daraghhollman/Main/Work/mercury/DataSets/manually_classified_random_forest_misclassifications.csv"

# If the file doesn't exist, create it
if not os.path.exists(output_csv):
    os.mknod(output_csv)

    with open(output_csv, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Misclassification Index", "Label"])

# Load the CSV data
random_forest_predictions = pd.read_csv(
    "/home/daraghhollman/Main/Work/mercury/DataSets/random_forest_predictions_all_data.csv"
)
features_dataset = pd.read_csv(
    "/home/daraghhollman/Main/Work/mercury/DataSets/combined_features.csv"
)

# Load Philpott Crossings
crossings = boundaries.Load_Crossings(utils.User.CROSSING_LISTS["Philpott"])

misclassifications = random_forest_predictions.loc[
    random_forest_predictions["Truth"] != random_forest_predictions["Prediction"]
]
features_dataset = features_dataset.iloc[misclassifications.iloc[:, 0]]

# Load the entire mission
print("Loading MESSENGER mission into memory")
messenger_data = mag.Load_Mission("/home/daraghhollman/messenger_full_mission.pickle")

sample_times = features_dataset[["Sample Start", "Sample End"]]
sample_times["Sample Start"] = pd.to_datetime(
    sample_times["Sample Start"], format="ISO8601"
)
sample_times["Sample End"] = pd.to_datetime(
    sample_times["Sample End"], format="ISO8601"
)

# Initialize a global index to track the row being processed
# We set the current index to the last one set already
try:
    current_index = pd.read_csv(output_csv).iloc[:, 0].iloc[-1] + 1
except ValueError:
    current_index = 0

# Create a function to save classification results
def Save_Classification(index, label):
    with open(output_csv, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([index, label])
    print(f"Row {index} classified as '{label}'.")


# Function to update the plot
def Update_Plot(ax, index):
    # First we need to clear what was there before
    ax.clear()

    start = sample_times.iloc[index]["Sample Start"]
    end = sample_times.iloc[index]["Sample End"]
    time_buffer = dt.timedelta(hours=2)

    ax.set_xlim(start - time_buffer / 4, end + time_buffer / 4)

    # Load data
    data = messenger_data.loc[
        messenger_data["date"].between(start - time_buffer, end + time_buffer)
    ]

    if len(data) == 0:
        raise ValueError(f"No data for sample at row {index}!")

    ax.plot(data["date"], data["|B|"], color="black", lw=1, label="|B|")
    ax.plot(data["date"], data["Bx"], color="#DC267F", lw=0.8, alpha=0.5, label="Bx")
    ax.plot(data["date"], data["By"], color="#648FFF", lw=0.8, alpha=0.5, label="By")
    ax.plot(data["date"], data["Bz"], color="#FFB000", lw=0.8, alpha=0.5, label="Bz")

    ax.set_ylabel("")

    mag_leg = ax.legend(
        bbox_to_anchor=(0.5, 1.1), loc="center", ncol=4, borderaxespad=0.5
    )

    ax.margins(0)
    ax.axvspan(start, end, color="grey", alpha=0.2)
    ax.axhline(0, color="grey", alpha=0.5, ls="dashed")
    boundaries.Plot_Crossing_Intervals(ax, data["date"].iloc[0], data["date"].iloc[-1], crossings)
    plotting.Add_Tick_Ephemeris(ax)

    ax.annotate(
        f"Misclassification #{str(index + 1).zfill(3)}",
        xy=(0, 1),
        xycoords="axes fraction",
        size=12,
        ha="left",
        va="top",
        bbox=dict(boxstyle="round", fc="w"),
    )

    # set the linewidth of each legend object
    for legobj in mag_leg.legend_handles:
        legobj.set_linewidth(2.0)

    fig = ax.get_figure()
    fig.suptitle(
        f"Truth: {misclassifications.iloc[index]['Truth']}    Prediction: {misclassifications.iloc[index]['Prediction']}"
    )

    plt.draw()


# Event handlers for button clicks
def Classify(label):
    global current_index
    Save_Classification(current_index, label)

    current_index += 1
    if current_index < len(misclassifications):
        Update_Plot(ax, current_index)
    else:
        print("All rows classified!")
        plt.close()


# Set up the plot
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.3)
Update_Plot(ax, current_index)

# Create buttons
labels = ["Random Forest Error", "Philpott Mislabelling", "Boundary Layer", "ICME", "Solar Wind Extrema (Low)", "Solar Wind Extrema (High)"]
button_length = 0.13
button_spacing = 0.15
button_left_margin = 0.02
buttons = []
for i, label in enumerate(labels):
    ax_button = plt.axes([button_left_margin + i * button_spacing, 0.05, button_length, 0.075])
    button = Button(ax_button, label)
    button.on_clicked(lambda _, lbl=label: Classify(lbl))
    buttons.append(button)

plt.show()
