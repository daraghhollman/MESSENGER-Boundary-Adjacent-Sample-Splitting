"""
A tool to classify misclassifications in the random forest data set
"""

import csv
import os
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from matplotlib.widgets import Button
import pandas as pd
from hermpy import mag, fips, boundaries, utils, plotting

output_csv = "/home/daraghhollman/Main/Work/mercury/DataSets/manually_classified_model_misclassifications.csv"

# If the file doesn't exist, create it
if not os.path.exists(output_csv):
    os.mknod(output_csv)

    with open(output_csv, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Misclassification Index", "Label"])

# Load the CSV data
random_forest_predictions = pd.read_csv(
    "/home/daraghhollman/Main/Work/mercury/DataSets/gradient_bossting_predictions_all_data.csv"
)
features_dataset = pd.read_csv(
    "/home/daraghhollman/Main/Work/mercury/DataSets/combined_features.csv"
)

# Load ICMEs csv
icmes = pd.read_csv(
    "/home/daraghhollman/Main/Work/mercury/DataSets/MESSENGER_ICME_Dates_Winslow.csv",
    parse_dates=["icme_start", "icme_end"],
)

# Load Philpott Crossings
crossings = boundaries.Load_Crossings(utils.User.CROSSING_LISTS["Philpott"])

misclassifications = random_forest_predictions.loc[
    random_forest_predictions["Truth"] != random_forest_predictions["Prediction"]
]
print(f"{len(misclassifications)} misclassifications")
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
except IndexError:
    current_index = 0


# Create a function to save classification results
def Save_Classification(index, label):
    with open(output_csv, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([index, label])
    print(f"Misclassification {index + 1} classified as '{label}'.")


# Function to update the plot
def Update_Plot(axes, index):

    mag_axis, fips_axis = axes
    fig = mag_axis.get_figure()

    # Remove any existing colorbar from the axis
    try:
        for axis in fig.axes:
            if "Proton" in axis.get_xlabel():
                fig.delaxes(axis)
    except IndexError:
        pass

    # Remove any existing ICME flags
    for text in fig.texts:
        if "ICME" in text.get_text():
            text.remove()

    fips_ax_divider = make_axes_locatable(fips_axis)
    fips_cax = fips_ax_divider.append_axes("top", size="3%", pad="2%")

    start = sample_times.iloc[index]["Sample Start"]
    end = sample_times.iloc[index]["Sample End"]
    time_buffer = dt.timedelta(hours=2)

    # Check if sample within an ICME window
    sample_is_in_icme = False
    icme_start = ""
    icme_end = ""
    for _, row in icmes.iterrows():
        if (start > row["icme_start"] and start < row["icme_end"]) or (
            end < row["icme_end"] and end > row["icme_start"]
        ):
            sample_is_in_icme = True
            icme_start = row["icme_start"]
            icme_end = row["icme_end"]
            break

        else:
            continue

    for ax in axes:
        # First we need to clear what was there before
        ax.clear()
        ax.set_xlim(start - time_buffer / 4, end + time_buffer / 4)
        ax.margins(0)

    if sample_is_in_icme:
        fig.text(
            0.03,
            0.92,
            f"Sample is within ICME\nStarting: {icme_start}, Ending: {icme_end}",
        )

    # Load data
    mag_data = messenger_data.loc[
        messenger_data["date"].between(start - time_buffer, end + time_buffer)
    ]
    fips_data = fips.Load_Between_Dates(
        utils.User.DATA_DIRECTORIES["FIPS"],
        start - time_buffer,
        end + time_buffer,
        strip=True,
    )
    fips_protons = np.transpose(fips_data["proton_energies"])

    try:
        fips_protons = np.delete(fips_protons, -1, 1)
        plot_fips = True
    except:
        print("Warning: No FIPS data")
        plot_fips = False

    fips_calibration = fips.Get_Calibration()
    fips_axis.set_yscale("log")

    if len(mag_data) == 0 or len(fips_data) == 0:
        raise ValueError(f"No data for sample at row {index}!")

    mag_axis.plot(mag_data["date"], mag_data["|B|"], color="black", lw=1, label="|B|")
    mag_axis.plot(
        mag_data["date"], mag_data["Bx"], color="#DC267F", lw=0.8, alpha=0.5, label="Bx"
    )
    mag_axis.plot(
        mag_data["date"], mag_data["By"], color="#648FFF", lw=0.8, alpha=0.5, label="By"
    )
    mag_axis.plot(
        mag_data["date"], mag_data["Bz"], color="#FFB000", lw=0.8, alpha=0.5, label="Bz"
    )

    mag_axis.set_ylabel("Magnetic Field Stregnth [nT]")

    mag_leg = mag_axis.legend(
        bbox_to_anchor=(0.5, 1.1), loc="center", ncol=4, borderaxespad=0.5
    )

    mag_axis.axvspan(start, end, color="grey", alpha=0.2)
    mag_axis.axvline(0, color="grey", alpha=0.5, ls="dashed")
    boundaries.Plot_Crossing_Intervals(
        mag_axis, mag_data["date"].iloc[0], mag_data["date"].iloc[-1], crossings
    )

    plotting.Add_Tick_Ephemeris(fips_axis)

    if plot_fips:
        protons_mesh = fips_axis.pcolormesh(
            fips_data["dates"],
            fips_calibration,
            fips_protons,
            norm="log",
            cmap="plasma",
        )

        colorbar_label = "Diff. Energy Flux [(keV/e)$^{-1}$ sec$^{-1}$ cm$^{-2}$]"

        plt.colorbar(
            protons_mesh,
            cax=fips_cax,
            label="Proton " + colorbar_label,
            orientation="horizontal",
            location="top",
        )

    else:
        fips_cax.remove()

    fips_axis.set_ylabel("Energy [keV]")

    mag_axis.annotate(
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

    fig = mag_axis.get_figure()
    fig.suptitle(
        f"Truth: {misclassifications.iloc[index]['Truth']}    Prediction: {misclassifications.iloc[index]['Prediction']}"
    )

    plt.draw()


def Open_Trajectory(_):
    print("Opening Trajectoy")
    global current_index

    # Get start and stop time for this index
    sample_start = sample_times.iloc[current_index]["Sample Start"]
    sample_end = sample_times.iloc[current_index]["Sample End"]
    time_buffer = dt.timedelta(hours=4)

    search_start = sample_start - time_buffer
    search_end = sample_end + time_buffer

    all_positions = messenger_data.loc[
        messenger_data["date"].between(search_start, search_end)
    ][["X MSM' (radii)", "Y MSM' (radii)", "Z MSM' (radii)"]]

    sample_positions = messenger_data.loc[
        messenger_data["date"].between(sample_start, sample_end)
    ][["X MSM' (radii)", "Y MSM' (radii)", "Z MSM' (radii)"]]

    # Create new plotting window
    fig, ax = plt.subplots()

    ax.plot(
        all_positions["X MSM' (radii)"],
        np.sqrt(
            all_positions["Y MSM' (radii)"] ** 2 + all_positions["Z MSM' (radii)"] ** 2
        ),
        color="grey",
        zorder=1,
    )
    ax.plot(
        sample_positions["X MSM' (radii)"],
        np.sqrt(
            sample_positions["Y MSM' (radii)"] ** 2
            + sample_positions["Z MSM' (radii)"] ** 2
        ),
        color="black",
        lw=5,
        zorder=2,
    )

    final_position = sample_positions.iloc[-1]
    previous_position = sample_positions.iloc[-2]

    arrow_direction = [
        final_position["X MSM' (radii)"] - previous_position["X MSM' (radii)"],
        np.sqrt(
            final_position["Y MSM' (radii)"] ** 2
            + final_position["Z MSM' (radii)"] ** 2
        )
        - np.sqrt(
            previous_position["Y MSM' (radii)"] ** 2
            + previous_position["Z MSM' (radii)"] ** 2
        ),
    ]
    arrow_direction /= np.linalg.norm(arrow_direction)
    arrow_length = 0.1
    ax.arrow(
        final_position["X MSM' (radii)"],
        np.sqrt(
            final_position["Y MSM' (radii)"] ** 2
            + final_position["Z MSM' (radii)"] ** 2
        ),
        arrow_length * arrow_direction[0],
        arrow_length * arrow_direction[1],
        width=0.01,
        head_width=0.2,
        head_length=0.2,
        ec="black",
        fc="black",
        zorder=5,
    )

    plotting.Format_Cylindrical_Plot(ax, 4)
    plotting.Plot_Magnetospheric_Boundaries(ax)

    plt.show()


# Event handlers for button clicks
def Classify(label):
    global current_index
    Save_Classification(current_index, label)

    current_index += 1
    if current_index < len(misclassifications):
        Update_Plot(axes, current_index)
    else:
        print("All rows classified!")
        plt.close()


# Set up the plot
fig, axes = plt.subplots(2, 1, sharex=True)
(mag_axis, fips_axis) = axes
plt.subplots_adjust(bottom=0.3, hspace=0.3)

Update_Plot(axes, current_index)

# Create buttons
labels = [
    "Random Forest Error",
    "Philpott Mislabelling",
    "Boundary Layer",
    "ICME",
    "Solar Wind Extrema",
    "Unknown",
]
button_length = 0.13
button_spacing = 0.15
button_left_margin = 0.02
buttons = []
for i, label in enumerate(labels):
    ax_button = plt.axes(
        [button_left_margin + i * button_spacing, 0.05, button_length, 0.075]
    )
    button = Button(ax_button, label)
    button.on_clicked(lambda _, lbl=label: Classify(lbl))
    buttons.append(button)

trajectory_button_ax = plt.axes([0.8, 0.9, button_length, 0.075])
trajectory_button = Button(trajectory_button_ax, "Open Trajectory")
trajectory_button.on_clicked(Open_Trajectory)

plt.show()
