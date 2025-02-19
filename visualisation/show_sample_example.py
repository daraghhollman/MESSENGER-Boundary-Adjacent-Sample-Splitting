"""
A short script to show what an example of a sample is for both the SW and the MSh
"""

import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
from hermpy import mag, boundaries, utils, plotting


colours = ["black", "#DC267F", "#648FFF", "#FFB000"]

# import crossings
crossings = boundaries.Load_Crossings(utils.User.CROSSING_LISTS["Philpott"])

# Limit to bow shock crossings only
crossings = crossings.loc[crossings["Type"].str.contains("BS")]

# Pick a crossing
crossing = crossings.iloc[1000]

# Determine start and stop times of each sample
if crossing["Type"] == "BS_IN":
    sw_sample_start = crossing["Start Time"] - dt.timedelta(minutes=10)
    sw_sample_end = crossing["Start Time"]

    msh_sample_start = crossing["End Time"]
    msh_sample_end = crossing["End Time"] + dt.timedelta(minutes=10)

else:
    msh_sample_start = crossing["Start Time"] - dt.timedelta(minutes=10)
    msh_sample_end = crossing["Start Time"]

    sw_sample_start = crossing["End Time"]
    sw_sample_end = crossing["End Time"] + dt.timedelta(minutes=10)


time_buffer = dt.timedelta(minutes=1)
all_times = [sw_sample_start, sw_sample_end, msh_sample_start, msh_sample_end]

# Load the data
data = mag.Load_Between_Dates(
    utils.User.DATA_DIRECTORIES["MAG"],
    min(all_times) - time_buffer,
    max(all_times) + time_buffer,
    aberrate=True,
)

sw_sample_data = data.loc[data["date"].between(sw_sample_start, sw_sample_end)]
msh_sample_data = data.loc[data["date"].between(msh_sample_start, msh_sample_end)]


# Plotting
plt.figure(figsize=(10,10))
mag_axis = plt.subplot2grid((2, 2), (0, 0), colspan=2)
left_sample_axis = plt.subplot2grid((2, 2), (1, 0))
right_sample_axis = plt.subplot2grid((2, 2), (1, 1), sharey=left_sample_axis)

axes = (mag_axis, left_sample_axis, right_sample_axis)

# Plot time series
mag_axis.plot(
    data["date"],
    data["|B|"],
    color=colours[0],
    lw=1,
    label="|B|",
    alpha=0.8,
)
mag_axis.plot(
    data["date"],
    data["Bx"],
    color=colours[1],
    lw=1,
    label="Bx",
    alpha=0.8,
)
mag_axis.plot(
    data["date"],
    data["By"],
    color=colours[2],
    lw=1,
    label="By",
    alpha=0.8,
)
mag_axis.plot(
    data["date"],
    data["Bz"],
    color=colours[3],
    lw=1,
    label="Bz",
    alpha=0.8,
)

boundaries.Plot_Crossing_Intervals(
    mag_axis,
    data["date"].iloc[0],
    data["date"].iloc[-1],
    crossings,
    color="black",
    lw=3,
    height=0.95,
)
plotting.Add_Tick_Ephemeris(
    mag_axis,
    include={
        "date",
        "hours",
        "minutes",
        "seconds",
    },
)

mag_legend = mag_axis.legend(bbox_to_anchor=(0.5, 1.1), loc="center", ncol=4, borderaxespad=0.5)

# set the linewidth of each legend object
for legobj in mag_legend.legend_handles:
    legobj.set_linewidth(3.0)


# Highlight samples
mag_axis.axvspan(sw_sample_start, sw_sample_end, color="lightgrey", alpha=0.5)
mag_axis.axvspan(msh_sample_start, msh_sample_end, color="lightgrey", alpha=0.5)

components = ["|B|", "Bx", "By", "Bz"]
if crossing["Type"] == "BS_IN":

    left_sample_axis.set_title("Solar Wind Sample")
    right_sample_axis.set_title("Magnetosheath Sample")

    for component, colour in zip(components, colours):

        left_sample_hist, bin_edges = np.histogram(sw_sample_data[component])
        bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2
        left_sample_axis.stairs(
            left_sample_hist, bin_edges, lw=3, orientation="horizontal", color=colour
        )

        right_sample_hist, bin_edges = np.histogram(msh_sample_data[component])
        bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2
        right_sample_axis.stairs(
            right_sample_hist, bin_edges, lw=3, orientation="horizontal", color=colour
        )
else:

    left_sample_axis.set_title("Magnetosheath Sample")
    right_sample_axis.set_title("Solar Wind Sample")

    for component, colour in zip(components, colours):

        left_sample_hist, bin_edges = np.histogram(msh_sample_data[component])
        bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2
        left_sample_axis.stairs(
            left_sample_hist, bin_edges, lw=3, orientation="horizontal", color=colour
        )

        right_sample_hist, bin_edges = np.histogram(sw_sample_data[component])
        bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2
        right_sample_axis.stairs(
            right_sample_hist, bin_edges, lw=3, orientation="horizontal", color=colour
        )

for ax in axes:
    ax.margins(x=0)

right_sample_axis.set_xlabel("# Data Points (seconds)")
left_sample_axis.set_xlabel("# Data Points (seconds)")

left_sample_axis.set_ylabel("Magnetic Field Strength [nT]")
mag_axis.set_ylabel("Magnetic Field Strength [nT]")

# plt.show()
plt.savefig("/home/daraghhollman/show_sample_example.svg", format="svg")
