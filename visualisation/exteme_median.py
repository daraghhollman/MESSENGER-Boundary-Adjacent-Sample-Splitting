"""
Script to find and plot samples with extreme skew values
"""

import datetime as dt

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from hermpy.utils import User
import hermpy.mag as mag
import hermpy.boundaries as boundaries
import hermpy.plotting as hermplot

crossings = boundaries.Load_Crossings(User.CROSSING_LISTS["Philpott"])

combined_features = pd.read_csv(
    "/home/daraghhollman/Main/Work/mercury/DataSets/combined_features.csv"
)
sw_features = combined_features.loc[ combined_features["label"] == "Solar Wind" ]

# Find rows with Median |B| (SW) > 100
extreme_median_samples = sw_features.loc[ np.abs(sw_features["Median |B|"]) > 100 ]

print(len(extreme_median_samples))

for i in range(len(extreme_median_samples)):

    selected_sample = extreme_median_samples.iloc[i]

    selected_sample["Sample Start"] = pd.to_datetime(selected_sample["Sample Start"])
    selected_sample["Sample End"] = pd.to_datetime(selected_sample["Sample End"])

    time_buffer = dt.timedelta(minutes=40)
    start = selected_sample["Sample Start"] - time_buffer
    end = selected_sample["Sample End"] + time_buffer


    data = mag.Load_Between_Dates(User.DATA_DIRECTORIES["MAG"], start, end)

    fig, ax = plt.subplots()

    # Shade sample
    ax.axvspan(pd.to_datetime(selected_sample["Sample Start"]), selected_sample["Sample End"], color="grey", alpha=0.5)

    # Plot Timeseries
    ax.plot(data["date"], data["|B|"], color="k")
    boundaries.Plot_Crossing_Intervals(ax, start, end, crossings)


    hermplot.Add_Tick_Ephemeris(ax)
    ax.set_ylabel("|B| [nT]")
    ax.set_yscale("log")
    
    sample_data = data.loc[ data["date"].between(selected_sample["Sample Start"], selected_sample["Sample End"]) ]

    plt.show()
