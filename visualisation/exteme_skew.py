"""
Script to find and plot samples with extreme skew values
"""

import datetime as dt

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from hermpy.utils import User
import hermpy.mag as mag
import hermpy.plotting as hermplot

combined_features = pd.read_csv(
    "/home/daraghhollman/Main/Work/mercury/DataSets/combined_features.csv"
)

# Find rows with abs( Skew(|B|) ) > 5
extreme_skew_samples = combined_features.loc[ np.abs(combined_features["Skew |B|"]) > 5 ]

for i in range(len(extreme_skew_samples)):

    selected_sample = extreme_skew_samples.iloc[i]

    selected_sample["Sample Start"] = pd.to_datetime(selected_sample["Sample Start"])
    selected_sample["Sample End"] = pd.to_datetime(selected_sample["Sample End"])

    time_buffer = dt.timedelta(minutes=20)
    start = selected_sample["Sample Start"] - time_buffer
    end = selected_sample["Sample End"] + time_buffer


    data = mag.Load_Between_Dates(User.DATA_DIRECTORIES["MAG"], start, end)

    fig, ax = plt.subplots()

    # Shade sample
    ax.axvspan(pd.to_datetime(selected_sample["Sample Start"]), selected_sample["Sample End"], color="grey", alpha=0.5)

    # Plot Timeseries
    ax.plot(data["date"], data["|B|"], color="k")


    hermplot.Add_Tick_Ephemeris(ax)
    ax.set_ylabel("|B| [nT]")
    
    sample_data = data.loc[ data["date"].between(selected_sample["Sample Start"], selected_sample["Sample End"]) ]

    print(np.max(sample_data["|B|"]))

    plt.show()
