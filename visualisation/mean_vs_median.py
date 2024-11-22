import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load data
combined_features = pd.read_csv(
    "/home/daraghhollman/Main/Work/mercury/DataSets/combined_features.csv"
)

solar_wind = combined_features.loc[combined_features["label"] == "Solar Wind"]
magnetosheath = combined_features.loc[combined_features["label"] == "Magnetosheath"]

fig, axes = plt.subplots(2, 2, sharex="all", sharey="all")


axes = axes.flatten()


for ax, component in zip(axes, ["|B|", "Bx", "By", "Bz"]):

    ax.scatter(
        solar_wind["Mean " + component],
        solar_wind["Median " + component],
        color="indianred",
        label="Solar Wind",
        alpha=0.5,
        marker="+",
    )
    ax.scatter(
        magnetosheath["Mean " + component],
        magnetosheath["Median " + component],
        color="cornflowerblue",
        label="Magnetosheath",
        alpha=0.5,
        marker="+",
    )

    ax.set_xlabel("Mean " + component)
    ax.set_ylabel("Median " + component)
    
    ax.set_aspect("equal")

axes[0].legend()


plt.show()


spikes = combined_features.loc[ (combined_features["Mean |B|"] - combined_features["Median |B|"]) > 500 ]

print(spikes)
