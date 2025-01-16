"""
Plots to show the spatial distribution of the misclassified samples in comparison with correctly classified
"""

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import pandas as pd
import hermpy.plotting as hermplot

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

# Load random forest predictions
predictions = pd.read_csv(
    "/home/daraghhollman/Main/Work/mercury/DataSets/random_forest_predictions.csv"
)

solar_wind_predictions = predictions.loc[ predictions["Truth"] == "Solar Wind" ]
magnetosheath_predictions = predictions.loc[ predictions["Truth"] == "Magnetosheath" ]

for region_predictions in [solar_wind_predictions, magnetosheath_predictions]:

    correct_predictions = region_predictions.loc[region_predictions["Truth"] == region_predictions["Prediction"]]
    incorrect_predictions = region_predictions.loc[
        region_predictions["Truth"] != region_predictions["Prediction"]
    ]


    fig, axes = plt.subplots(1, 2)

    (xy_axis, xz_axis) = axes

    correct_samples = all_samples.iloc[correct_predictions.iloc[:, 0]]
    incorrect_samples = all_samples.iloc[incorrect_predictions.iloc[:, 0]]

    # We want to find the ratio of classified to misclassified samples per spatial bin
    size = 8  # radii
    bins = size * 2
    correct_histogram_xy, x_edges, y_edges = np.histogram2d(
        correct_samples["X MSM' (radii)"],
        correct_samples["Y MSM' (radii)"],
        range=[[-size, size], [-size, size]],
        bins=bins,
    )
    incorrect_histogram_xy, _, _ = np.histogram2d(
        incorrect_samples["X MSM' (radii)"],
        incorrect_samples["Y MSM' (radii)"],
        range=[[-size, size], [-size, size]],
        bins=bins,
    )

    correct_histogram_xz, _, z_edges = np.histogram2d(
        correct_samples["X MSM' (radii)"],
        correct_samples["Z MSM' (radii)"],
        range=[[-size, size], [-size, size]],
        bins=bins,
    )
    incorrect_histogram_xz, _, _ = np.histogram2d(
        incorrect_samples["X MSM' (radii)"],
        incorrect_samples["Z MSM' (radii)"],
        range=[[-size, size], [-size, size]],
        bins=bins,
    )

    ratio_histogram_xy = incorrect_histogram_xy / correct_histogram_xy
    ratio_histogram_xz = incorrect_histogram_xz / correct_histogram_xz

    pcolor_xy = xy_axis.pcolormesh(x_edges, y_edges, ratio_histogram_xy.T)
    pcolor_xz = xz_axis.pcolormesh(x_edges, z_edges, ratio_histogram_xz.T)

    for ax, image, plane in zip(axes, [pcolor_xy, pcolor_xz], ["xy", "xz"]):

        ax.set_aspect("equal")
        ax.set_xlim(-size, size)
        ax.set_ylim(-size, size)

        hermplot.Plot_Mercury(
            ax, frame="MSM'", plane=plane, shaded_hemisphere="left", alpha=0
        )
        hermplot.Add_Labels(ax, frame="MSM'", plane=plane)

        hermplot.Plot_Magnetospheric_Boundaries(ax, zorder=5)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("bottom", size="5%", pad=0.6)

        fig.colorbar(
            image,
            cax=cax,
            orientation="horizontal",
            label="(# Misclassified / # Correctly Classified) per bin",
        )


    plt.show()

    # Cylindrical plot
    fig, ax = plt.subplots()

    correct_histogram_cyl, _, _ = np.histogram2d(
        correct_samples["X MSM' (radii)"],
        np.sqrt(
            correct_samples["Z MSM' (radii)"] ** 2 + correct_samples["Y MSM' (radii)"] ** 2
        ),
        range=[[-size, size], [-size, size]],
        bins=bins,
    )
    incorrect_histogram_cyl, _, _ = np.histogram2d(
        incorrect_samples["X MSM' (radii)"],
        np.sqrt(
            incorrect_samples["Z MSM' (radii)"] ** 2
            + incorrect_samples["Y MSM' (radii)"] ** 2
        ),
        range=[[-size, size], [-size, size]],
        bins=bins,
    )

    show_dots = False
    if show_dots:
        ax.scatter(
            correct_samples["X MSM' (radii)"],
            np.sqrt(
                correct_samples["Z MSM' (radii)"] ** 2 + correct_samples["Y MSM' (radii)"] ** 2
            ),
            color="blue",
            zorder=10
        )
        ax.scatter(
            incorrect_samples["X MSM' (radii)"],
            np.sqrt(
                incorrect_samples["Z MSM' (radii)"] ** 2
                + incorrect_samples["Y MSM' (radii)"] ** 2
            ),
            color="red",
            zorder=10
        )

    ratio_histogram_cyl = incorrect_histogram_cyl / correct_histogram_cyl

    image = ax.pcolormesh(x_edges, y_edges, ratio_histogram_cyl.T)

    hermplot.Format_Cylindrical_Plot(ax, 5)
    hermplot.Plot_Magnetospheric_Boundaries(ax, zorder=5)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("bottom", size="5%", pad=0.6)

    fig.colorbar(
        image,
        cax=cax,
        orientation="horizontal",
        label="(# Misclassified / # Correctly Classified) per bin",
    )

    ax.set_title(f"Random forest misclassification ratio for {region_predictions.iloc[0]['Truth']} samples (N={len(correct_samples) + len(incorrect_samples)})\n adjacent to Philpott bow shock crossing intervals")

    plt.show()
