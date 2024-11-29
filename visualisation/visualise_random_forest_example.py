"""
Script to index a list of random forest outputs and plot the time series and distrbution.
"""

import datetime as dt

import hermpy.mag as mag
import hermpy.plotting_tools as hermplot
import hermpy.boundary_crossings as boundaries
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import spiceypy as spice


def main():
    colours = ["#648FFF", "#785EF0", "#DC267F", "#FE6100", "#FFB000"]

    spice.furnsh("/home/daraghhollman/Main/SPICE/messenger/metakernel_messenger.txt")
    crossings = boundaries.Load_Crossings("/home/daraghhollman/Main/Work/mercury/DataSets/philpott_2020.xlsx")

    # Load the csv files
    magnetosheath_samples = pd.read_csv(
        "/home/daraghhollman/Main/Work/mercury/DataSets/magnetosheath_sample_10_mins.csv"
    )
    solar_wind_samples = pd.read_csv(
        "/home/daraghhollman/Main/Work/mercury/DataSets/solar_wind_sample_10_mins.csv"
    )

    total_samples = pd.concat((solar_wind_samples, magnetosheath_samples))

    components = ["|B|", "Bx", "By", "Bz"]
    for component in components:
        total_samples[component] = total_samples[component].apply(
            lambda x: list(map(float, x.strip("[]").split(",")))
        )

    results = pd.read_csv(
        "/home/daraghhollman/Main/Work/mercury/DataSets/random_forest_predictions.csv"
    )

    # results = results.loc[ abs(results["P(Solar Wind)"] - results["P(Magnetosheath)"]) <= 0.5 ]
    # results = results.loc[results["Truth"] == "Magnetosheath"]
    # results = results.loc[results["Truth"] != results["Prediction"]]

    for _, row in results.iterrows():

        selected_result = row
        i = row.iloc[0]

        # We create two axes side-by-side
        # and afterwards, we add an axis below to represent the probability
        fig = plt.figure()

        histogram_axis = plt.subplot(2, 2, 1)
        mag_axis = plt.subplot(2, 2, 2)
        probability_axis = plt.subplot(2, 1, 2)

        histogram_axis.hist(
            total_samples.iloc[i]["|B|"],
            color="black",
            orientation="horizontal",
            histtype="step",
            lw=2,
        )
        histogram_axis.hist(
            total_samples.iloc[i]["Bx"],
            color=colours[2],
            orientation="horizontal",
            histtype="step",
            lw=2,
        )
        histogram_axis.hist(
            total_samples.iloc[i]["By"],
            color=colours[0],
            orientation="horizontal",
            histtype="step",
            lw=2,
        )
        histogram_axis.hist(
            total_samples.iloc[i]["Bz"],
            color=colours[-1],
            orientation="horizontal",
            histtype="step",
            lw=2,
        )
        mag_axis.sharey(histogram_axis)

        # Plot the area around the sample
        time_buffer = dt.timedelta(minutes=30)

        try:
            sample_start = dt.datetime.strptime(total_samples.iloc[i]["Sample Start"], "%Y-%m-%d %H:%M:%S.%f")
            sample_end = dt.datetime.strptime(total_samples.iloc[i]["Sample End"], "%Y-%m-%d %H:%M:%S.%f")
        except:
            sample_start = dt.datetime.strptime(total_samples.iloc[i]["Sample Start"], "%Y-%m-%d %H:%M:%S")
            sample_end = dt.datetime.strptime(total_samples.iloc[i]["Sample End"], "%Y-%m-%d %H:%M:%S")

        # Load the new data
        data = mag.Load_Between_Dates(
            "/home/daraghhollman/Main/data/mercury/messenger/mag/avg_1_second/",
            sample_start - time_buffer,
            sample_end + time_buffer,
            strip=True
        )

        mag_axis.plot(
            data["date"],
            data["|B|"],
            color="black",
            lw=1,
            label="|B|",
            alpha=0.8,
        )
        mag_axis.plot(
            data["date"],
            data["Bx"],
            color=colours[2],
            lw=1,
            label="Bx",
            alpha=0.8,
        )
        mag_axis.plot(
            data["date"],
            data["By"],
            color=colours[0],
            lw=1,
            label="By",
            alpha=0.8,
        )
        mag_axis.plot(
            data["date"],
            data["Bz"],
            color=colours[-1],
            lw=1,
            label="Bz",
            alpha=0.8,
        )

        mag_axis.axvspan(sample_start, sample_end, color="grey", alpha=0.2)
        boundaries.Plot_Crossing_Intervals(mag_axis, data["date"].iloc[0], data["date"].iloc[-1], crossings)
        hermplot.Add_Tick_Ephemeris(mag_axis)

        mag_axis.legend(
            bbox_to_anchor=(0.5, 1.1), loc="center", ncol=4, borderaxespad=0.5
        )

        histogram_axis.set_xlabel("# Seconds")
        histogram_axis.set_ylabel("Field Strength [nT]")

        Number_Line(probability_axis)
        probability_axis.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
        probability_axis.xaxis.set_minor_locator(ticker.MultipleLocator(0.01))

        probability_axis.scatter(
            selected_result["P(Solar Wind)"], 0, color="indianred", zorder=5
        )

        probability_axis.set_xlabel("Solar Wind Sample Probability")
        probability_axis.text(
            1.05, 0, "Solar Wind", va="center", ha="center", rotation=90
        )
        probability_axis.text(
            -0.05, 0, "Magnetosheath", va="center", ha="center", rotation=90
        )

        fig.suptitle(
            f"Truth: {selected_result['Truth']}    Prediction: {selected_result['Prediction']}"
        )

        plt.show()


# https://matplotlib.org/2.0.2/examples/ticks_and_spines/tick-locators.html
# Setup a plot such that only the bottom spine is shown
def Number_Line(ax):
    ax.spines["right"].set_color("none")
    ax.spines["left"].set_color("none")
    ax.yaxis.set_major_locator(ticker.NullLocator())
    ax.spines["top"].set_color("none")
    ax.xaxis.set_ticks_position("bottom")
    ax.tick_params(which="major", width=1.00)
    ax.tick_params(which="major", length=5)
    ax.tick_params(which="minor", width=0.75)
    ax.tick_params(which="minor", length=2.5)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.1, 1)
    ax.patch.set_alpha(0.0)


if __name__ == "__main__":
    main()
