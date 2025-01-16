"""
A script to investigate the temporal spread of misclassified samples.
Looking at the time series of probabilities for correct and incorrect samples.
"""

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.cluster
from sklearn.metrics import silhouette_score

cluster = "none"

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
direction = "outbound"
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

fig, ax = plt.subplots()


match cluster:
    case "none":
        plt.scatter(
            correctly_classified_start_times,
            correct_predictions["Correctness"],
            color="black",
            marker=".",
        )

        plt.scatter(
            misclassified_start_times,
            incorrect_predictions["Correctness"],
            color="indianred",
            marker=".",
        )

    case "kmeans":

        check_number_of_clusters = False

        if check_number_of_clusters:

            fits = []
            scores = []
            k_values = range(2, 20)
            for k in k_values:
                data = np.reshape(mdates.date2num(misclassified_start_times), (-1, 1))
                model = sklearn.cluster.KMeans(n_clusters=k).fit(data)

                fits.append(model)

                scores.append(silhouette_score(data, model.labels_, metric="euclidean"))

            plt.plot(k_values, scores)

            plt.show()
            fig, ax = plt.subplots()

        num_clusters = 7
        kmeans = sklearn.cluster.KMeans(n_clusters=num_clusters)

        incorrect_predictions["KMeans Cluster"] = kmeans.fit_predict(
            np.reshape(mdates.date2num(misclassified_start_times), (-1, 1))
        )

        plt.scatter(
            correctly_classified_start_times,
            correct_predictions["Correctness"],
            color="grey",
        )

        for i in range(num_clusters):
            plt.scatter(
                np.array(misclassified_start_times)[
                    incorrect_predictions["KMeans Cluster"] == i
                ],
                incorrect_predictions["Correctness"][
                    incorrect_predictions["KMeans Cluster"] == i
                ],
            )

    case "mean shift":

        data = np.reshape(mdates.date2num(misclassified_start_times), (-1, 1))

        mean_shift = sklearn.cluster.MeanShift(bin_seeding=True)
        mean_shift.fit(data)
        labels = mean_shift.labels_

        cluster_centers = mean_shift.cluster_centers_

        labels_unique = np.unique(labels)
        n_clusters_ = len(labels_unique)

        print(n_clusters_)


plt.ylim(0, 1)
plt.margins(0)
plt.xlabel("Date")
plt.ylabel("P ( Truth )")

plt.show()


fig, ax = plt.subplots()
for i in range(len(misclassified_start_times)):
    ax.axvline(
        misclassified_start_times[i],
        color="indianred",
    )
ax.set_yticks([])
plt.show()
