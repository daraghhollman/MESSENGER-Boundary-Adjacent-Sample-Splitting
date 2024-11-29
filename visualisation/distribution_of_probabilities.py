"""
Script to investigate the distribution of probabilities output from the random forest
"""

import matplotlib.pyplot as plt
import pandas as pd

# Load random forest results.
results = pd.read_csv(
    "/home/daraghhollman/Main/Work/mercury/DataSets/random_forest_predictions.csv"
)
correct_results = results.loc[ results["Truth"] == results["Prediction"] ]
incorrect_results = results.loc[ results["Truth"] != results["Prediction"] ]

fig, ax = plt.subplots()

probability_keys = ["P(Solar Wind)", "P(Magnetosheath)"]

ax.hist(results[probability_keys[0]], density=True, color="black")
ax.set_xlabel(probability_keys[0])
ax.set_ylabel("# Observations per Bin")


plt.show()

fig, axes = plt.subplots(1, 2, sharey=True)

axis_labels = ["P(Solar Wind) (Correct)", "P(Solar Wind) (Misclassified)"]

for ax, label, sub_result in zip(axes, axis_labels, [correct_results, incorrect_results]):

    ax.hist(sub_result[probability_keys[0]], density=True, color="black")
    ax.set_xlabel(label)
    ax.margins(0)

    if ax == axes[0]:
        ax.set_ylabel("# Observations per Bin")


plt.show()
