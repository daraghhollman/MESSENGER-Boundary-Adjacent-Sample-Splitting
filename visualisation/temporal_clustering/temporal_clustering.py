"""
A script to investigate the temporal spread of misclassified samples.
It aims to answer the question: if one sample is misclassified, are
subsequent samples more likely to be misclassified.
"""

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Import sample features
combined_features = pd.read_csv(
    "/home/daraghhollman/Main/Work/mercury/DataSets/combined_features.csv",
)

# Load random forest predictions
predictions = pd.read_csv(
    "/home/daraghhollman/Main/Work/mercury/DataSets/random_forest_predictions.csv"
)

correct_predictions = predictions.loc[predictions["Truth"] == predictions["Prediction"]]
incorrect_predictions = predictions.loc[
    predictions["Truth"] != predictions["Prediction"]
]

# Filter by label
incorrect_predictions = incorrect_predictions.loc[ incorrect_predictions["Truth"] == "Magnetosheath" ]

misclassified_indices = incorrect_predictions[incorrect_predictions.columns[0]]
misclassified_start_times = combined_features["Sample Start"].iloc[
    misclassified_indices
]
misclassified_start_times = pd.to_datetime(
    misclassified_start_times, format="ISO8601"
).tolist()

dates = mdates.date2num(misclassified_start_times)
dates.sort()

time_between_misclassifications = np.roll(dates, -1)[:-1] - dates[:-1]
time_between_misclassifications = mdates.num2timedelta(time_between_misclassifications)
# Convert to days
time_between_misclassifications = [
    t.total_seconds() / 86400 for t in time_between_misclassifications
]

bin_size = 5  # days
plt.hist(
    time_between_misclassifications,
    color="black",
    bins=np.arange(0, np.max(time_between_misclassifications), bin_size),
    density=True
)

plt.xlabel("Days Between Subsequent Misclassifications")
plt.ylabel("Fraction of observations per bin")

plt.margins(0)

plt.show()
