"""
Are there connected strings of missclassifications? If so, how many are long
"""

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load random forest predictions
predictions = pd.read_csv(
    "/home/daraghhollman/Main/Work/mercury/DataSets/random_forest_predictions.csv"
)

# Filter by label
# incorrect_predictions = incorrect_predictions.loc[ incorrect_predictions["Truth"] == "Magnetosheath" ]

current_string_length = 0
for _, prediction in predictions
