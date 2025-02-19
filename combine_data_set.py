import pandas as pd

for boundary in ["Bow Shock", "Magnetopause"]:
    # Load the feature datasets
    if boundary == "Bow Shock":
        features_a = pd.read_csv(
            "/home/daraghhollman/Main/Work/mercury/DataSets/bow_shock/solar_wind_features.csv"
        )
        features_b = pd.read_csv(
            "/home/daraghhollman/Main/Work/mercury/DataSets/bow_shock/magnetosheath_features.csv"
        )

        features_a_label = "Solar Wind"
        features_b_label = "Magnetosheath"

        output_label = "bow_shock"

    elif boundary == "Magnetopause":
        features_a = pd.read_csv(
            "/home/daraghhollman/Main/Work/mercury/DataSets/magnetopause/magnetosphere_features.csv"
        )
        features_b = pd.read_csv(
            "/home/daraghhollman/Main/Work/mercury/DataSets/magnetopause/magnetosheath_features.csv"
        )

        features_a_label = "Magnetosphere"
        features_b_label = "Magnetosheath"

        output_label = "magnetopause"

    else:
        raise ValueError(f"Unknown boundary input: {boundary}. Options are 'Bow Shock' or 'Magnetopause'.")

    features = [
        "Mean",
        "Median",
        "Standard Deviation",
        "Skew",
        "Kurtosis",
        "Dip Statistic",
        "Dip P-Value",
        "Grazing Angle (deg.)",
        "Heliocentric Distance (AU)",
        "Local Time (hrs)",
        "Latitude (deg.)",
        "Magnetic Latitude (deg.)",
        "X MSM' (radii)",
        "Y MSM' (radii)",
        "Z MSM' (radii)",
        "Is Inbound?",
        "Sample Start",
        "Sample End",
    ]
    expanded_feature_labels = ["|B|", "Bx", "By", "Bz"]

    # Select only the columns we want to keep
    features_a = features_a[features].copy()
    features_b = features_b[features].copy()

    # Process each dataset
    for dataset in [features_a, features_b]:
        for feature in features[0:7]:

            # Convert elements from list-like strings to lists of floats
            dataset[feature] = dataset[feature].apply(
                lambda s: list(map(float, s.strip("[]").split()))
            )

            # Expand feature lists into new columns
            expanded_columns = (
                dataset[feature]
                .apply(pd.Series)
                .rename(lambda x: f"{feature} {expanded_feature_labels[x]}", axis=1)
            )

            # Assign new columns back to the original dataset
            dataset[expanded_columns.columns] = expanded_columns

        # Drop original feature columns
        dataset.drop(columns=features[0:7], inplace=True)

    features_a["label"] = features_a_label
    features_b["label"] = features_b_label

    combined_features = pd.concat([features_a, features_b], ignore_index=True)

    combined_features.to_csv(
        f"/home/daraghhollman/Main/Work/mercury/DataSets/{output_label}/combined_features.csv"
    )
