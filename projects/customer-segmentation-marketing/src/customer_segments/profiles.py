"""Aggregate observed profiles and descriptive, non-causal segment names."""
import numpy as np
import pandas as pd

SPEND_COLUMNS = ["MntWines", "MntFruits", "MntMeatProducts", "MntFishProducts", "MntSweetProducts", "MntGoldProds"]


def summarize(frame, labels):
    observed = frame.copy()
    observed["total_spend"] = observed[SPEND_COLUMNS].sum(axis=1, min_count=len(SPEND_COLUMNS))
    observed["segment"] = labels
    columns = ["Income", "total_spend", "NumDealsPurchases", "Recency"]
    profiles = observed.groupby("segment")[columns].mean()
    profiles["customers"] = observed.groupby("segment").size()
    profiles["share"] = profiles["customers"] / len(frame)
    profiles["income_observed"] = observed.groupby("segment")["Income"].count()
    spending_midpoint = profiles["total_spend"].median()
    deal_midpoint = profiles["NumDealsPurchases"].median()
    profiles["description"] = [
        f"Segment {segment}: {'higher' if row.total_spend >= spending_midpoint else 'lower'} spending, "
        f"{'more' if row.NumDealsPurchases >= deal_midpoint else 'fewer'} discounted purchases"
        for segment, row in profiles.iterrows()
    ]
    return profiles.reset_index()


def assignments(bundle, frame):
    pipeline = bundle["pipeline"]
    projected = pipeline[:-1].transform(frame)
    labels = pipeline.named_steps["kmeans"].predict(projected)
    probabilities = bundle["gmm"].predict_proba(projected)
    sorted_probabilities = np.sort(probabilities, axis=1)
    gap = sorted_probabilities[:, -1] - sorted_probabilities[:, -2]
    descriptions = {int(row["segment"]): row["description"] for row in bundle["profiles"]}
    result = pd.DataFrame({
        "row_number": np.arange(len(frame)), "kmeans_segment": labels,
        "segment_description": [descriptions[int(label)] for label in labels],
        "gmm_component": probabilities.argmax(axis=1),
        "gmm_max_probability": probabilities.max(axis=1),
        "gmm_probability_gap": gap,
        "gmm_boundary": gap <= bundle["config"]["boundary_margin"],
    })
    for index in range(probabilities.shape[1]):
        result[f"gmm_probability_{index}"] = probabilities[:, index]
    return result
