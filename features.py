"""Shared cleaning / feature engineering / label logic (used by train.py and app.py)."""
from datetime import datetime
import numpy as np
import pandas as pd

CAT = ["State", "City", "Property_Type", "Furnished_Status", "Public_Transport_Accessibility",
       "Facing", "Owner_Type", "Availability_Status", "Security"]
NUM = ["BHK", "Size_in_SqFt", "Price_in_Lakhs", "Price_per_SqFt", "Age_of_Property", "Floor_No",
       "Total_Floors", "Nearby_Schools", "Nearby_Hospitals", "Parking_Space", "Amenity_Count",
       "Floor_Ratio", "Infra_Score", "City_PPS_Ratio", "City_Level"]
FEATURES = NUM + CAT
RAW_NUM = ["BHK", "Size_in_SqFt", "Price_in_Lakhs", "Year_Built", "Floor_No", "Total_Floors",
           "Age_of_Property", "Nearby_Schools", "Nearby_Hospitals", "Parking_Space"]
HORIZON = 5  # years


def transport_score(s):
    s = pd.Series(s).astype(str).str.lower()
    return np.select([s.str.contains("high"), s.str.contains("med")], [3, 2], 1)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Duplicates, missing values, type fixes, base engineered features."""
    df = df.drop_duplicates().copy()
    df = df.drop(columns=[c for c in ["ID"] if c in df.columns])
    for c in RAW_NUM:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "Year_Built" in df.columns:
        df["Age_of_Property"] = df["Age_of_Property"].fillna(datetime.now().year - df["Year_Built"])
    for c in CAT + ["Locality"]:
        if c in df.columns:
            df[c] = df[c].fillna("Unknown").astype(str).str.strip()
    for c in RAW_NUM:
        if c in df.columns:
            df[c] = df[c].fillna(df[c].median())
    df = df[(df["Size_in_SqFt"] > 0) & (df["Price_in_Lakhs"] > 0)].copy()
    df["Amenity_Count"] = df["Amenities"].fillna("").astype(str).apply(
        lambda x: len([a for a in x.split(",") if a.strip()])) if "Amenities" in df.columns else 0
    return df


def add_derived(df: pd.DataFrame, city_med: dict, overall_med: float) -> pd.DataFrame:
    """Features that depend on dataset-level stats (saved in meta.json so the app reuses them)."""
    df = df.copy()
    df["Price_per_SqFt"] = df["Price_in_Lakhs"] * 1e5 / df["Size_in_SqFt"]  # Rs per sq ft
    df["Floor_Ratio"] = df["Floor_No"] / df["Total_Floors"].clip(lower=1)
    df["Transport_Score"] = transport_score(df["Public_Transport_Accessibility"])
    df["Infra_Score"] = df["Nearby_Schools"] + df["Nearby_Hospitals"] + 2 * df["Transport_Score"]
    df["Is_Ready"] = df["Availability_Status"].astype(str).str.lower().str.contains("ready").astype(int)
    cm = df["City"].map(city_med).fillna(overall_med)
    df["City_PPS_Ratio"] = df["Price_per_SqFt"] / cm
    df["City_Level"] = np.clip(np.log(cm / overall_med), -1, 1)
    return df


def good_investment_label(df: pd.DataFrame, infra_median: float) -> pd.Series:
    """Multi-factor score (brief option 3): >=3 of 5 criteria -> Good Investment."""
    score = ((df["City_PPS_Ratio"] <= 1).astype(int) + (df["BHK"] >= 3).astype(int) + df["Is_Ready"]
             + (df["Transport_Score"] >= 2).astype(int) + (df["Infra_Score"] >= infra_median).astype(int))
    return (score >= 3).astype(int)


def growth_rate(df: pd.DataFrame, rng=None) -> pd.Series:
    """Assumed annual appreciation (dataset has no future prices). Feature-based + optional noise."""
    r = (0.05 + 0.006 * (df["Transport_Score"] - 1) + 0.01 * df["Is_Ready"]
         - 0.0004 * df["Age_of_Property"].clip(upper=50) + 0.01 * df["City_Level"])
    if rng is not None:
        r = r + rng.normal(0, 0.006, len(df))
    return r.clip(0.03, 0.14)


def future_price(price, rate, years=HORIZON):
    return price * (1 + rate) ** years
