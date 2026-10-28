"""Synthetic stand-in with the same schema as india_housing_prices.csv (for smoke tests only)."""
import argparse, numpy as np, pandas as pd
p = argparse.ArgumentParser(); p.add_argument("--n", type=int, default=20000); p.add_argument("--out", default="india_housing_prices.csv")
a = p.parse_args(); rng = np.random.default_rng(0); n = a.n
cities = {"Maharashtra": ["Mumbai", "Pune", "Nagpur"], "Karnataka": ["Bangalore", "Mysore"], "Delhi": ["New Delhi"],
          "Madhya Pradesh": ["Bhopal", "Indore"], "Tamil Nadu": ["Chennai", "Coimbatore"], "Telangana": ["Hyderabad"]}
base = {"Mumbai": 3.0, "Pune": 1.4, "Nagpur": .7, "Bangalore": 1.8, "Mysore": .8, "New Delhi": 2.2, "Bhopal": .6,
        "Indore": .7, "Chennai": 1.3, "Coimbatore": .7, "Hyderabad": 1.5}
st = rng.choice(list(cities), n); city = np.array([rng.choice(cities[s]) for s in st])
size = rng.integers(500, 4500, n); bhk = np.clip((size // 700) + rng.integers(0, 2, n), 1, 5)
pps = np.array([base[c] for c in city]) * rng.lognormal(-3.2, .35, n) * 1.0  # lakhs per sqft
price = (pps * size).round(2)
df = pd.DataFrame({"ID": range(1, n + 1), "State": st, "City": city,
    "Locality": [f"Locality_{i}" for i in rng.integers(1, 60, n)],
    "Property_Type": rng.choice(["Apartment", "Villa", "Independent House"], n), "BHK": bhk, "Size_in_SqFt": size,
    "Price_in_Lakhs": price, "Price_per_SqFt": (price / size).round(4), "Year_Built": rng.integers(1990, 2025, n),
    "Furnished_Status": rng.choice(["Unfurnished", "Semi-furnished", "Furnished"], n),
    "Floor_No": rng.integers(0, 30, n), "Total_Floors": rng.integers(1, 35, n),
    "Nearby_Schools": rng.integers(1, 11, n), "Nearby_Hospitals": rng.integers(1, 11, n),
    "Public_Transport_Accessibility": rng.choice(["Low", "Medium", "High"], n), "Parking_Space": rng.integers(0, 4, n),
    "Security": rng.choice(["Yes", "No"], n),
    "Amenities": [", ".join(rng.choice(["Gym", "Pool", "Clubhouse", "Garden", "Playground"], rng.integers(0, 5), replace=False)) for _ in range(n)],
    "Facing": rng.choice(["North", "South", "East", "West"], n), "Owner_Type": rng.choice(["Owner", "Builder", "Broker"], n),
    "Availability_Status": rng.choice(["Ready_to_Move", "Under_Construction"], n)})
df["Age_of_Property"] = 2025 - df["Year_Built"]
df.loc[rng.choice(n, n // 50, replace=False), "Nearby_Schools"] = np.nan
df = pd.concat([df, df.sample(100, random_state=1)])
df.to_csv(a.out, index=False); print("wrote", a.out, df.shape)
