import json
from pathlib import Path
import joblib, numpy as np, pandas as pd, plotly.express as px, plotly.graph_objects as go, streamlit as st
from features import FEATURES, HORIZON, add_derived, future_price

ROOT = Path(__file__).parent / "models"
st.set_page_config(page_title="Real Estate Investment Advisor", page_icon="🏠", layout="wide")


@st.cache_resource
def load():
    return (joblib.load(ROOT / "classifier.joblib"), joblib.load(ROOT / "regressor.joblib"),
            json.load(open(ROOT / "meta.json")))


@st.cache_data
def load_sample():
    return pd.read_csv(ROOT / "sample.csv.gz")


if not (ROOT / "meta.json").exists():
    st.error("No trained models found. Run `python train.py --data india_housing_prices.csv` first."); st.stop()
clf, reg, meta = load(); data = load_sample(); opt = meta["options"]

st.title("🏠 Real Estate Investment Advisor")
st.caption("Is it a good investment, and what will it be worth in 5 years?")
tab1, tab2, tab3 = st.tabs(["🔮 Predict", "🔎 Explore & Filter", "📊 Model Insights"])

# ------------------------------------------------------------------ Predict
with tab1:
    c1, c2, c3 = st.columns(3)
    with c1:
        state = st.selectbox("State", sorted(meta["city_by_state"]))
        city = st.selectbox("City", meta["city_by_state"][state])
        ptype = st.selectbox("Property type", opt["Property_Type"])
        bhk = st.number_input("BHK", 1, 10, 3)
        size = st.number_input("Size (sq ft)", 100, 20000, 1200, step=50)
        price = st.number_input("Price (₹ Lakhs)", 1.0, 100000.0, 90.0, step=5.0)
    with c2:
        year = st.number_input("Year built", 1950, 2030, 2015)
        furn = st.selectbox("Furnished status", opt["Furnished_Status"])
        floor = st.number_input("Floor no.", 0, 100, 5); tfloors = st.number_input("Total floors", 1, 100, 12)
        parking = st.number_input("Parking spaces", 0, 10, 1)
        avail = st.selectbox("Availability", opt["Availability_Status"])
    with c3:
        schools = st.slider("Nearby schools", 0, 10, 5); hosp = st.slider("Nearby hospitals", 0, 10, 5)
        transport = st.selectbox("Public transport access", opt["Public_Transport_Accessibility"])
        security = st.selectbox("Security", opt["Security"]); amen = st.slider("No. of amenities", 0, 10, 3)
        facing = st.selectbox("Facing", opt["Facing"]); owner = st.selectbox("Owner type", opt["Owner_Type"])

    if st.button("Analyse property", type="primary"):
        row = pd.DataFrame([{"State": state, "City": city, "Property_Type": ptype, "BHK": bhk, "Size_in_SqFt": size,
            "Price_in_Lakhs": price, "Age_of_Property": max(0, 2026 - year), "Furnished_Status": furn, "Floor_No": floor,
            "Total_Floors": tfloors, "Nearby_Schools": schools, "Nearby_Hospitals": hosp,
            "Public_Transport_Accessibility": transport, "Parking_Space": parking, "Security": security,
            "Amenity_Count": amen, "Facing": facing, "Owner_Type": owner, "Availability_Status": avail}])
        row = add_derived(row, meta["city_median_pps"], meta["overall_median_pps"])[FEATURES]
        prob = float(clf.predict_proba(row)[0, 1]); rate = float(reg.predict(row)[0])
        fut = float(future_price(price, rate)); gain = fut - price

        a, b, c = st.columns(3)
        a.metric("Verdict", "✅ Good Investment" if prob >= 0.5 else "❌ Not a Good Investment")
        b.metric(f"Estimated price after {HORIZON} yrs", f"₹ {fut:,.1f} L", f"{gain:+,.1f} L ({rate:.1%}/yr)")
        c.metric("Model confidence", f"{max(prob, 1 - prob):.0%}")
        g = go.Figure(go.Indicator(mode="gauge+number", value=prob * 100, title={"text": "Good-investment probability (%)"},
            gauge={"axis": {"range": [0, 100]}, "bar": {"color": "green" if prob >= .5 else "crimson"},
                   "threshold": {"line": {"color": "black", "width": 3}, "value": 50}}))
        g.update_layout(height=260, margin=dict(t=50, b=0))
        t = pd.DataFrame({"Year": list(range(0, HORIZON + 1)), "Price (₹ L)": [future_price(price, rate, y) for y in range(HORIZON + 1)]})
        l, r = st.columns(2)
        l.plotly_chart(g, width="stretch")
        r.plotly_chart(px.line(t, x="Year", y="Price (₹ L)", markers=True, title="Projected price trend"), width="stretch")
        imp = pd.Series(meta["importance_clf"]).sort_values()
        st.plotly_chart(px.bar(imp, orientation="h", title="Top factors driving the classification",
                               labels={"value": "importance", "index": ""}), width="stretch")
        st.caption("⚠️ Educational tool. The 5-year price uses an assumed appreciation model, not a market guarantee.")

# ------------------------------------------------------------------ Explore
with tab2:
    f1, f2, f3, f4 = st.columns(4)
    states = f1.multiselect("State", sorted(data.State.unique()))
    cities = f2.multiselect("City", sorted(data[data.State.isin(states)].City.unique() if states else data.City.unique()))
    bhks = f3.multiselect("BHK", sorted(data.BHK.unique().astype(int)))
    only_good = f4.checkbox("Good investments only")
    p_lo, p_hi = float(data.Price_in_Lakhs.quantile(.01)), float(data.Price_in_Lakhs.quantile(.99))
    s_lo, s_hi = int(data.Size_in_SqFt.min()), int(data.Size_in_SqFt.max())
    pr = st.slider("Price (₹ Lakhs)", p_lo, p_hi, (p_lo, p_hi)); sz = st.slider("Area (sq ft)", s_lo, s_hi, (s_lo, s_hi))
    d = data[data.Price_in_Lakhs.between(*pr) & data.Size_in_SqFt.between(*sz)]
    if states: d = d[d.State.isin(states)]
    if cities: d = d[d.City.isin(cities)]
    if bhks: d = d[d.BHK.isin(bhks)]
    if only_good: d = d[d.Good_Investment == 1]
    st.write(f"**{len(d):,}** matching properties")
    st.dataframe(d.sort_values("Price_per_SqFt").head(300), width="stretch", hide_index=True)
    if len(d):
        x, y = st.columns(2)
        top = d.groupby("City").Price_in_Lakhs.mean().sort_values(ascending=False).head(15).reset_index()
        x.plotly_chart(px.bar(top, x="City", y="Price_in_Lakhs", title="Average price by city (₹ L)"), width="stretch")
        piv = d.pivot_table(index="State", columns="Property_Type", values="Price_per_SqFt", aggfunc="mean")
        y.plotly_chart(px.imshow(piv, aspect="auto", color_continuous_scale="YlOrRd", title="Avg ₹/sq ft: State × Property type"), width="stretch")
        x, y = st.columns(2)
        x.plotly_chart(px.scatter(d.sample(min(3000, len(d)), random_state=1), x="Size_in_SqFt", y="Price_in_Lakhs",
                                  color="Property_Type", opacity=.5, title="Size vs price"), width="stretch")
        tr = d.assign(Age=pd.cut(d.Age_of_Property, [-1, 5, 10, 20, 30, 100])).groupby("Age", observed=True).Price_per_SqFt.mean().reset_index()
        tr["Age"] = tr["Age"].astype(str)
        y.plotly_chart(px.line(tr, x="Age", y="Price_per_SqFt", markers=True, title="₹/sq ft by property age (yrs)"), width="stretch")

# ------------------------------------------------------------------ Insights
with tab3:
    st.subheader("Classification — Good Investment")
    ct = pd.DataFrame({m: v["test"] for m, v in meta["classification"].items()}).T
    st.dataframe(ct.style.format("{:.4f}").highlight_max(axis=0, color="#c8f7c5"), width="stretch")
    st.success(f"Best model: **{meta['best_classifier']}** (selected on validation F1)")
    cm = np.array(meta["classification"][meta["best_classifier"]]["confusion_matrix"])
    st.plotly_chart(px.imshow(cm, text_auto=True, x=["Pred: No", "Pred: Yes"], y=["Actual: No", "Actual: Yes"],
                              color_continuous_scale="Blues", title="Confusion matrix (test set)"), width="stretch")
    st.subheader("Regression — 5-year price")
    rt = pd.DataFrame({m: v["test"] for m, v in meta["regression"].items()}).T
    st.dataframe(rt.style.format("{:.4f}").highlight_min(subset=["rmse_price", "mae_price", "rmse_rate"], color="#c8f7c5"), width="stretch")
    st.success(f"Best model: **{meta['best_regressor']}** (selected on validation RMSE)")
    imp = pd.Series(meta["importance_reg"]).sort_values()
    st.plotly_chart(px.bar(imp, orientation="h", title="Top growth drivers (regressor)", labels={"value": "importance", "index": ""}), width="stretch")
