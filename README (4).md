# Real Estate Investment Advisor
1. `pip install -r requirements.txt`
2. Download `india_housing_prices.csv` (link in the brief) into this folder
3. `python train.py --data india_housing_prices.csv` (add `--sample 100000` for a faster run)
4. `streamlit run app.py`
5. `mlflow ui --backend-store-uri sqlite:///mlflow.db` to view experiments / registered models

`make_sample_data.py` makes a synthetic CSV with the same schema, for smoke tests only.
