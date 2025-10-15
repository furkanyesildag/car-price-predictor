import argparse
import os
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor


def load_data(csv_path: str) -> pd.DataFrame:
	return pd.read_csv(csv_path)


def build_pipeline(df: pd.DataFrame) -> Pipeline:
	feature_cols = [
		"brand",
		"series",
		"model",
		"year",
		"km",
	]
	target_col = "price"

	categorical_features = ["brand", "series", "model"]
	numeric_features = ["year", "km"]

	categorical_transformer = Pipeline(
		steps=[
			("imputer", SimpleImputer(strategy="most_frequent")),
			("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
		]
	)

	numeric_transformer = Pipeline(
		steps=[
			("imputer", SimpleImputer(strategy="median")),
			("scaler", StandardScaler(with_mean=False)),
		]
	)

	preprocessor = ColumnTransformer(
		transformers=[
			("cat", categorical_transformer, categorical_features),
			("num", numeric_transformer, numeric_features),
		]
	)

	model = RandomForestRegressor(
		n_estimators=300,
		random_state=42,
		n_jobs=-1,
	)

	pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
	return pipeline


def main() -> int:
	parser = argparse.ArgumentParser(description="Train car price model")
	parser.add_argument("--csv", default=os.path.join("data", "cars.csv"))
	parser.add_argument("--out", default=os.path.join("models", "car_price_pipeline.joblib"))
	args = parser.parse_args()

	df = load_data(args.csv)
	df = df[df["price"].notnull()]
	# Basic sanity filters
	df = df[(df["year"] >= 1980) & (df["year"] <= 2025)]
	df = df[(df["km"] >= 0) & (df["km"] <= 1_000_000)]
	df = df[(df["price"] > 0) & (df["price"] <= 20_000_000)]

	X = df[["brand", "series", "model", "year", "km"]]
	y = df["price"]

	X_train, X_test, y_train, y_test = train_test_split(
		X, y, test_size=0.2, random_state=42
	)

	pipeline = build_pipeline(df)
	pipeline.fit(X_train, y_train)

	pred = pipeline.predict(X_test)
	mae = mean_absolute_error(y_test, pred)
	r2 = r2_score(y_test, pred)
	print(f"MAE: {mae:.0f}")
	print(f"R2: {r2:.3f}")

	os.makedirs(os.path.dirname(args.out), exist_ok=True)
	joblib.dump(pipeline, args.out)
	print(f"Saved model to {args.out}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())


