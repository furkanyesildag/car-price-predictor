import argparse
import json
import joblib
import pandas as pd


def main() -> int:
	parser = argparse.ArgumentParser(description="Predict car price from JSON input")
	parser.add_argument("--model", default="models/car_price_pipeline.joblib")
	parser.add_argument("--input_json", help="Inline JSON or path to JSON file")
	args = parser.parse_args()

	model = joblib.load(args.model)

	# Determine if input_json is a path or a JSON string
	try:
		if args.input_json.endswith(".json"):
			with open(args.input_json, "r", encoding="utf-8") as f:
				data = json.load(f)
		else:
			data = json.loads(args.input_json)
	except Exception:
		print("Invalid input_json. Provide a JSON string or a .json file path.")
		return 1

	if isinstance(data, dict):
		records = [data]
	elif isinstance(data, list):
		records = data
	else:
		print("JSON must be an object or an array of objects.")
		return 1

	df = pd.DataFrame.from_records(records)
	# Ensure required columns exist; missing ones will be filled by pipeline imputers
	for col in ["brand", "series", "model", "year", "km"]:
		if col not in df.columns:
			df[col] = None

	pred = model.predict(df[["brand", "series", "model", "year", "km"]])
	for i, price in enumerate(pred):
		print(f"record {i}: predicted_price={int(price)}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())


