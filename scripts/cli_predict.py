import argparse
import os
import sys
import joblib
import pandas as pd


def load_model(model_path: str):
	return joblib.load(model_path)


def load_data(csv_path: str) -> pd.DataFrame:
	if not os.path.exists(csv_path):
		return pd.DataFrame()
	return pd.read_csv(csv_path)


def prompt_choice(options: list, title: str) -> str:
	if not options:
		raise ValueError("Seçenek bulunamadı")
	while True:
		print("")
		print(title)
		for idx, opt in enumerate(options, start=1):
			print(f"  {idx}. {opt}")
		raw = input("Seçiminiz (numara): ").strip()
		try:
			i = int(raw)
		except ValueError:
			print("Lütfen numara girin.")
			continue
		if 1 <= i <= len(options):
			return options[i - 1]
		print("Geçersiz seçim. Tekrar deneyin.")


def prompt_int(prompt: str, min_val: int = None, max_val: int = None) -> int:
	while True:
		text = input(prompt).strip()
		try:
			val = int(text)
		except ValueError:
			print("Lütfen sayı girin.")
			continue
		if min_val is not None and val < min_val:
			print(f"En az {min_val} olmalı.")
			continue
		if max_val is not None and val > max_val:
			print(f"En fazla {max_val} olmalı.")
			continue
		return val


def main() -> int:
	parser = argparse.ArgumentParser(description="Etkileşimli fiyat tahmini")
	parser.add_argument("--model", default=os.path.join("models", "car_price_pipeline.joblib"))
	parser.add_argument("--csv", default=os.path.join("data", "cars.csv"))
	args = parser.parse_args()

	try:
		model = load_model(args.model)
	except Exception as e:
		print(f"Model yüklenemedi: {e}")
		return 1

	df = load_data(args.csv)
	if df.empty:
		print("data/cars.csv bulunamadı ya da boş. Önce veri çıkarın.")
		return 1

	# Menü: marka → seri → model
	brands = sorted(df["brand"].dropna().astype(str).unique().tolist())
	brand = prompt_choice(brands, "Marka seçin:")

	series_options = (
		df.loc[df["brand"].astype(str).str.lower() == brand.lower(), "series"]
		.dropna().astype(str).unique().tolist()
	)
	series_options = sorted(series_options)
	series = prompt_choice(series_options, f"Seri seçin ({brand}):")

	model_options = (
		df.loc[
			(df["brand"].astype(str).str.lower() == brand.lower())
			& (df["series"].astype(str).str.lower() == series.lower()),
			"model",
		]
		.dropna().astype(str).unique().tolist()
	)
	model_options = sorted(model_options)
	model_name = prompt_choice(model_options, f"Model seçin ({brand} {series}):")
	year = prompt_int("Yıl (1980-2025): ", 1980, 2025)
	km = prompt_int("KM (0-1000000): ", 0, 1_000_000)

	row = {
		"brand": brand,
		"series": series,
		"model": model_name,
		"year": year,
		"km": km,
	}

	pred_df = pd.DataFrame([row])
	pred = model.predict(pred_df[["brand", "series", "model", "year", "km"]])[0]

	print("")
	print("— Tahmin —")
	print(f"Model tahmini: {int(pred):,} TL".replace(",", "."))

	if not df.empty:
		subset = df.copy()
		# Basit rehber: aynı marka/seri için medyan
		mask = (subset["brand"].astype(str).str.lower() == brand.lower()) & (
			subset["series"].astype(str).str.lower() == series.lower()
		)
		median_brand_series = subset.loc[mask, "price"].median() if mask.any() else None
		median_brand = subset.loc[
			subset["brand"].astype(str).str.lower() == brand.lower(), "price"
		].median()

		print("")
		print("— Referans —")
		if pd.notnull(median_brand_series):
			print(f"{brand} {series} medyan: {int(median_brand_series):,} TL".replace(",", "."))
		if pd.notnull(median_brand):
			print(f"{brand} geneli medyan: {int(median_brand):,} TL".replace(",", "."))

	print("")
	print("Not: Bu değerler ilan verilerine dayalı istatistiksel tahminlerdir.")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())


