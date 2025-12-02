import os
import json
import joblib
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score


def ensure_dir(path: str) -> None:
	if not os.path.exists(path):
		os.makedirs(path, exist_ok=True)


def load_data(csv_path: str) -> pd.DataFrame:
	df = pd.read_csv(csv_path)
	df = df[df['price'].notnull()]
	df = df[(df['year'] >= 1980) & (df['year'] <= 2025)]
	df = df[(df['km'] >= 0) & (df['km'] <= 1_000_000)]
	df = df[(df['price'] > 0) & (df['price'] <= 20_000_000)]
	for c in ['brand','series','model']:
		df[c] = df[c].astype(str).replace({'nan':'Unknown'})
	return df


def save_metrics_json(path: str, metrics: dict) -> None:
	with open(path, 'w', encoding='utf-8') as f:
		json.dump(metrics, f, ensure_ascii=False, indent=2)


def save_table_csv(path: str, df: pd.DataFrame) -> None:
	df.to_csv(path, index=False)


def plot_predictions(y_true: pd.Series, y_pred: np.ndarray, out_file: str) -> None:
	plt.figure(figsize=(6,6))
	plt.scatter(y_true, y_pred, alpha=0.2)
	minv, maxv = float(np.min(y_true)), float(np.max(y_true))
	plt.plot([minv, maxv], [minv, maxv], 'r--')
	plt.xlabel('Gerçek')
	plt.ylabel('Tahmin')
	plt.title('Gerçek vs Tahmin')
	plt.tight_layout()
	plt.savefig(out_file, dpi=150)
	plt.close()


def plot_error_hist(y_true: pd.Series, y_pred: np.ndarray, out_file: str) -> None:
	errors = y_pred - y_true
	plt.figure(figsize=(8,5))
	sns.histplot(errors, bins=50, kde=True)
	plt.title('Hata dağılımı (tahmin - gerçek)')
	plt.tight_layout()
	plt.savefig(out_file, dpi=150)
	plt.close()


def main() -> int:
	csv_path = os.path.join('data', 'cars.csv')
	model_path = os.path.join('models', 'car_price_pipeline.joblib')
	reports_dir = os.path.join('reports')
	ensure_dir(reports_dir)

	df = load_data(csv_path)
	features = ['brand','series','model','year','km']
	X = df[features]
	y = df['price']
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

	model = joblib.load(model_path)
	y_pred = model.predict(X_test)
	mae = mean_absolute_error(y_test, y_pred)
	r2 = r2_score(y_test, y_pred)

	metrics = { 'MAE': float(mae), 'R2': float(r2), 'n_test': int(len(y_test)) }
	save_metrics_json(os.path.join(reports_dir, 'metrics.json'), metrics)

	# Özet tablo: marka-seri düzeyinde medyan gerçek ve medyan tahmin
	result_df = X_test.copy()
	result_df = result_df.assign(true_price=y_test.values, pred_price=y_pred)
	summary = result_df.groupby(['brand','series'], as_index=False).agg(
		count=('pred_price','count'),
		median_true=('true_price','median'),
		median_pred=('pred_price','median'),
	)
	summary = summary.sort_values('count', ascending=False).head(50)
	save_table_csv(os.path.join(reports_dir, 'summary_brand_series.csv'), summary)

	# Görseller
	plot_predictions(y_test, y_pred, os.path.join(reports_dir, 'pred_vs_true.png'))
	plot_error_hist(y_test, y_pred, os.path.join(reports_dir, 'error_hist.png'))

	print('Saved reports to', reports_dir)
	print(metrics)
	return 0


if __name__ == '__main__':
	raise SystemExit(main())


