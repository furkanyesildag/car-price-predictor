import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def ensure_dir(path: str) -> None:
	if not os.path.exists(path):
		os.makedirs(path, exist_ok=True)


def load_and_clean(csv_path: str) -> pd.DataFrame:
	df = pd.read_csv(csv_path)
	df = df[df['price'].notnull()]
	df = df[(df['year'] >= 1980) & (df['year'] <= 2025)]
	df = df[(df['km'] >= 0) & (df['km'] <= 1_000_000)]
	df = df[(df['price'] > 0) & (df['price'] <= 20_000_000)]
	for c in ['brand','series','model']:
		df[c] = df[c].astype(str).replace({'nan':'Unknown'})
	return df


def plot_distributions(df: pd.DataFrame, out_dir: str) -> None:
	plt.figure(figsize=(12,4))
	plt.subplot(1,3,1)
	sns.histplot(df['year'], bins=30, kde=True)
	plt.title('Yıl dağılımı')
	plt.subplot(1,3,2)
	sns.histplot(df['km'], bins=30, kde=True)
	plt.title('KM dağılımı')
	plt.subplot(1,3,3)
	sns.histplot(df['price'], bins=30, kde=True)
	plt.title('Fiyat dağılımı')
	plt.tight_layout()
	plt.savefig(os.path.join(out_dir, 'distributions.png'), dpi=150)
	plt.close()


def plot_relationships(df: pd.DataFrame, out_dir: str) -> None:
	plt.figure(figsize=(12,5))
	plt.subplot(1,2,1)
	sns.regplot(data=df, x='year', y='price', scatter_kws={'alpha':0.2})
	plt.subplot(1,2,2)
	sns.regplot(data=df, x='km', y='price', scatter_kws={'alpha':0.2})
	plt.tight_layout()
	plt.savefig(os.path.join(out_dir, 'relationships.png'), dpi=150)
	plt.close()


def plot_top_brand_medians(df: pd.DataFrame, out_dir: str, top_n: int = 15) -> None:
	brand_med = (
		df.groupby('brand', as_index=False)['price']
		.median()
		.sort_values('price', ascending=False)
		.head(top_n)
	)
	plt.figure(figsize=(10,6))
	sns.barplot(data=brand_med, x='price', y='brand', orient='h')
	plt.xlabel('Medyan Fiyat')
	plt.ylabel('Marka')
	plt.title(f'En yüksek medyan fiyata sahip {top_n} marka')
	plt.tight_layout()
	plt.savefig(os.path.join(out_dir, 'top_brand_medians.png'), dpi=150)
	plt.close()


def main() -> int:
	csv_path = os.path.join('data', 'cars.csv')
	out_dir = os.path.join('reports')
	ensure_dir(out_dir)

	df = load_and_clean(csv_path)
	plot_distributions(df, out_dir)
	plot_relationships(df, out_dir)
	plot_top_brand_medians(df, out_dir)
	print(f"Saved plots to {out_dir}")
	return 0


if __name__ == '__main__':
	raise SystemExit(main())


