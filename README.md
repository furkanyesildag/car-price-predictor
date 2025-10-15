# Araba Fiyat Tahmini - Hızlı Başlangıç (Windows PowerShell)

## 1) Sanal ortam ve kurulum
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -U pip
pip install -r requirements.txt
```

## 2) Metin dosyalarından CSV üret
Metin dosyalarınız proje kökünde `*.txt` veya `*.tx` şeklinde ise:
```powershell
python scripts/extract_to_csv.py --output data/cars.csv
```
Belirli dosyaları vermek isterseniz:
```powershell
python scripts/extract_to_csv.py --inputs "2.metinbelgesi.tx" "Yeni Metin Belgesi (2).txt" --output data/cars.csv
```

## 3) Modeli eğit
```powershell
python scripts/train_model.py --csv data/cars.csv --out models/car_price_pipeline.joblib
```
Örnek çıktı: `MAE` ve `R2` skorları konsola yazdırılır, model `models/` altına kaydedilir.

## 4) Tahmin al
Tek kayıt (inline JSON):
```powershell
python scripts/predict.py --model models/car_price_pipeline.joblib --input_json '{"brand":"Renault","series":"Clio","model":"1.0 SCe Joy","year":2023,"km":80000}'
```
Dosyadan (JSON dosyası bir dizi/tekil obje olabilir):
```powershell
python scripts/predict.py --model models/car_price_pipeline.joblib --input_json sample.json
```

## Notlar
- Veri şeması `brand, series, model, year, km, price` alanlarına dayanır. `price` yalnızca eğitimde kullanılır.
- Aşırı uç değerleri temel filtrelerle temizliyoruz. İsterseniz mantığı `scripts/train_model.py` içinde düzenleyebilirsiniz.
- Eğer metin JSON formatınızda `cars` dizisi farklı bir isimdeyse `scripts/extract_to_csv.py` içinde güncelleyin.

