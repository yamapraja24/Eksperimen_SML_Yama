import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from pathlib import Path
import joblib
import argparse



BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR.parent / "Energy_raw"
OUT_DIR = BASE_DIR / "energy_preprocessing"
TARGET = "Energy Consumption"

def load_data(raw_dir: Path) -> pd.DataFrame:
    files = sorted(Path(raw_dir).glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"Tidak ada file CSV di {raw_dir}")
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

def preprocess_data(df):
    
    df_clean = df.copy()
    
    # 1. Hapus Duplikasi Data
    df_clean = df_clean.drop_duplicates()

    # 2. Identifikasi Kolom Berdasarkan Tipe Data
    numerical_cols = df_clean.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = df_clean.select_dtypes(include=['object', 'category']).columns

    # 3. Imputasi Missing Values (Numerik -> Median)
    for col in numerical_cols:
        median_val = df_clean[col].median()
        df_clean[col] = df_clean[col].fillna(median_val)

    # 4. Imputasi Missing Values (Kategorik -> Mode)
    for col in categorical_cols:
        if len(df_clean[col].mode()) > 0:
            mode_val = df_clean[col].mode()[0]
            df_clean[col] = df_clean[col].fillna(mode_val)

    # 5. Standarisasi Variabel Numerik (Fitur Prediktor)
    numerical_features = [
        "Square Footage",
        "Number of Occupants",
        "Appliances Used",
        "Average Temperature"
    ]
    
    scaler = StandardScaler()
    df_clean[numerical_features] = scaler.fit_transform(df_clean[numerical_features])

    # 6. One-Hot Encoding Variabel Kategorik
    cat_to_encode = ['Building Type', 'Day of Week']
    encoder = OneHotEncoder(drop='first', sparse_output=False, dtype=int)
    
    encoded_array = encoder.fit_transform(df_clean[cat_to_encode])
    encoded_col_names = encoder.get_feature_names_out(cat_to_encode)
    
    df_encoded_cols = pd.DataFrame(encoded_array, columns=encoded_col_names, index=df_clean.index)

    # 7. Penggabungan Kembali Menjadi Dataframe Final
    df_final = df_clean.drop(columns=cat_to_encode)
    df_final = pd.concat([df_final, df_encoded_cols], axis=1)
    
    return df_final, scaler, encoder

def main(raw_dir: Path = RAW_DIR, out_dir: Path = OUT_DIR) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(raw_dir)
    df_ready, trained_scaler, trained_encoder = preprocess_data(df)

    df_ready.to_csv(out_dir / "data_preprocessed.csv", index=False)
    print(f"Selesai. data_preprocessed={df_ready.shape}-> {out_dir}")

    

if __name__ == "__main__":
    print("--- Memulai Proses Preprocessing Data ---")
    
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("--raw-dir", default=str(RAW_DIR))
        parser.add_argument("--out-dir", default=str(OUT_DIR))
        args = parser.parse_args()
        main(Path(args.raw_dir), Path(args.out_dir))
        
    except Exception as e:
        print(f"[Error] Terjadi kesalahan saat menjalankan script: {e}")