# convert_csvs_to_sqlite.py
import pandas as pd
import sqlite3
import argparse
from pathlib import Path
import numpy as np

SQL_TYPE_MAP = {
    'int64': 'INTEGER',
    'float64': 'REAL',
    'object': 'TEXT',
    'bool': 'INTEGER',
    'datetime64[ns]': 'TEXT'
}

def df_to_sqlite(df: pd.DataFrame, db_path: str, table_name: str):
    conn = sqlite3.connect(db_path)
    # Convert datetimes to iso string
    for col in df.select_dtypes(include=['datetime64[ns]']).columns:
        df[col] = df[col].dt.strftime("%Y-%m-%dT%H:%M:%S")
    df.to_sql(table_name, conn, if_exists='replace', index=False)
    conn.close()
    print(f"Wrote {len(df):,} rows to {db_path} :: table {table_name}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--heart_csv', default='C:/Users/TARIF/multi_agent/data/heart.csv')
    parser.add_argument('--cancer_csv', default='C:/Users/TARIF/multi_agent/data/The_Cancer_data_1500_V2.csv')
    parser.add_argument('--diabetes_csv', default='C:/Users/TARIF/multi_agent/data/diabetes.csv')
    parser.add_argument('--out_dir', default='C:/Users/TARIF/multi_agent/output')

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mapping = [
        (args.heart_csv, out_dir / 'heart_disease.db', 'heart_disease'),
        (args.cancer_csv, out_dir / 'cancer.db', 'cancer'),
        (args.diabetes_csv, out_dir / 'diabetes.db', 'diabetes'),
    ]

    for csv_path, db_path, table_name in mapping:
        csv_path = Path(csv_path)
        if not csv_path.exists():
            print(f"File {csv_path} not found — skipping.")
            continue
        df = pd.read_csv(csv_path)
        # Quick cleaning: normalize column names
        df.columns = [c.strip().replace(' ', '_').lower() for c in df.columns]
        df_to_sqlite(df, str(db_path), table_name)

if __name__ == '__main__':
    main()
