"""
preprocessing.py
Membersihkan nama wilayah, menangani missing value, sinkronisasi dengan GeoJSON.
"""
import pandas as pd
import numpy as np
import json
import re
import os

# ────────────────────────────────────────────────────────────────
# PATHS
# ────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_IN    = os.path.join(BASE, 'data', 'dataset_leaflet_ready.csv')
GEO_IN    = os.path.join(BASE, 'data', 'all_kabkota_ind.geojson')
CSV_OUT   = os.path.join(BASE, 'data', 'dataset_processed.csv')

# ────────────────────────────────────────────────────────────────
# 1. LOAD DATA
# ────────────────────────────────────────────────────────────────
print("📂 Loading data...")
df = pd.read_csv(CSV_IN)
with open(GEO_IN, encoding='utf-8') as f:
    geojson = json.load(f)

print(f"   CSV  : {df.shape[0]:,} rows, {df['wilayah'].nunique()} wilayah")
print(f"   GeoJSON: {len(geojson['features'])} features")

# ────────────────────────────────────────────────────────────────
# 2. BERSIHKAN NAMA WILAYAH (hapus KABUPATEN / KOTA)
# ────────────────────────────────────────────────────────────────
def clean_name(name: str) -> str:
    """Hapus prefiks KABUPATEN/KAB./KOTA dan normalisasi ke UPPER."""
    if pd.isna(name):
        return name
    s = str(name).strip().upper()
    s = re.sub(r'^(KABUPATEN|KAB\.?|KOTA)\s+', '', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

df['wilayah_clean']  = df['wilayah'].apply(clean_name)
df['geo_name_clean'] = df['geo_name'].apply(clean_name)

# Bersihkan nama di GeoJSON juga
for feat in geojson['features']:
    p = feat['properties']
    p['name_clean']     = clean_name(p['name'])
    p['alt_name_clean'] = clean_name(p.get('alt_name', p['name']))

print("\n✅ Nama wilayah dibersihkan (KABUPATEN/KOTA dihapus)")

# ────────────────────────────────────────────────────────────────
# 3. SINKRONISASI DENGAN GEOJSON
# ────────────────────────────────────────────────────────────────
geo_clean_set = {f['properties']['name_clean'] for f in geojson['features']}
csv_names     = set(df['geo_name_clean'].unique())

matched   = csv_names & geo_clean_set
unmatched = csv_names - geo_clean_set

print(f"\n🗺️  Sinkronisasi GeoJSON:")
print(f"   Matched  : {len(matched)} wilayah")
print(f"   Unmatched: {len(unmatched)} wilayah")
if unmatched:
    print(f"   Contoh   : {sorted(unmatched)[:10]}")

# Hanya simpan baris yang tersinkronisasi
df = df[df['geo_name_clean'].isin(geo_clean_set)].copy()
print(f"   Setelah filter: {df.shape[0]:,} rows, {df['geo_name_clean'].nunique()} wilayah")

# ────────────────────────────────────────────────────────────────
# 4. TANGANI MISSING VALUES
# ────────────────────────────────────────────────────────────────
print(f"\n⚠️  Missing values sebelum imputasi:")
print(f"   ipm        : {df['ipm'].isna().sum()}")
print(f"   kemiskinan : {df['kemiskinan'].isna().sum()}")

df = df.sort_values(['geo_name_clean', 'tahun']).reset_index(drop=True)

# Interpolasi linear per wilayah
for col in ['ipm', 'kemiskinan']:
    df[col] = df.groupby('geo_name_clean')[col].transform(
        lambda x: x.interpolate(method='linear', limit_direction='both'))

# Sisa (awal/akhir seri) diisi median global
for col in ['ipm', 'kemiskinan']:
    df[col] = df[col].fillna(df[col].median())

print(f"✅ Missing values setelah imputasi:")
print(f"   ipm        : {df['ipm'].isna().sum()}")
print(f"   kemiskinan : {df['kemiskinan'].isna().sum()}")

# ────────────────────────────────────────────────────────────────
# 5. SIMPAN
# ────────────────────────────────────────────────────────────────
df.to_csv(CSV_OUT, index=False)
print(f"\n💾 Tersimpan → {CSV_OUT}")
print("✅ preprocessing.py selesai!\n")
