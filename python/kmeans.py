"""
kmeans.py
Melakukan K-Means clustering berdasarkan rata-rata IPM & Kemiskinan
dan menghasilkan GeoJSON + JSON yang siap digunakan dashboard.
"""
import pandas as pd
import numpy as np
import json
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# ────────────────────────────────────────────────────────────────
# PATHS
# ────────────────────────────────────────────────────────────────
BASE   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA   = os.path.join(BASE, 'data')
GEO_IN = os.path.join(DATA, 'all_kabkota_ind.geojson')   # original (sudah dibersihkan)
CSV_IN = os.path.join(DATA, 'dataset_processed.csv')      # output preprocessing.py

# ────────────────────────────────────────────────────────────────
# 1. LOAD
# ────────────────────────────────────────────────────────────────
print("📂 Loading processed data...")
df = pd.read_csv(CSV_IN)
with open(GEO_IN, encoding='utf-8') as f:
    geojson = json.load(f)

import re

def clean_name(name):
    if pd.isna(name):
        return name
    s = str(name).strip().upper()
    s = re.sub(r'^(KABUPATEN|KAB\.?|KOTA)\s+', '', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

for feat in geojson['features']:
    p = feat['properties']
    if 'name_clean' not in p:
        p['name_clean'] = clean_name(p['name'])

# ────────────────────────────────────────────────────────────────
# 1.5 Convert "-" to NaN for math
# ────────────────────────────────────────────────────────────────
df_calc = df.copy()
df_calc['ipm'] = pd.to_numeric(df_calc['ipm'].replace("-", np.nan), errors='coerce')
df_calc['kemiskinan'] = pd.to_numeric(df_calc['kemiskinan'].replace("-", np.nan), errors='coerce')

# ────────────────────────────────────────────────────────────────
# 2. PIVOT — satu baris per wilayah
# ────────────────────────────────────────────────────────────────
pivot = df_calc.groupby('geo_name_clean').agg(
    wilayah        = ('wilayah', 'first'),
    ipm_mean       = ('ipm',        'mean'),
    ipm_latest     = ('ipm',        lambda x: x.dropna().iloc[-1] if len(x.dropna())>0 else np.nan),
    kemiskinan_mean= ('kemiskinan', 'mean'),
    kemiskinan_latest=('kemiskinan',lambda x: x.dropna().iloc[-1] if len(x.dropna())>0 else np.nan),
    ipm_trend      = ('ipm',        lambda x: round(x.dropna().iloc[-1] - x.dropna().iloc[0], 2) if len(x.dropna())>1 else 0),
    kemiskinan_trend=('kemiskinan', lambda x: round(x.dropna().iloc[-1] - x.dropna().iloc[0], 2) if len(x.dropna())>1 else 0),
).reset_index()

print(f"   Pivot shape: {pivot.shape}")

# ────────────────────────────────────────────────────────────────
# 3. K-MEANS (k=4)
# ────────────────────────────────────────────────────────────────
FEATURES = ['ipm_mean', 'kemiskinan_mean', 'ipm_trend', 'kemiskinan_trend']
valid_mask = pivot[FEATURES].notna().all(axis=1)
valid_pivot = pivot[valid_mask].copy()

X        = valid_pivot[FEATURES].values
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)

np.random.seed(42)
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10, max_iter=300)
valid_pivot['cluster_raw'] = kmeans.fit_predict(X_scaled)
pivot['cluster_raw'] = np.nan
pivot.loc[valid_mask, 'cluster_raw'] = valid_pivot['cluster_raw']

# ── Beri label semantis: makin tinggi IPM & rendah kemiskinan = Klaster 1
cs = valid_pivot.groupby('cluster_raw')[['ipm_mean', 'kemiskinan_mean']].mean()
score = cs['ipm_mean'] - cs['kemiskinan_mean']
rank  = score.rank(ascending=False).astype(int)
label_map   = {idx: f"Klaster {rank[idx]}" for idx in rank.index}
color_map   = {
    label_map[cs.index[rank[cs.index]==1][0]]: '#22c55e',   # Klaster 1 – terbaik – hijau
    label_map[cs.index[rank[cs.index]==2][0]]: '#3b82f6',   # Klaster 2 – biru
    label_map[cs.index[rank[cs.index]==3][0]]: '#f59e0b',   # Klaster 3 – kuning
    label_map[cs.index[rank[cs.index]==4][0]]: '#ef4444',   # Klaster 4 – terburuk – merah
}
pivot['cluster']       = pivot['cluster_raw'].map(label_map).fillna('No Data')
pivot['cluster_color'] = pivot['cluster'].map(color_map).fillna('#9ca3af')

print("\n📊 Klaster Centroid (mean):")
for lbl in sorted(label_map.values()):
    sub = pivot[pivot['cluster']==lbl]
    print(f"   {lbl}: n={len(sub):3d}, IPM={sub['ipm_mean'].mean():.2f}, "
          f"Kemiskinan={sub['kemiskinan_mean'].mean():.2f}")

# ────────────────────────────────────────────────────────────────
# 4. SIMPAN cluster_summary.csv
# ────────────────────────────────────────────────────────────────
pivot.drop(columns='cluster_raw').to_csv(
    os.path.join(DATA, 'cluster_summary.csv'), index=False)

# ────────────────────────────────────────────────────────────────
# 5. ENRICH GEOJSON
# ────────────────────────────────────────────────────────────────
cdict = pivot.set_index('geo_name_clean').to_dict(orient='index')

def format_val(v):
    return round(float(v), 2) if pd.notna(v) else "-"

for feat in geojson['features']:
    nc = feat['properties'].get('name_clean', '')
    if nc in cdict:
        c = cdict[nc]
        feat['properties'].update({
            'cluster'          : c['cluster'],
            'cluster_color'    : c['cluster_color'],
            'ipm_mean'         : format_val(c['ipm_mean']),
            'ipm_latest'       : format_val(c['ipm_latest']),
            'kemiskinan_mean'  : format_val(c['kemiskinan_mean']),
            'kemiskinan_latest': format_val(c['kemiskinan_latest']),
            'ipm_trend'        : format_val(c['ipm_trend']),
            'kemiskinan_trend' : format_val(c['kemiskinan_trend']),
            'wilayah'          : c['wilayah'],
        })
    else:
        feat['properties'].update({
            'cluster':'No Data','cluster_color':'#9ca3af',
            'ipm_mean':"-",'ipm_latest':"-",
            'kemiskinan_mean':"-",'kemiskinan_latest':"-",
            'ipm_trend':"-",'kemiskinan_trend':"-",'wilayah':nc,
        })

with open(os.path.join(DATA, 'all_kabkota_clustered.geojson'), 'w', encoding='utf-8') as f:
    json.dump(geojson, f, ensure_ascii=False)

# ────────────────────────────────────────────────────────────────
# 6. SIMPAN timeseries.json (untuk chart detail)
# ────────────────────────────────────────────────────────────────
ts = {}
for _, row in df.iterrows():
    nm  = row['geo_name_clean']
    yr  = int(row['tahun'])
    if nm not in ts:
        ts[nm] = {
            'wilayah': row['wilayah'],
            'years'  : [], 'ipm': [], 'kemiskinan': [],
            'cluster': cdict.get(nm, {}).get('cluster', 'No Data'),
            'cluster_color': cdict.get(nm, {}).get('cluster_color', '#9ca3af'),
        }
    ts[nm]['years'].append(yr)
    
    val_ipm = row['ipm']
    ts[nm]['ipm'].append(round(float(val_ipm), 2) if pd.notna(val_ipm) and val_ipm != "-" else "-")
    
    val_kem = row['kemiskinan']
    ts[nm]['kemiskinan'].append(round(float(val_kem), 2) if pd.notna(val_kem) and val_kem != "-" else "-")

with open(os.path.join(DATA, 'timeseries.json'), 'w', encoding='utf-8') as f:
    json.dump(ts, f, ensure_ascii=False)

# ────────────────────────────────────────────────────────────────
# 7. STATS.JSON (ringkasan untuk kartu statistik)
# ────────────────────────────────────────────────────────────────
latest_year = int(df_calc['tahun'].max())
df_latest   = df_calc[df_calc['tahun'] == latest_year]

stats = {
    'total_wilayah'     : int(pivot.shape[0]),
    'latest_year'       : latest_year,
    'ipm_nasional'      : round(float(df_latest['ipm'].mean()), 2),
    'kemiskinan_nasional': round(float(df_latest['kemiskinan'].mean()), 2),
    'ipm_tertinggi'     : {
        'nama' : pivot.loc[pivot['ipm_latest'].idxmax(), 'wilayah'],
        'nilai': round(float(pivot['ipm_latest'].max()), 2)},
    'ipm_terendah'      : {
        'nama' : pivot.loc[pivot['ipm_latest'].idxmin(), 'wilayah'],
        'nilai': round(float(pivot['ipm_latest'].min()), 2)},
    'kemiskinan_tertinggi': {
        'nama' : pivot.loc[pivot['kemiskinan_latest'].idxmax(), 'wilayah'],
        'nilai': round(float(pivot['kemiskinan_latest'].max()), 2)},
    'cluster_counts'    : pivot['cluster'].value_counts().to_dict(),
    'years'             : sorted(df_calc['tahun'].unique().tolist()),
    'color_map'         : color_map,
}

with open(os.path.join(DATA, 'stats.json'), 'w', encoding='utf-8') as f:
    json.dump(stats, f, ensure_ascii=False, indent=2)

print(f"\n💾 Output tersimpan di: {DATA}")
print("   ✓ dataset_processed.csv")
print("   ✓ cluster_summary.csv")
print("   ✓ all_kabkota_clustered.geojson")
print("   ✓ timeseries.json")
print("   ✓ stats.json")
print("\n✅ kmeans.py selesai!\n")
