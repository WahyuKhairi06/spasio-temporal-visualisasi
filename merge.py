"""
merge.py  — Pipeline orchestrator
Jalankan script ini untuk mengeksekusi seluruh pipeline:
  1. preprocessing.py  — Bersihkan & sinkronisasi
  2. kmeans.py         — K-Means clustering & export

Cara pakai:
  cd spatio-dashboard/python
  python merge.py
"""
import subprocess, sys, os, time

BASE   = os.path.dirname(os.path.abspath(__file__))
PYTHON = sys.executable

STEPS = [
    ('Preprocessing & Sinkronisasi GeoJSON', os.path.join(BASE, 'preprocessing.py')),
    ('K-Means Clustering & Export JSON',     os.path.join(BASE, 'kmeans.py')),
]

def run(label, script):
    print(f"\n{'─'*60}")
    print(f"▶  {label}")
    print(f"   Script: {script}")
    print('─'*60)
    t0 = time.time()
    result = subprocess.run([PYTHON, script], capture_output=False)
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"\n❌ GAGAL: {label} (exit code {result.returncode})")
        sys.exit(1)
    print(f"   ⏱  Selesai dalam {elapsed:.1f} detik")

print("\n" + "═"*60)
print("  PIPELINE: Dashboard Spasio-Temporal IPM & Kemiskinan")
print("  Indonesia — 514 Kabupaten/Kota")
print("═"*60)
t_total = time.time()

for lbl, script in STEPS:
    run(lbl, script)

total = time.time() - t_total
print(f"\n{'═'*60}")
print(f"✅  Seluruh pipeline selesai dalam {total:.1f} detik")
print(f"   Buka index.html di browser untuk melihat dashboard.")
print("═"*60 + "\n")
