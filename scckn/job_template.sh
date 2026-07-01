#!/bin/bash
#$ -N job_ismi          # Job adı (qstat'ta görünür)
#$ -q scc               # Hangi queue: scc (normal) veya long (uzun işler)
#$ -pe smp 8            # Kaç CPU core kullanacaksın
#$ -l h_vmem=16G        # Core başına RAM (toplam = 8 x 16G = 128G)
#$ -l h_rt=12:00:00     # Max çalışma süresi (saat:dakika:saniye)
#$ -o output.log        # Standart çıktı dosyası
#$ -e error.log         # Hata çıktısı dosyası
#$ -cwd                 # Job'u scriptin bulunduğu dizinde çalıştır

# --- Ortam kurulumu ---
# module load python/3.x  # Gerekirse Python modülünü yükle

# --- Çalıştırılacak komut ---
python script.py

# Notebook çalıştırmak istersen:
# jupyter nbconvert --to notebook --execute notebook.ipynb --output sonuc.ipynb
