# Job Gönderme Rehberi (Grid Engine)

## Queue Tipleri ve Limitler

| Queue | Max Süre | Kullanım |
|-------|----------|---------|
| **scc** | 10 gün | Genel CPU işleri |
| **long** | 120 gün | Uzun süreli işler |
| **old** | 30 gün | Eski node'lar |
| **gpu** | 10 gün | GPU işleri |

Default süre 7 gün. Belirtmezsen otomatik 7 gün verilir.

---

## Temel Job Script

```bash
#!/bin/bash
#$ -N job_ismi           # Job adı (qstat'ta görünür)
#$ -q scc                # Queue seçimi
#$ -pe smp 8             # Kaç CPU core (shared memory)
#$ -l h_vmem=16G         # Core başına RAM (toplam = 8 x 16G = 128G)
#$ -l h_rt=24:00:00      # Max süre (ss:dd:sn)
#$ -o output.log         # Çıktı dosyası
#$ -e error.log          # Hata dosyası
#$ -cwd                  # Scriptin bulunduğu dizinde çalıştır

module load conda
source activate python-3.13

python script.py
```

**Mac'te script yazarken dikkat:** Dosya formatı CRLF olmamalı. Cluster'a yükleyince şunu çalıştır:
```bash
dos2unix job.sh
```

---

## Python Ortamı

```bash
module load conda               # Anaconda'yı yükle
conda env list                  # Mevcut ortamları listele
source activate python-3.13     # Python 3.13 ortamını aktif et (424 paket dahil)
```

Jupyter'da kullanmak istersen:
```bash
module load conda
source activate python-3.13
python -m ipykernel install --user --name python-3.13 --display-name "Python (3.13)"
```

---

## Notebook'u Job Olarak Çalıştırma

```bash
#!/bin/bash
#$ -N notebook_job
#$ -q scc
#$ -pe smp 8
#$ -l h_vmem=16G
#$ -l h_rt=12:00:00
#$ -cwd

module load conda
source activate python-3.13

jupyter nbconvert --to notebook --execute notebook.ipynb --output sonuc.ipynb
```

---

## Array Jobs (Çok Sayıda Benzer İş)

Sampling veya parametre taraması yapıyorsan array job kullan — tek script ile yüzlerce iş gönderebilirsin.

```bash
#!/bin/bash
#$ -N sampling_array
#$ -q scc
#$ -pe smp 4
#$ -l h_vmem=8G
#$ -l h_rt=6:00:00
#$ -t 1-100            # 100 ayrı iş (SGE_TASK_ID = 1,2,...,100)
#$ -tc 20              # Aynı anda max 20 tanesi çalışır
#$ -cwd

module load conda
source activate python-3.13

python script.py $SGE_TASK_ID   # Her iş farklı ID ile çalışır
```

Tek bir task'ı silmek için:
```bash
qdel <Job-ID> -t <Task-ID>
```

---

## Temel Komutlar

```bash
qsub job.sh                  # Job gönder
qstat -u emrecan.ulu         # Kendi joblarını gör
qstat -f -u \*               # Tüm kullanıcıların jobları
qdel 1234                    # Job sil
qalter -q long               # Çalışmayan jobu farklı queue'ya taşı
qlogin -q scc                # İnteraktif session aç
qacct -j 1234                # Biten job'un kaynak kullanımını gör
qstat -j 1234                # Hatalı job'un detayını gör
```

---

## Job Hata Verirse

Job "E" durumuna düşerse:
```bash
qstat -j <jobid>    # Hata sebebini göster
```

Job reddedilirse ("no suitable queues"):
```bash
qsub -w v job.sh    # Neden reddedildiğini göster
```

---

## E-posta Bildirimi

```bash
#$ -m bea                           # b=başlarken, e=bitince, a=hata olunca
#$ -M emrecan.ulu@uni-konstanz.de   # Mail adresi
```

---

## İş Bittikten Sonra Kaynak Kontrolü

```bash
qacct -j <jobid>    # Gerçekte ne kadar RAM/süre kullandığını gösterir
```

Sonraki joblar için h_vmem ve h_rt'yi buna göre ayarla.
