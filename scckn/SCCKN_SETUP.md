# SCCKN Cluster - Kurulum ve Kullanım Rehberi

## Dökümanlar

| Dosya | İçerik |
|-------|--------|
| [SCCKN_SETUP.md](SCCKN_SETUP.md) | Bu dosya — kurulum adımları |
| [JOB_SUBMISSION.md](JOB_SUBMISSION.md) | Job gönderme, queue'lar, array jobs, Python ortamı |
| [STORAGE.md](STORAGE.md) | Disk kotası, dizinler, dosya transferi |
| [TIPS.md](TIPS.md) | Checkpointing, modüller, JupyterHub kernel ekleme |
| [RULES.md](RULES.md) | Kullanım kuralları, limitler, yayın teşekkürü |
| [GPU_JOB_DESIGN.md](GPU_JOB_DESIGN.md) | Cross-project GPU pipeline design, priority, dependencies, and recovery |
| [hybrid_gpu_submit_template.sh](hybrid_gpu_submit_template.sh) | Reusable two-resource-class submitter template |
| [staged_gpu_runner_template.sh](staged_gpu_runner_template.sh) | Resumable staged-job runner template |

---

## Hesap Bilgileri

| Alan | Değer |
|------|-------|
| Kullanıcı adı | emrecan.ulu |
| Frontend 1 | scc.uni-konstanz.de |
| Frontend 2 | scc2.uni-konstanz.de |
| Home dizini | /home/scc/emrecan.ulu |
| JupyterHub | https://scc2.uni-konstanz.de |
| Sistem Yöneticisi | Stefan.Gerlach@uni-konstanz.de |

---

## Cluster Kaynakları

Scheduler: **SGE (Son of Grid Engine)**

### Queue Tipleri

| Queue | Ne için | Tipik Node Kapasitesi |
|-------|---------|----------------------|
| **scc** | Normal CPU işleri | 20–128 core/node |
| **long** | Uzun süreli işler (saatler/günler) | Aynı node'lar |
| **gpu** | GPU gerektiren işler | scc192: 140 slot, scc213: 80 slot |
| **old** | Eski node'lar | 6–48 core |

### Öne Çıkan Node'lar (CPU/RAM işleri için)

| Node | Core | Notlar |
|------|------|--------|
| scc200–scc212 | 128 core/node | En güçlü CPU node'ları |
| scc180–scc189 | 40–48 core/node | Dengeli seçenek |
| scc150–scc177 | 32 core/node | Çok sayıda node, genellikle dolu |

Frontend (giriş makinesi): 80 thread, 754 GB RAM — doğrudan iş çalıştırma, sadece giriş/hazırlık için.

---

## 1. SSH Key Kurulumu

Şifre girmeden cluster'a bağlanmak için SSH key oluşturuldu.

```bash
# Key oluştur (local Mac'te)
ssh-keygen -t ed25519 -C "scckn"

# Public key'i cluster'a gönder
ssh-copy-id emrecan.ulu@scc.uni-konstanz.de
```

---

## 2. SSH Config

`~/.ssh/config` dosyasına eklendi:

```
Host scckn
    HostName scc.uni-konstanz.de
    User emrecan.ulu
    IdentityFile ~/.ssh/id_ed25519
```

Artık bağlanmak için sadece şunu yazmak yeterli:

```bash
ssh scckn
```

---

## 3. Dosya Transferi (Cyberduck)

macFUSE/sshfs M4 Mac'te kernel extension gerektirdiği için tercih edilmedi. Cyberduck kullanılıyor.

### Kurulum

[cyberduck.io](https://cyberduck.io) — ücretsiz indir ve kur.

### Bağlantı Ayarları

- Open Connection → **SFTP** seç
- Server: `scc.uni-konstanz.de`
- Username: `emrecan.ulu`
- SSH Private Key: `~/.ssh/id_ed25519`
- Connect

Finder gibi çalışır, sürükle bırak ile dosya yükle/indir.

---

## 4. JupyterHub

Tarayıcıdan erişim: [https://scc2.uni-konstanz.de](https://scc2.uni-konstanz.de)

- Aynı kullanıcı adı ve şifre ile giriş yapılır
- Cyberduck ile yüklenen dosyalar burada otomatik görünür (aynı home dizini)

---

## 5. Yazılım Modülleri

Cluster'da modüller `module` komutuyla yüklenir.

```bash
module load conda                  # Anaconda'yı yükle
source activate python-3.13        # Python 3.13 ortamını aktif et (424 paket dahil)
```

Sampling, matrix ve clustering işleri için `python-3.13` ortamı yeterli — numpy, scipy, scikit-learn hepsi dahil.

Detaylar için: [JOB_SUBMISSION.md](JOB_SUBMISSION.md)

---

## 6. Job Gönderme (SGE)

### Jupyter Notebook → Job

Notebook'u doğrudan queue'ya gönderemezsin. İki yol var:

**Yol 1 — Notebook'u script'e çevir (önerilen):**
```bash
# Cluster'da
jupyter nbconvert --to script notebook.ipynb
# notebook.py oluşur, onu qsub ile gönder
qsub job_template.sh
```

**Yol 2 — Notebook'u olduğu gibi çalıştır:**
```bash
# job_template.sh içinde şu satırı kullan:
jupyter nbconvert --to notebook --execute notebook.ipynb --output sonuc.ipynb
```

### Job Script Şablonu (`job_template.sh`)

```bash
#!/bin/bash
#$ -N job_ismi          # Job adı
#$ -q scc               # Queue: scc (normal) veya long (uzun işler)
#$ -pe smp 8            # Kaç CPU core
#$ -l h_vmem=16G        # Core başına RAM (toplam = 8 x 16G = 128G)
#$ -l h_rt=12:00:00     # Max süre (ss:dd:sn)
#$ -o output.log        # Çıktı dosyası
#$ -e error.log         # Hata dosyası
#$ -cwd                 # Scriptin bulunduğu dizinde çalıştır

python script.py
# veya:
# jupyter nbconvert --to notebook --execute notebook.ipynb --output sonuc.ipynb
```

### Temel Komutlar

```bash
qsub job_template.sh        # Job gönder
qstat -u emrecan.ulu        # Kendi joblarını gör
qdel <job_id>               # Job iptal et
```

---

## 6. Uzun Süreli İşler için tmux

SSH bağlantısı kopsa bile işlemin devam etmesi için:

```bash
tmux new -s ana_oturum      # Yeni session aç
# Ctrl+B ardından D          # Detach (bağlantıyı arka plana al)
tmux attach -t ana_oturum   # Geri dön
tmux ls                     # Session listesi
```

---

## Hızlı Başvuru

```bash
ssh scckn                        # Cluster'a bağlan
exit                             # Cluster'dan çık
qsub job_template.sh             # Job gönder
qstat -u emrecan.ulu             # Jobları gör
qdel <job_id>                    # Job iptal et
jupyter nbconvert --to script notebook.ipynb  # Notebook → Python script
tmux new -s isim                 # Yeni tmux session
tmux attach -t isim              # Var olan session'a dön
```
