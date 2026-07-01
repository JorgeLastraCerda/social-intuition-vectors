# Faydalı Tricks

## Checkpointing (Uzun İşler İçin)

Uzun süren simülasyonlar için checkpointing kullan — bağlantı kopsa veya sistem hata verse kaldığın yerden devam edersin.

```bash
dmtcp_checkpoint -b -i 21600 -c checkpoint_dir/ python script.py
```

- `-i 21600` → Her 6 saatte bir checkpoint yazar
- `-b` → Koordinatör aynı CPU'da çalışır
- `-c` → Checkpoint dosyalarının yazılacağı klasör

Devam ettirmek için:
```bash
./checkpoint_dir/dmtcp_restart_script.sh
```

**Not:** Sadece serial ve OpenMP işler desteklenir. MPI işlerde çalışmaz.

---

## Modül Komutları

```bash
module load conda                  # Anaconda yükle
source activate python-3.13        # Python ortamını aktif et
module list                        # Yüklü modülleri gör
module purge                       # Tüm modülleri kaldır
module avail numlib                # Belirli kategoride ne var
module whatis mkl                  # Modül hakkında kısa bilgi
```

Kalıcı yapmak için `~/.bashrc` dosyasına ekle:
```bash
module load conda
source activate python-3.13
```

---

## JupyterHub'a Kendi Ortamını Ekleme

```bash
module load conda
source activate python-3.13
python -m ipykernel install --user --name python-3.13 --display-name "Python (3.13)"
```

JupyterHub'da "Python (3.13)" kernel'ı olarak görünür.

---

## İnteraktif Session

Uzun terminal işleri için direkt node'a bağlan:
```bash
qlogin -q scc
```

---

## İşten Sonra Kaynak Kullanımını Kontrol Et

```bash
qacct -j <jobid>
```

Gerçekte ne kadar RAM ve süre kullandığını gösterir. Sonraki jobları buna göre ayarla — fazla istemek sıra bekleme süresini uzatır.

---

## Dosya Sıkıştırma

Büyük veriyi saklarken sıkıştır, disk kotasını korursun:

```bash
# Hızlı ve iyi sıkıştırma
plzip -9 dosya.dat              # Tek dosya
tar --lzip -cf arsiv.tar.lz veri/  # Klasör

# En iyi sıkıştırma (yavaş)
zpaq -m 5 a arsiv.zpaq veri/
```

gzip ve zip nispeten zayıf sıkıştırır, büyük veriler için plzip tercih et.
