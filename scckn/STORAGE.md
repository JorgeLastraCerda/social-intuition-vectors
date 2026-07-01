# Veri Depolama

## Kota Kontrolü

```bash
/software/bin/quota    # Mevcut disk kullanımını gösterir
```

---

## Dizinler ve Kotalar

| Dizin | Kısayol | Boyut | Kota | Backup | Kullanım |
|-------|---------|-------|------|--------|---------|
| `/home/scc/emrecan.ulu` | `$HOME` | 11 TB toplam | **100 GB** | Günlük | Ayarlar, kodlar |
| `/work` | `/work` | 640 TB | Yok | Aylık | Aktif veri |
| `/localscratch` | `/scratch` | 1-3 TB/node | Yok | **Yok** | Geçici dosyalar |

**Önemli:** Home dizinine büyük veri koyma. İşlemler için `/work` kullan.

---

## Dosya Transferi

Cluster'a/Cluster'dan veri taşımak için her zaman `rsync` kullan (`cp` veya `mv` değil).

```bash
# Mac → Cluster
rsync -avhSPz emrecan.ulu@scc.uni-konstanz.de:/work/emrecan.ulu/ ./yerel/

# Cluster → Mac
rsync -avhSPz ./yerel/ emrecan.ulu@scc.uni-konstanz.de:/work/emrecan.ulu/
```

Hızlı bağlantı (1 GB/s üstü) için:
```bash
rsync -avhSP -e "ssh -T -c aes128-ctr -o Compression=no -x" host:kaynak/ hedef/
```

---

## Backup Politikası

- **Home dizini:** Günlük backup + 15 dakikalık snapshot'lar
- **/work:** Aylık backup
- **/localscratch:** Backup **yok** — geçici iş dosyaları için

Backup'tan dosya geri almak için Stefan'a yaz.

---

## Hesap Silinmesi

Cluster kullanmayı bırakınca Stefan'a bildir. Aksi halde:
- 5 yıl inaktif → hesap silinir
- Veriler en az 10 yıl arşivlenir
