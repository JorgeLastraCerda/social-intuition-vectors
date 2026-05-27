# Claude için Talimatlar

## Bu Proje Nedir?

Bu dizin, Universität Konstanz'ın SCCKN (Scientific Compute Cluster) sistemine bağlanmak ve kullanmak için yapılan kurulum adımlarını içeriyor. Kullanıcı bu cluster'ı araştırma amaçlı kullanıyor: dosya yükleme/indirme, job başlatma, Jupyter üzerinden çalışma.

---

## SCCKN_SETUP.md Nasıl Güncellenmeli?

Her yeni kurulum adımı, yapılandırma değişikliği veya öğrenilen yeni komut sonrasında `SCCKN_SETUP.md` dosyası güncellenmelidir.

### Güncelleme Kuralları

- Her yeni adım tamamlandığında ilgili bölümü güncelle
- Yeni bir konu ekleniyorsa (örn. job scheduler, Python env kurulumu) yeni bir numara altında bölüm aç
- Eski ve artık kullanılmayan bilgileri sil, dosyayı temiz tut
- Komutları her zaman kod bloğu içinde göster
- Hızlı Başvuru bölümünü en sık kullanılan komutlarla güncel tut

---

## Açıklama Dili ve Stili

### Türkçe Yaz

Tüm açıklamalar Türkçe olmalı. Komutlar ve teknik terimler (ssh, mount, tmux vb.) olduğu gibi kalır.

### Sade Tut

- Kısa cümleler kullan
- Jargonu açıkla ("kernel extension = sistem çekirdeğine eklenen düşük seviyeli yazılım")
- Adım adım yaz, her adım tek bir işi anlatsın
- Neden yapıldığını bir cümleyle belirt ("SSH key oluşturuldu → her girişte şifre yazmak gerekmesin diye")

### Tablo ve Kod Blokları

- Hesap bilgileri ve sabit değerler tablo formatında
- Tüm komutlar ``` kod bloğu ``` içinde
- Uzun açıklamalar yerine kısa bullet list tercih et

---

## Genel Davranış Kuralları

- Kullanıcı teknik detayları merak ettiğinde kısa ve net açıkla
- Bir şeyin güvenli olup olmadığı sorulursa önce araştır, sonra cevap ver
- Kurulum adımlarını sıralı ve test edilebilir şekilde sun
- Bir adım tamamlandığında belgeyi güncelle, yarım bırakma
- Ekstra özellik veya kurulum önerme — sadece kullanıcının istediğini yap
