# Benchmark ve Optimizasyon Raporu

RAG soru-cevap sistemi. Baseline konfigurasyondan final konfigurasyona kadar
yapilan tum olcumler, karar gerekceleri ve sonuclar.

---

## 1. Ozet

| Metrik | Baseline | Final | Degisim |
|---|---|---|---|
| Faithfulness | 0.7377 | **0.8785** | **+0.141** |
| Answer Relevancy | 0.6555 | **0.7659** | **+0.110** |
| Context Precision | 0.8048 | 0.7866 | -0.018 |
| Context Recall | 0.6426 | 0.6598 | +0.017 |

**17 olcum, 16 farkli konfigurasyon.** Tum olcumlerde NaN sayisi sifir
(basarisiz is yok).

| Konfigurasyon | Baseline | Final |
|---|---|---|
| Chunk boyutu / overlap | 500 / 50 | 800 / 80 |
| Metin temizligi | Yok | **Var** |
| Retrieval | Dense | Dense |
| retrieve_k -> top_k | 5 -> 5 | **20 -> 5** |
| Reranker | Yok | **bge-reranker-v2-m3** |
| Hybrid search | Yok | Yok (olculdu, reddedildi) |
| Prompt varyanti | baseline | **cited** |
| Temperature | 0.0 | 0.0 |

---

## 2. Olcum metodolojisi

### 2.1 Degerlendirme

Ragas kutuphanesi, LLM-as-a-judge yaklasimi. Hakem modeli:
`openai/gpt-oss-120b` (uretim modelinden bagimsiz).

| Metrik | Olctugu | Kategori |
|---|---|---|
| Faithfulness | Cevabin baglama sadakati (halusinasyon kontrolu) | Generation |
| Answer Relevancy | Cevabin soruyu karsilama duzeyi | Generation |
| Context Precision | Getirilen chunk'larin isabeti | Retrieval |
| Context Recall | Gerekli bilginin yakalanma orani | Retrieval |

### 2.2 Retrieval F1

Context Precision ve Context Recall arasinda dogrudan bir denge var. Iki
metrigi tek sayida ozetlemek icin harmonik ortalama kullanildi:

```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

Aritmetik ortalama tercih edilmedi; tek tarafa kacan konfigurasyonlari
cezalandirmiyor. Ornegin Precision=0.90 / Recall=0.30 icin aritmetik ortalama
0.60, harmonik ortalama 0.45 verir.

### 2.3 Olcum gurultusu

Ayni pipeline ciktisi iki kez degerlendirildi (ayni veri, ayni ayar,
`temperature=0`):

| Metrik | 1. olcum | 2. olcum | Fark |
|---|---|---|---|
| Faithfulness | 0.7349 | 0.7405 | +0.0056 |
| Answer Relevancy | 0.6595 | 0.6515 | -0.0080 |
| Context Precision | 0.8056 | 0.8040 | -0.0016 |
| Context Recall | 0.6486 | 0.6365 | -0.0121 |

**Gurultu seviyesi: ±0.012.**

**Yorumlama kurali:** 0.03'ten buyuk farklar anlamli kabul edildi.
0.01-0.02 araligi gurultu bandi sayildi.

### 2.4 Gecerlilik kontrolu

Her olcumde metrik basina NaN sayisi raporlandi. NaN, hesaplanamamis metrik
demektir ve ortalamaya dahil edilmez; bu skorlari bilinmeyen yonde saptirir.

Ilk denemelerde rate limit ve token yetersizligi nedeniyle NaN olusmustu; ayni
veriyle yapilan uc hatali calistirmada Faithfulness 0.61 / 0.64 / 0.67 gibi
farkli sonuclar cikti. Ayarlar duzeltildikten sonra tum olcumler temiz alindi.

---

## 3. Tum deneyler

Kronolojik sira. Kalin degerler her sutundaki en iyi sonucu gosterir.

| # | Deney | chunk | k | Rerank | Prompt | Temizlik | Faith. | Ans.Rel. | Ctx.Prec. | Ctx.Rec. | F1 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | baseline | 500 | 5 | - | base | - | 0.7377 | 0.6555 | 0.8048 | 0.6426 | 0.7137 |
| 2 | k3 | 500 | 3 | - | base | - | 0.7079 | 0.6550 | 0.8444 | 0.5479 | 0.6647 |
| 3 | k10 | 500 | 10 | - | base | - | 0.7245 | 0.6812 | 0.7406 | 0.7360 | 0.7383 |
| 4 | k20 | 500 | 20 | - | base | - | 0.7475 | 0.6291 | 0.7184 | 0.7535 | 0.7353 |
| 5 | chunk200 | 200 | 10 | - | base | - | 0.6656 | 0.6335 | 0.6717 | 0.4881 | 0.5653 |
| 6 | chunk800 | 800 | 10 | - | base | - | 0.7232 | 0.6290 | 0.8240 | 0.7227 | 0.7701 |
| 7 | chunk800_k5 | 800 | 5 | - | base | - | 0.7600 | 0.6977 | 0.8528 | 0.7073 | 0.7733 |
| 8 | chunk1000_k4 | 1000 | 4 | - | base | - | 0.6661 | 0.6182 | **0.8741** | 0.7307 | **0.7960** |
| 9 | hybrid | 800 | 5 | - | base | - | 0.7188 | 0.6517 | 0.8001 | 0.6911 | 0.7416 |
| 10 | rerank20_5 | 800 | 20>5 | + | base | - | 0.7850 | 0.7219 | 0.8081 | 0.6858 | 0.7419 |
| 11 | rerank10_5 | 800 | 10>5 | + | base | - | 0.7848 | 0.7220 | 0.7924 | 0.6682 | 0.7250 |
| 12 | prompt_strict | 800 | 20>5 | + | strict | - | 0.5986 | 0.4720 | 0.8174 | 0.6487 | 0.7233 |
| 13 | prompt_cited | 800 | 20>5 | + | cited | - | 0.8244 | 0.7025 | 0.8173 | 0.7025 | 0.7556 |
| 14 | **clean_corpus** | 800 | 20>5 | + | cited | + | **0.8785** | 0.7659 | 0.7866 | 0.6598 | 0.7176 |
| 15 | clean_top3 | 800 | 20>3 | + | cited | + | 0.8492 | **0.7698** | 0.8222 | 0.6095 | 0.7000 |
| 16 | clean_thresh01 | 800 | 20>5* | + | cited | + | 0.8602 | 0.7681 | 0.8046 | 0.5582 | 0.6591 |

\* Skor esigi 0.1; ortalama baglam sayisi 2.2'ye dustu.

Deney 1'in degerleri iki bagimsiz olcumun ortalamasidir.

---

## 4. Deney serileri

### 4.1  k parametresi taramasi

Sabit: chunk=500/50, reranker yok, baseline prompt.

| Metrik | k=3 | k=5 | k=10 | k=20 |
|---|---|---|---|---|
| Context Precision | **0.8444** | 0.8048 | 0.7406 | 0.7184 |
| Context Recall | 0.5479 | 0.6426 | 0.7360 | **0.7535** |
| Faithfulness | 0.7079 | 0.7377 | 0.7245 | **0.7475** |
| Answer Relevancy | 0.6550 | 0.6555 | **0.6812** | 0.6291 |
| Retrieval F1 | 0.6647 | 0.7137 | **0.7383** | 0.7353 |
| Olcum suresi | 39 dk | 47 dk | 68 dk | 111 dk |

**Bulgular**

- Precision-Recall takasi ders kitabi davranisi sergiledi: k arttikca Precision
  monoton dustu (0.8444 -> 0.7184), Recall monoton yukseldi (0.5479 -> 0.7535).
- k=3'te Recall coktu (0.5479) ve bu Faithfulness'i da asagi cekti (seride en
  dusuk). Baglamda yeterli bilgi olmadiginda model bosluklari doldurmaya
  calisiyor.
- Answer Relevancy k=10'da tepe yapip k=20'de belirgin dustu (-0.052). Uzun
  baglamda modelin odagini kaybetmesi ("lost in the middle") tutarli bir
  aciklama sunuyor.

**Karar: k = 10.** F1 tepe noktasi burada. k=20'ye geciste Recall kazanci
+0.018 (gurultu bandina yakin), Precision kaybi -0.022; ek maliyetin karsiligi
yok. Ayrica k=20 olcumu 1.6 kat uzun surdu.

**Metodolojik not.** Aradaki her tam sayi denenmedi. Ardisik k degerleri
arasindaki farkin gurultu bandinin (±0.012) altinda kalmasi bekleniyordu;
genis araliklarla tarama, ayni bilgiyi ucte bir surede verdi.

### 4.2  Chunk boyutu taramasi

Sabit: k=10 (chunk1000 deneyinde k=4, chunk800_k5'te k=5).
Overlap her deneyde chunk_size'in %10'u.

| Metrik | 200/k10 | 500/k10 | 800/k10 | 800/k5 | 1000/k4 |
|---|---|---|---|---|---|
| Context Precision | 0.6717 | 0.7406 | 0.8240 | 0.8528 | **0.8741** |
| Context Recall | 0.4881 | **0.7360** | 0.7227 | 0.7073 | 0.7307 |
| Faithfulness | 0.6656 | 0.7245 | 0.7232 | **0.7600** | 0.6661 |
| Answer Relevancy | 0.6335 | 0.6812 | 0.6290 | **0.6977** | 0.6182 |
| Retrieval F1 | 0.5653 | 0.7383 | 0.7701 | 0.7733 | **0.7960** |

**Bulgular**

**chunk=200 dort metrikte de kotulesti.** Beklenti, kucuk chunk'in en azindan
Precision'i artirmasiydi; gerceklesmedi. 200 karakter anlamsal butunluk
esiginin altinda kaliyor: chunk tam bir fikri barindiramiyor, cumlenin
ortasindan baslayip ortasinda bitiyor. Hakem model parcayi tek basina anlamsiz
buldugu icin alakasiz sayiyor.

**chunk=800'de Precision +0.083 sicradi.** 800 karakter, veri setindeki tipik
bir forum cevabini butun halinde barindiracak buyuklukte.

**Toplam baglam uzunlugu hipotezi.** chunk=800/k=10 kombinasyonunda Answer
Relevancy dustu (-0.052) -- k=20 deneyindekiyle ayni buyuklukte. Ortak nokta
toplam baglam uzunlugu: 800 × 10 ≈ 8000 karakter. Hipotez, k yariya
indirilerek test edildi.

**Hipotez dogrulandi (800/k5).** Toplam baglam 8000'den 4000 karaktere
inince:

| Metrik | 800/k10 | 800/k5 | Fark |
|---|---|---|---|
| Answer Relevancy | 0.6290 | 0.6977 | +0.069 |
| Faithfulness | 0.7232 | 0.7600 | +0.037 |
| Context Precision | 0.8240 | 0.8528 | +0.029 |

Recall -0.015 ile gurultu bandinda kaldi.

**chunk=1000/k=4: retrieval iyilesti, generation bozuldu.** Toplam baglam yine
~4000 karakter ama parcalar daha buyuk. F1 serinin en iyisine ulasti (0.7960),
buna karsilik Faithfulness -0.094 ve Answer Relevancy -0.080 dustu. 1000
karakterlik chunk'lar retrieval acisindan avantajli (parca butun bir cevabi
iceriyor) ama generation icin dezavantajli (her chunk birden fazla fikir ve
alakasiz bolum barindiriyor).

**Karar: chunk=800 / overlap=80 / k=5.** Dort metrikten ucunde en yuksek
deger. 1000/k4 retrieval F1'de 0.023 onde ancak generation metriklerinde
0.08-0.09 geride.

### 4.3  Hybrid search (BM25 + dense)

Sabit: chunk=800/80, k=5. Reciprocal Rank Fusion ile birlestirme.

| Metrik | Dense | Hybrid | Fark |
|---|---|---|---|
| Context Precision | **0.8528** | 0.8001 | -0.053 |
| Context Recall | **0.7073** | 0.6911 | -0.016 |
| Faithfulness | **0.7600** | 0.7188 | -0.041 |
| Answer Relevancy | **0.6977** | 0.6517 | -0.046 |
| Retrieval F1 | **0.7733** | 0.7416 | -0.032 |

**Karar: REDDEDILDI.** Dort metrik de dustu.

**Aciklama.** Beklenti, BM25'in embedding'in zayif oldugu alanlari (kod,
kisaltma, ozel isim) kapatmasiydi. Corpus forum cevaplarindan olusuyor ve
yaygin finansal kelimeler ("account", "bank", "money", "business") her yerde
geciyor. BM25 bu kelimelerin yogunlastigi chunk'lari one cikariyor, ancak
yaygin olduklari icin ayirt edici degiller.

Ikinci etken RRF'in esit agirlikli birlestirme yapmasi: dense arama tek basina
yeterince iyi calisirken, zayif bir ikinci sinyali esit agirlikla karistirmak
siralamayi bozuyor.

### 4.4  Reranker

Model secimi once dogrulandi. Sorgu: "How do I deposit a third party cheque?"

| Dokuman | Qwen3-Reranker-8B | bge-reranker-v2-m3 |
|---|---|---|
| "The weather in Ankara is cold in winter." | **0.903** (1.) | 0.000017 (3.) |
| "Just have the associate sign the back..." | 0.880 (2.) | 0.023 (2.) |
| "A third party cheque requires endorsement..." | 0.562 (3.) | **0.807** (1.) |

Qwen3-Reranker konuyla ilgisiz cumleyi en alakali, dogrudan cevabi ise en
alakasiz olarak siraladi. Skorlarin birbirine yakinligi anlamli bir ayrim
yapamadigini gosteriyor. `bge-reranker-v2-m3` secildi.

**Sonuclar** (chunk=800/80):

| Metrik | Rerankersiz | 20 -> 5 | 10 -> 5 |
|---|---|---|---|
| Faithfulness | 0.7600 | **0.7850** | 0.7848 |
| Answer Relevancy | 0.6977 | 0.7219 | **0.7220** |
| Context Precision | **0.8528** | 0.8081 | 0.7924 |
| Context Recall | **0.7073** | 0.6858 | 0.6682 |
| Retrieval F1 | **0.7733** | 0.7419 | 0.7250 |
| Dort metrik ortalamasi | 0.7545 | 0.7502 | 0.7419 |

**Bulgular**

- Beklenti gerceklesmedi: reranker'in amaci genis aday havuzundan eleme
  yaparak Precision'i korumakti; her iki retrieval metrigi de dustu.
- Buna karsilik generation belirgin iyilesti (her ikisi de ~+0.025, gurultu
  bandinin uzerinde).
- Aday havuzu genisligi (20 vs 10) generation metriklerini etkilemedi
  (fark 0.0002 mertebesinde); reranker her iki havuzdan da buyuk olcude ayni
  chunk'lari seciyor. Retrieval metrikleri genis havuzda daha iyi.

**Karar: KULLANILDI, retrieve_k=20 -> top_k=5.**

Gerekce: Faithfulness ve Answer Relevancy son kullanicinin dogrudan
deneyimledigi ciktiyi olcuyor. Context Precision ve Recall ara asama
metrikleridir. Faithfulness dogrudan halusinasyon kontrolu islevi goruyor ve
finansal alanda uydurma cevabin maliyeti yuksek.

Dort metrigin duz ortalamasi neredeyse esit oldugu icin bu karar tartismaya
aciktir; her iki konfigurasyonun skorlari yukarida birlikte sunulmustur.

### 4.5  Prompt muhendisligi

Sabit: chunk=800/80, retrieve_k=20 -> top_k=5, reranker acik.

| Metrik | baseline | strict | cited |
|---|---|---|---|
| Faithfulness | 0.7850 | 0.5986 | **0.8244** |
| Answer Relevancy | **0.7219** | 0.4720 | 0.7025 |
| Context Precision | 0.8081 | **0.8174** | 0.8173 |
| Context Recall | 0.6858 | 0.6487 | **0.7025** |
| Retrieval F1 | 0.7419 | 0.7233 | 0.7556 |
| Dort metrik ortalamasi | 0.7502 | 0.6342 | **0.7617** |

**strict varyanti: hedeflenen metrik ters yonde hareket etti.**

Prompt'a "cikarim yapma, genelleme yapma, bilgileri birlestirme" kurallari
eklendi. Amac Faithfulness'i yukseltmekti; 0.186 dusurdu.

Mekanizma: "bilgi yok" yaniti veren soru sayisi 6'dan 12'ye cikti (%40).
Baglam dogru cevabi icerdigi durumlarda bile model, bilgiyi soruya baglamak
icin gereken en kucuk adimi atmayi reddetti. Ayrica "baglam bunu
dogrulamiyor" gibi meta-iddialar uretmeye basladi; bunlar baglamdan
cikarilabilir olgusal iddialar olmadigi icin hakem model tarafindan
desteklenmemis sayildi.

**cited varyanti: kaynak gosterimi.**

Kisitlama getirmek yerine modelden her iddianin sonuna kaynak blogu (`[1]`,
`[2]`) yazmasi istendi. Faithfulness projedeki en yuksek degere ulasti
(0.8244, +0.039).

Mekanizma: alinti zorunlulugu, uretim sirasinda surekli baglama donmeyi
gerektiriyor; baglamda karsiligi olmayan bir sey soylemek zorlasiyor.

**Karar: cited.**

**Genel cikarim.** Iki varyant zit yonde sonuc verdi. Modele *ne
yapamayacagini* soylemek (strict) islevsizlik yaratti; *nasil yapmasi
gerektigini* gostermek (cited) hedeflenen metrigi yukseltti.

### 4.6  Corpus temizligi

Veri setindeki context bloklari birden fazla forum cevabinin birlestirilmis
hali. Corpus taramasi:

| Bozukluk | Adet |
|---|---|
| Cift tirnak (`""`) | 104 |
| Tirnak sonrasi bosluksuz harf | 87 |
| Coklu bosluk | 260 |
| Bosluksuz cumle sonu (`.X`) | 25 |

Ornek: `...Before you convert to S-Corp.You don't need to notify the IRS...`

Bu bozukluklar recursive chunking'in `". "` ayiricisini devre disi birakiyor;
algoritma bir alt seviyeye dusuyor ve chunk sinirlari cumle ortasindan
geciyor.

Uygulanan dort kural: cift tirnaklarin tekile indirilmesi, cumle sonu
noktalamasindan sonra bosluk eklenmesi, tirnak sonrasi bosluk eklenmesi,
coklu bosluklarin teke indirilmesi.

Sonuc: 138 -> 137 chunk; chunk'larin %85'inin metni degisti.

| Metrik | Temizliksiz | Temizlikli | Fark |
|---|---|---|---|
| Faithfulness | 0.8244 | **0.8785** | +0.054 |
| Answer Relevancy | 0.7025 | **0.7659** | +0.063 |
| Context Precision | **0.8173** | 0.7866 | -0.031 |
| Context Recall | **0.7025** | 0.6598 | -0.043 |
| Dort metrik ortalamasi | 0.7617 | **0.7727** | +0.011 |

**Karar: UYGULANDI.**

Generation metrikleri projedeki en yuksek degerlere ulasti. Chunk sinirlari
artik cumle ortasindan gecmiyor; model butun cumleler okuyor. Retrieval
metriklerindeki dusus, chunk sinirlarinin degismesiyle ground truth
eslesmesinin kaymasindan kaynaklaniyor.

**Bu, tek bir degisiklikle elde edilen en buyuk kazanc oldu.** Hicbir
parametre ayari Faithfulness'a +0.054 katkida bulunmadi.

### 4.7  Reddedilen ek denemeler

**top_k = 3.** Hipotez: reranker zaten en iyi adaylari sectigine gore daha az
chunk vermek gurultuyu azaltip Faithfulness'i yukseltebilir.

| Metrik | top_k=5 | top_k=3 |
|---|---|---|
| Faithfulness | **0.8785** | 0.8492 |
| Context Recall | **0.6598** | 0.6095 |
| Dort metrik ortalamasi | **0.7727** | 0.7627 |

Hipotez dogrulanmadi. Faithfulness beklenenin tersine dustu (-0.029);
Recall'daki -0.050'lik dusus sebebi acikliyor. Uc chunk yetersiz kaliyor.
Bu, k=3 deneyinde gozlenen kalibin tekrari.

**Reranker skor esigi.** Skor dagilimi: medyan 0.0268, min 0.0003, max 0.9987.
Hipotez: sabit top_k yerine skor esigi ile baglam sayisini dinamik belirlemek.

Esik = 0.1 uygulandiginda ortalama baglam sayisi 5'ten 2.2'ye dustu.

| Metrik | Esiksiz | Esik 0.1 |
|---|---|---|
| Faithfulness | **0.8785** | 0.8602 |
| Context Recall | **0.6598** | 0.5582 |
| Dort metrik ortalamasi | **0.7727** | 0.7478 |

Hipotez dogrulanmadi. Context Recall -0.102 ile agir kayip verdi.

Cikarim: reranker skorlarinin mutlak degerleri guvenilir bir alaka olcusu
degil. Dusuk skorlu bir chunk, siralamada geride olsa da cevabin bir parcasini
tasiyabiliyor. Goreli siralama (top_k) mutlak esikten daha saglikli calisiyor.

---

## 5. Vektor veritabani karsilastirmasi

Ragas olcumu disinda yapilan performans karsilastirmasi. Ayni 138 chunk ve
ayni vektorlerle ChromaDB kurulup 30 sorgu her iki yontemde calistirildi.

| Yontem | Sorgu suresi (ort.) | 2. calistirma |
|---|---|---|
| NumPy (exact, brute-force) | 0.203 ms | 0.282 ms |
| ChromaDB (HNSW) | 4.374 ms | 5.167 ms |

- NumPy 18-21 kat daha hizli
- Sonuc ortusme orani: **%100** (iki yontem ayni chunk'lari donduruyor)
- ChromaDB indeksleme suresi: 1.07 saniye

**Karar: NumPy tabanli exact search.**

ANN indeksleri milyonlarca vektor icinde arama yaparken devreye giren bir
optimizasyondur. 138 vektorde indeksin kendisi ek yuk olusturuyor; ChromaDB'nin
yavasligi HNSW algoritmasindan degil, katman maliyetinden kaynaklaniyor
(SQLite kalicilik, serilestirme, API cagrisi).

Ayrica exact search kesin sonuc verir, ANN yaklasik calisir. Bu olcekte
yaklasiklik icin bir sebep yok.

---

## 6. Ilerleme adimlari

Baseline'dan final'e giden yolda her adimin katkisi:

| Adim | Degisiklik | Faithfulness | Ans. Relevancy |
|---|---|---|---|
| 0 | Baseline (500/50, k=5) | 0.7377 | 0.6555 |
| 1 | chunk 500 -> 800, k 5 (sabit) | 0.7600 (+0.022) | 0.6977 (+0.042) |
| 2 | Reranker eklendi (20 -> 5) | 0.7850 (+0.025) | 0.7219 (+0.024) |
| 3 | Prompt: baseline -> cited | 0.8244 (+0.039) | 0.7025 (-0.019) |
| 4 | Corpus temizligi | **0.8785 (+0.054)** | **0.7659 (+0.063)** |

Toplam: Faithfulness +0.141, Answer Relevancy +0.110.

Her adim gurultu bandinin (±0.012) uzerinde katki sagladi. En buyuk tek katki
corpus temizliginden geldi.

---

## 7. Sonuclar ve cikarimlar

**1. Retrieval metrikleri ile uretim kalitesi her zaman ayni yone gitmiyor.**

Uc ayri deneyde bu ayrisma gozlendi:

| Deney | Retrieval | Generation |
|---|---|---|
| chunk=1000/k=4 | Yukseldi (F1 0.7960) | Coktu (-0.09) |
| Reranker | Dustu (F1 0.7419) | Yukseldi (+0.025) |
| Corpus temizligi | Dustu (F1 0.7176) | Yukseldi (+0.06) |

Yalnizca retrieval skorlarina veya yalnizca F1'e bakarak optimizasyon yapmak
yaniltici olurdu.

**2. Toplam baglam uzunlugu, chunk boyutu ve k'dan daha belirleyici.**

Chunk boyutu × k carpimi ~4000 karakter civarinda tutuldugunda en iyi sonuclar
alindi. 8000 karakterde "lost in the middle" etkisi gozlendi; 2400 karakterde
bilgi yetersiz kaldi.

**3. Prompt'ta kisitlama getirmek yerine yapi getirmek etkili.**

Yasak listesi (strict) modeli islevsizlestirdi ve hedeflenen metrigi 0.186
dusurdu. Kaynak gosterimi (cited) ayni metrigi 0.039 yukseltti.

**4. Veri kalitesi, parametre ayarindan daha buyuk kazanc sagladi.**

Corpus temizligi tek basina Faithfulness'a +0.054 katti. Hicbir hiperparametre
degisimi bu kadar etkili olmadi.

**5. Negatif sonuclar da sonuctur.**

Denenen alti yaklasimdan ucu reddedildi: hybrid search, top_k=3, reranker skor
esigi. Bu denemeler, kullanilmayan yaklasimlarin neden kullanilmadigina dair
olculmus gerekce sagliyor.

---

## 8. Sinirlamalar

- **Veri seti 30 ornek iceriyor.** Istatistiksel guc dusuk; tek bir sorunun
  puani genel ortalamayi %3 kaydirabiliyor.
- **Hakem modeli tek.** `openai/gpt-oss-120b` disinda bir hakemle olcum
  yapilmadi; farkli bir hakemle mutlak skorlar degisebilir. Goreli
  karsilastirmalarin etkilenmesi beklenmez.
- **Olcum gurultusu ±0.012.** Bu bandin altindaki farklar yorumlanmadi.
- **Context Recall 0.66 seviyesinde kaldi.** Denenmemis bir yaklasim sorgu
  genisletme (query expansion) olabilir.
- **Retrieval gorevi gorece kolay.** Corpus'ta 30 bagimsiz blok var ve her
  sorunun cevabi kendi blogunda. Gercek dunya senaryolarinda dokuman havuzu
  cok daha buyuk ve karisik olur.

---

## 9. Final konfigurasyon

```
chunk_size        = 800
chunk_overlap     = 80
metin temizligi   = acik
retrieval         = dense (NumPy exact search)
retrieve_k        = 20
top_k             = 5
reranker          = bge-reranker-v2-m3
hybrid search     = kapali
prompt varyanti   = cited
temperature       = 0.0
max_tokens        = 512
```

| Metrik | Deger |
|---|---|
| Faithfulness | 0.8785 |
| Answer Relevancy | 0.7659 |
| Context Precision | 0.7866 |
| Context Recall | 0.6598 |
| Retrieval F1 | 0.7176 |
