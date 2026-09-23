# Proje Gozlemleri

Deneyler boyunca tutulan teknik notlar.

---

## Veri seti

- `vibrantlabsai/fiqa` (`ragas_eval_v3`) ayri bir dokuman corpus'u icermiyor.
  Sadece 30 soru ve her sorunun `retrieved_contexts` alani var. Indekslenecek
  bilgi havuzu bu context'lerden olusturuldu. Mimari acidan ilk karar noktasi
  buydu.
- Toplam 30 dokuman, ~88.400 karakter. Her satirda tam 1 context, ortalama
  2.946 karakter.
- Context bloklari birden fazla forum cevabinin birlestirilmis hali. Cevaplar
  arasinda ayirici yok; bazi birlesme noktalarinda noktadan sonra bosluk
  bile yok (`...to be there.Anybody can deposit...`). Bu, dogal cumle
  ayiricisinin devreye girmesini engelliyor.

## Ortam

- SSL denetimi yapan aglarda Python HTTPS isteklerinde sertifika hatasi
  olusuyor; `pip-system-certs` + `certifi.where()` ile cozuldu.
- `truststore` paketi Python 3.13 ile uyumsuz (`RecursionError`). `certifi`
  ile degistirildi.

## Model altyapisi

- Servis vLLM uzerinde calisiyor, OpenAI uyumlu.
- Embedding boyutu: 4096.
- Judge modeli (`openai/gpt-oss-120b`) cevabin yaninda `reasoning` alani
  uretiyor. Dusunme adimi token harciyor; `max_tokens` dusuk tutulursa
  `content` bos donuyor. Ragas cagirilarinda dikkat edilmeli.
- Uretim modeli olarak `Qwen3-Next-80B-A3B-Instruct` kullanildi. Serviste
  daha buyuk `Qwen3.5-397B-A17B-FP8` de mevcut.

## Embedding dogrulamasi

Anlamsal aramanin kelime aramasindan farkini gosteren olcum:

| Cift | Benzerlik |
|---|---|
| "net profit increased by 27 percent" - "revenue showed significant growth" | 0.6149 |
| "net profit increased by 27 percent" - "the cat is sleeping on the sofa" | 0.2272 |

Iki finansal cumlede ortak kelime yok; klasik kelime aramasi eslesme
bulamazdi. Embedding anlamsal yakinligi yakaliyor.

## Chunking yaklasimi

Recursive splitting kullanildi. Ayiricilar en dogaldan en zorlamaya dogru
siralanmis bir hiyerarsi olusturuyor:

```
["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""]
```

Algoritma metni once paragraf siniri (`\n\n`) uzerinden bolmeyi dener. Ortaya
cikan parcalardan biri hedef boyutu asiyorsa, o parca icin fonksiyon kendini
bir alt ayirici ile yeniden cagirir. Boylece bolme mumkun olan en dogal
noktadan yapilir; karakter bazli zorlama bolme yalnizca son care olarak
devreye girer.

Amac anlamsal butunlugu korumak: paragraf sinirinda anlam zaten tamamlanmis
durumda, cumle ortasinda kesmek ise baglami koparir.

Bu, LangChain'in `RecursiveCharacterTextSplitter` sinifiyla ayni yaklasim;
proje kapsaminda saf Python ile yeniden yazildi.

### Corpus kaynakli sinirlama

Veri setindeki context bloklari birden fazla forum cevabinin birlestirilmis
hali. Birlesme noktalarinda noktadan sonra bosluk birakilmamis:

```
...you don't even technically have to be there.Anybody can deposit money...
...every bank will say the same thing.To do this, I need a state-issued...
```

Bu durumda `". "` ayiricisi eslesmiyor ve algoritma bir alt seviyeye (virgul,
bosluk) dusuyor. Sonuc olarak farkli kisilerin cevaplari ayni chunk icinde
karisabiliyor; bu Context Precision'i dusuren etkenlerden biri.

Bu gozlem daha sonra corpus temizligi deneyine yol acti (bkz. Deney serisi 6).

## Retrieval gozlemleri (baseline: chunk=500, overlap=50, k=3)

Ilk uc soruda dogru dokumanin chunk'i 1. sirada geldi. Ancak:

- **q_002'de skorlar cok yakin:** dogru chunk 0.5776, ikinci 0.5763,
  ucuncu (yanlis dokumandan) 0.5620. Aradaki fark binde birler seviyesinde.
  Siralama kirilgan; parametre degisiminde kolayca bozulabilir. Reranker'in
  fayda saglayabilecegi tipik durum.
- **q_000'de konu-dogru ama kaynak-yanlis chunk'lar geldi:** 2. ve 3. sirada
  `doc_008` chunk'lari var. Ayni konuyu (cek/banka hesabi) isliyor ama farkli
  bir soruya ait. Context Precision'i dusurecek bir durum.

## Beklenen sinirlar

Corpus'ta 30 bagimsiz blok var ve her sorunun cevabi kendi blogunda. Retrieval
gorevi gorece kolay: 30 samanlikta 1 igne. Baseline skorlarinin bastan yuksek
cikmasi ve optimizasyon etkilerinin kucuk farklarla gorunmesi bekleniyor.
Sonuclar yorumlanirken bu durum goz onunde tutulmali.

## Deney tekrarlanabilirligi

### temperature = 0.0

Tum LLM cagirilarinda `temperature=0` kullanildi. Gerekce: hiperparametre
optimizasyonunda skor degisiminin kaynagi belirsiz olmamali.

Temperature > 0 oldugunda model her cagrida farkli kelime secebilir. Bu
durumda k=5'ten k=10'a gecildiginde skorun degismesi ya parametre etkisidir
ya da modelin rastgeleligidir; ayirt edilemez. Deterministik cikti, olculen
farkin yalnizca degistirilen parametreden kaynaklandigini garanti eder.

Bedeli: uretilen cevaplar bir miktar daha mekanik olabilir. Degerlendirme
odakli bir projede bu kabul edilebilir bir takas.

### Embedding cache

Ayni metnin vektoru bir kez hesaplanip diske yaziliyor
(`data/cache/embeddings.json`). Cache anahtari model adi + metin icerigi
uzerinden SHA-256 ile uretiliyor.

Model adinin anahtara dahil edilmesi kritik: farkli bir embedding modeline
gecildiginde cache eski vektorleri dondurmez. Aksi halde sessiz ve tespiti
zor bir olcum hatasi olusurdu.

Etki: chunk parametreleri sabitken yapilan deneyler (k, prompt, reranker)
embedding asamasini tamamen atliyor. Ilk indeksleme 9.7 saniye, sonraki
calistirmalar aninda.

### Prompt yapisi

Baglam bloklari numaralandirilarak veriliyor (`[1]`, `[2]`, ...). Veri
setindeki context'ler birden fazla forum cevabinin birlestirilmis hali
oldugu icin, numaralandirma modelin ayri kaynaklari birbirine karistirmasini
azaltiyor.

Baglam once, soru sonra yerlestirildi. Uzun girdilerde modellerin ortadaki
bilgiyi kacirma egilimi ("lost in the middle") nedeniyle soru, cevap
uretimine en yakin konumda tutuldu.

## Ilk uctan uca sorgu (baseline: chunk=500, overlap=50, k=5)

q_000 icin uretilen cevap ground truth ile buyuk olcude ortusuyor. Model
baglamdaki nuansi korumus: "Bank of America gibi bazi bankalar bunun federal
duzenleme oldugunu *iddia edebilir*" seklinde aktarmis, kesin gercek olarak
sunmamis. Kaynak metin de zaten bir forum kullanicisinin iddiasiydi. System
prompt'un baglam sadakati kurali calisiyor.

Ancak getirilen 5 chunk'tan 2'si `doc_008`'den geldi (skorlar 0.7739 ve
0.7526). Konu ayni (cek/isletme hesabi) ama farkli bir soruya ait dokuman.
Bu tur "konu-dogru, kaynak-yanlis" chunk'lar Context Precision'i dusurecek.
Reranker ve k degeri optimizasyonunda oncelikli hedef.

## Baseline pipeline calistirmasi (chunk=500, overlap=50, k=5)

- 30 soru, 186 saniye (6.2 sn/soru). Her deney turu icin pipeline maliyeti
  ~3 dakika. Deney sayisi bu sureye gore planlanmali.
- Bos cevap yok. Cevap uzunlugu ort. 509 karakter (min 81, max 1182).
- **6 soruda "yeterli bilgi yok" yaniti alindi**: q_002, q_011, q_015, q_020,
  q_027, q_028 (%20).

Bu oranin iki olasi aciklamasi var:

1. Model durust davraniyor; retrieval yanlis baglam getirdiginde uydurmak
   yerine cekiniyor. Faithfulness'i korur.
2. Retrieval gercekten basarisiz; dogru bilgi corpus'ta var ama bulunamiyor.
   Context Recall ve Answer Relevance dusuk cikar.

Ragas olcumu bu ayrimi netlestirecek. q_002 dikkat cekici: retrieval
testinde de skorlari birbirine cok yakin cikmisti (0.5776 / 0.5763 / 0.5620).
Zayif eslesme sinyali iki olcumde de tutarli.

## Ragas surum uyumlulugu

`pip install ragas` en guncel surumu (0.4.3) kurdu, ancak bu surum
`langchain-community`'nin eski yapisini bekliyor. Kurulan yeni surumler
(langchain-community 0.4.2, langchain-openai 1.1.9) o modulleri kaldirmis
oldugu icin import hatasi olustu:

```
ModuleNotFoundError: No module named 'langchain_community.chat_models.vertexai'
ImportError: cannot import name 'ContextOverflowError' from 'langchain_core.exceptions'
```

Cozum, uyumlu surum kombinasyonunu acikca sabitlemek oldu:

| Paket | Surum |
|---|---|
| ragas | 0.2.14 |
| langchain-community | 0.3.14 |
| langchain-openai | 0.2.14 |
| langchain-core | 0.3.63 |

Ders: hizli gelisen ekosistemlerde bagimlilik surumleri `requirements.txt`
icinde sabitlenmeli. Aksi halde ayni kod farkli bir tarihte kurulunca
calismayabilir.

## Ragas calistirma ayarlari

Ilk denemelerde iki tur hata alindi:

- `RateLimitError`: servis 15 istek/pencere siniri koyuyor. `max_workers=4`
  bu siniri asiyordu; 2'ye dusuruldu.
- `LLMDidNotFinishException`: hakem modeli reasoning urettigi icin
  `max_tokens` yetersiz kaliyordu. 2048 -> 4096 -> 8192 kademeli artirildi.

Hatali calistirmalarda basarisiz isler NaN olarak dusuyor ve ortalamaya dahil
edilmiyor. Bu, skorlari bilinmeyen yonde saptiriyor. Ayni veriyle yapilan uc
hatali calistirmada faithfulness 0.61 / 0.64 / 0.67 gibi farkli sonuclar
cikti. Bu nedenle her olcumde NaN sayisi raporlaniyor; sifir olmayan olcumler
gecersiz sayiliyor.

Paralellik artirmanin sureye etkisi olmadi (max_workers 2 ve 4 arasinda fark
yok). Darbogaz servis yanit suresi; yogun saatlerde belirgin sekilde
yavasliyor.

## Baseline sonuclari (chunk=500, overlap=50, k=5)

| Metrik | Skor |
|---|---|
| Context Precision | 0.8056 |
| Faithfulness | 0.7349 |
| Answer Relevancy | 0.6595 |
| Context Recall | 0.6486 |

Olcum temiz: 120 isin tamami tamamlandi, hicbir metrikte NaN yok.
Sure: 47 dakika (95 sn/soru).

### Yorum

Context Recall en zayif halka. Gerekli bilginin yaklasik ucte biri hic
getirilemiyor. Pipeline ciktisindaki 6 "yeterli bilgi yok" yaniti bu skorla
tutarli: retrieval basarisiz oldugunda model uydurmak yerine cekiliyor.

Answer Relevancy dusuklugu buyuk olcude Recall'in sonucu. Baglamda cevap
yoksa model ya eksik cevapliyor ya cekiliyor; her iki durum da bu metrigi
dusuruyor.

Precision yuksek (0.8056), Recall dusuk (0.6486). Sistem "az ama isabetli"
getiriyor. k=5 muhtemelen dar. Ilk optimizasyon hedefi Context Recall.

## Olcum gurultusu

Ayni pipeline ciktisi iki kez degerlendirildi (ayni veri, ayni ayar,
temperature=0). Amac: hakem modelin ne kadar tutarli puanladigini olcmek.

| Metrik | 1. olcum | 2. olcum | Fark |
|---|---|---|---|
| Faithfulness | 0.7349 | 0.7405 | +0.0056 |
| Answer Relevancy | 0.6595 | 0.6515 | -0.0080 |
| Context Precision | 0.8056 | 0.8040 | -0.0016 |
| Context Recall | 0.6486 | 0.6365 | -0.0121 |

**Gurultu seviyesi: ~±0.012.**

`temperature=0` ayarina ragmen skorlar tam olarak ayni cikmiyor. vLLM
tarafinda batch isleme ve kayan nokta islem sirasi tam determinizmi garanti
etmiyor; ayrica hakem modeli reasoning uretiyor ve bu adim oynakliga acik.

**Deney yorumlama kurali:** 0.03'ten buyuk farklar anlamli kabul edilir.
0.01-0.02 araligi gurultu bandinda; bu tur farklar icin tekrar olcum gerekir.

Context Precision en kararli metrik (±0.0016), Context Recall en oynak
(±0.0121). Recall, ground truth'u iddialara ayirip her birini baglamda
aramayi gerektirdigi icin daha fazla yorum iceriyor.

## Halusinasyon kontrolu dogrulamasi

Corpus disi bir soru soruldu: "What is the capital of France?"

Sistem yaniti: "The provided context does not contain enough information to
answer this question."

Model bu bilgiyi egitim verisinden biliyor, ancak system prompt'un "sadece
verilen baglami kullan" kurali nedeniyle kullanmadi.

Retrieval yine de 5 chunk getirdi (her zaman en yakin k tanesini dondurur),
ancak benzerlik skorlari 0.19-0.23 araligindaydi. Karsilastirma icin corpus
ici bir soruda en yuksek skor 0.7986'ydi. Skor araligi, eslesme olmadigi
durumu acikca gosteriyor.

Bu gozlem iki seyi dogruluyor:

1. System prompt'un halusinasyon kontrolu calisiyor.
2. Benzerlik skoru, retrieval basarisizligini tespit etmek icin kullanilabilir
   bir sinyal. Ileride bir esik degeri altinda kalan chunk'lari eleyerek
   Context Precision iyilestirilebilir. (Bu fikir daha sonra test edildi;
   bkz. "Reranker skor esigi".)

---

# Deney serisi 1: k parametresi taramasi

Sabit ayarlar: chunk_size=500, overlap=50, temperature=0, reranker yok,
hybrid search yok. Yalnizca retrieval'da cekilen chunk sayisi (k) degistirildi.

| Metrik | k=3 | k=5 | k=10 | k=20 |
|---|---|---|---|---|
| Context Precision | **0.8444** | 0.8048 | 0.7406 | 0.7184 |
| Context Recall | 0.5479 | 0.6426 | 0.7360 | **0.7535** |
| Faithfulness | 0.7079 | 0.7377 | 0.7245 | **0.7475** |
| Answer Relevancy | 0.6550 | 0.6555 | **0.6812** | 0.6291 |
| Retrieval F1 | 0.6647 | 0.7137 | **0.7383** | 0.7353 |
| Olcum suresi | 39 dk | 47 dk | 68 dk | 111 dk |

(k=5 degerleri iki bagimsiz olcumun ortalamasi. Tum olcumlerde NaN sayisi 0.
Gurultu bandi ±0.012.)

## Retrieval F1 hesabi

Context Precision ve Context Recall arasinda dogrudan bir denge var: k
arttikca biri yukselir, digeri duser. Iki metrigi tek sayida ozetlemek icin
harmonik ortalama (F1) kullanildi:

```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

Aritmetik ortalama yerine harmonik ortalama tercih edildi. Sebep: aritmetik
ortalama tek tarafa kacan konfigurasyonlari cezalandirmaz. Ornegin
Precision=0.90 / Recall=0.30 icin aritmetik ortalama 0.60 verir, harmonik
ortalama 0.45. Ikinci deger durumu daha dogru yansitir, cunku tek bacagi
topal bir retrieval sistemi pratikte kotudur.

Bu, bilgi erisimi literaturunde precision-recall ikilisi icin standart
ozetleme yontemidir.

Faithfulness ve Answer Relevancy icin benzer bir bilesim yapilmadi; bu iki
metrik arasinda yapisal bir denge iliskisi yok, harmonik ortalamanin teorik
dayanagi olmaz.

## Gozlemler

**Precision-Recall takasi net sekilde gozlendi.** k arttikca Precision
monoton dusuyor (0.8444 -> 0.7184), Recall monoton yukseliyor
(0.5479 -> 0.7535). Ders kitabi davranisi.

**k=3 en kotu secenek.** Precision en yuksek degerine ulassa da Recall
cokuyor (0.5479) ve bu Faithfulness'i da asagi cekiyor (0.7079, tum
seride en dusuk). Yorum: baglamda yeterli bilgi olmadiginda model
bosluklari doldurmaya calisiyor. "Az ama isabetli" stratejisi bu veri
setinde ise yaramiyor.

**Answer Relevancy k=10'da tepe yapip k=20'de belirgin dusuyor**
(0.6812 -> 0.6291, -0.052). Bu, gurultu bandinin cok uzerinde bir kayip.
Muhtemel aciklama: uzun baglamda modelin ortadaki bilgiyi kacirma egilimi
("lost in the middle"). 20 chunk'lik baglamda model odagini kaybediyor ve
cevap dagiliyor.

**Faithfulness'ta net egilim yok** (0.7079 / 0.7377 / 0.7245 / 0.7475).
Farklarin cogu gurultu bandina yakin. k parametresinin halusinasyon uzerinde
belirgin bir etkisi gozlenmedi.

## Karar: k = 10

Uc gerekce:

1. **Retrieval F1 tepe noktasi.** F1 k=10'da maksimum (0.7383). k=20'de
   artis durdu, hafif geriledi (0.7353). k=10 -> k=20 gecisinde Recall
   kazanci +0.018 (gurultu bandina yakin), Precision kaybi -0.022. Yani
   ek maliyetin karsiligi yok.

2. **Answer Relevancy k=10'da en yuksek.** k=20'deki -0.052'lik dusus
   anlamli bir bozulma. Uretim kalitesi acisindan k=10 belirgin sekilde
   ustun.

3. **Maliyet.** k=20 olcumu 111 dakika surdu, k=10'un 1.6 kati. Marjinal
   fayda yokken bu artis kabul edilemez. Ayrica her istekte iki kat token
   tuketimi demek.

Bu karar sonrasi `config.py` varsayilanlari `RETRIEVE_K = 10`, `TOP_K = 10`
olarak guncellendi. Sonraki deneyler bu deger sabit tutularak yapildi.

## Metodolojik not

k icin 3, 5, 10, 20 degerleri tarandi; aradaki her tam sayi denenmedi.
Gerekce: olcum gurultusu ±0.012 seviyesinde ve ardisik k degerleri arasindaki
farkin bu bandin altinda kalmasi bekleniyor. Ornegin k=5 -> k=10 arasindaki
F1 farki 0.025; tek adimlik farklar bunun besde biri mertebesinde olacagi
icin ayirt edilemezdi. Genis araliklarla tarama yapip egrinin seklini
belirlemek, ayni bilgiyi ucte bir surede veriyor.

---

# Deney serisi 2: chunk boyutu

Sabit ayarlar: k=10, temperature=0, reranker yok, hybrid search yok.
Overlap her deneyde chunk_size'in %10'u olarak orantili tutuldu.

## chunk_size = 200 (overlap 20)

| Metrik | chunk=500 | chunk=200 | Fark |
|---|---|---|---|
| Context Precision | 0.7406 | 0.6717 | -0.069 |
| Context Recall | 0.7360 | 0.4881 | -0.248 |
| Faithfulness | 0.7245 | 0.6656 | -0.059 |
| Answer Relevancy | 0.6812 | 0.6335 | -0.048 |
| Retrieval F1 | 0.7383 | 0.5653 | -0.173 |

Dort metrik de dustu; hicbir tarafta kazanc yok. Bu beklenmedik bir sonuc:
kucuk chunk'in en azindan Precision'i artirmasi beklenirdi.

**Yorum.** 200 karakter, anlamsal butunluk esiginin altinda kaliyor. Chunk
artik tam bir fikri barindiramiyor; cumlenin ortasindan baslayip ortasinda
bitiyor. Hakem model "bu chunk soruyla alakali mi?" degerlendirmesini
yaparken parcayi tek basina anlamsiz buluyor ve alakasiz sayiyor. Bu,
Precision'in da dusmesini acikliyor.

Recall'daki cokus (-0.248) daha dogrudan: k=10 sabit oldugu icin model yine
10 chunk goruyor, ancak her biri cok kucuk. Toplam baglam miktari 500'luk
konfigurasyonun yarisindan az kaliyor. Bilgi hem parcalanmis hem yetersiz.

**Cikarim:** Chunking yalnizca bir boyut meselesi degil; bir anlamsal
butunluk esigi var. 200 karakter bu veri setinde o esigin altinda.

chunk=300 denenmedi. Egri 200 yonunde acikca asagi gittigi icin 300'un de
500'un altinda kalmasi bekleniyor; olcum butcesi 500 ustu degerlere ayrildi.

## chunk_size = 800 (overlap 80)

| Metrik | chunk=200 | chunk=500 | chunk=800 |
|---|---|---|---|
| Context Precision | 0.6717 | 0.7406 | **0.8240** |
| Context Recall | 0.4881 | **0.7360** | 0.7227 |
| Faithfulness | 0.6656 | **0.7245** | 0.7232 |
| Answer Relevancy | 0.6335 | **0.6812** | 0.6290 |
| Retrieval F1 | 0.5653 | 0.7383 | **0.7701** |

**Precision belirgin sekilde artti (+0.083).** Yorum: 800 karakter, veri
setindeki tipik bir forum cevabini butun halinde barindiracak buyuklukte.
Hakem model chunk'i degerlendirirken kendi icinde tutarli bir metin goruyor;
alaka karari netlesiyor.

**Recall neredeyse degismedi (-0.013, gurultu bandinda).** Beklenti Recall'in
dusmesiydi -- buyuk chunk'larda k=10 ile toplam kapsanan dokuman sayisi
azalir. Ancak bu gerceklesmedi. Corpus'ta 30 bagimsiz blok bulundugu ve her
sorunun cevabi kendi blogunda oldugu icin, buyuk chunk'lar dogru blogu daha
butun halinde yakaliyor olabilir.

**Retrieval F1 0.7701 ile serinin en iyisi** (onceki en iyi: k=10,
chunk=500'de 0.7383).

**Answer Relevancy dustu (-0.052).** Bu, k=20 deneyinde gozlenen dususle ayni
buyuklukte. Ortak nokta: toplam baglam uzunlugu. k=10 × 800 karakter ≈ 8000
karakterlik baglam olusuyor. Modelin uzun baglamda odagini kaybetmesi
("lost in the middle") tutarli bir aciklama sunuyor.

Bu gozlem, chunk boyutu ve k parametresinin bagimsiz olmadigini gosteriyor:
belirleyici olan ikisinin carpimi, yani toplam baglam uzunlugu. Bir sonraki
deney bu hipotezi test ediyor (chunk=800, k=5).

## chunk_size = 800, k = 5

Onceki deneyde ortaya cikan hipotez: belirleyici olan chunk boyutu veya k
degil, ikisinin carpimi -- yani modele verilen toplam baglam uzunlugu. Bu
deney hipotezi test ediyor: chunk buyuk kalirken k yariya indiriliyor.

| Metrik | 500/k5 | 500/k10 | 800/k10 | **800/k5** |
|---|---|---|---|---|
| Context Precision | 0.8048 | 0.7406 | 0.8240 | **0.8528** |
| Context Recall | 0.6426 | **0.7360** | 0.7227 | 0.7073 |
| Faithfulness | 0.7377 | 0.7245 | 0.7232 | **0.7600** |
| Answer Relevancy | 0.6555 | 0.6812 | 0.6290 | **0.6977** |
| Retrieval F1 | 0.7137 | 0.7383 | 0.7701 | **0.7733** |

**Hipotez dogrulandi.** Toplam baglam 8000 karakterden (800×10) 4000
karaktere (800×5) inince:

- Answer Relevancy 0.6290 -> 0.6977 (+0.069)
- Faithfulness 0.7232 -> 0.7600 (+0.037)
- Context Precision 0.8240 -> 0.8528 (+0.029)

Uc metrik de anlamli sekilde iyilesti. Recall -0.015 ile gurultu bandinda
kaldi, yani kayip yok denecek kadar az.

**Yorum.** Model daha az ama daha dolu baglamla belirgin sekilde daha odakli
calisiyor. 8000 karakterlik baglamda gozlenen dagilma ("lost in the middle"),
4000 karakterde ortadan kalkiyor. Ayni zamanda az gurultu, sapma firsatini
azaltarak Faithfulness'i yukseltiyor.

**Baseline'a gore toplam kazanc** (500/k5 -> 800/k5):

| Metrik | Fark |
|---|---|
| Context Precision | +0.048 |
| Context Recall | +0.065 |
| Faithfulness | +0.022 |
| Answer Relevancy | +0.042 |
| Retrieval F1 | +0.060 |

Dort metrik de yukari. Tek metrigi digerleri pahasina yukselten bir
optimizasyon degil, genel bir iyilesme.

## chunk_size = 1000, k = 4

Toplam baglam yine ~4000 karakter, ancak parcalar daha buyuk. Amac: "anlamsal
butunluk" hipotezini bir adim daha zorlamak.

| Metrik | 800/k5 | 1000/k4 | Fark |
|---|---|---|---|
| Context Precision | 0.8528 | **0.8741** | +0.021 |
| Context Recall | 0.7073 | **0.7307** | +0.023 |
| Retrieval F1 | 0.7733 | **0.7960** | +0.023 |
| Faithfulness | **0.7600** | 0.6661 | -0.094 |
| Answer Relevancy | **0.6977** | 0.6182 | -0.080 |

**Retrieval iyilesti, generation bozuldu.** Her iki retrieval metrigi de
yukseldi ve F1 serinin en iyisine ulasti (0.7960). Buna karsilik Faithfulness
ve Answer Relevancy gurultu bandinin cok uzerinde dustu.

**Yorum.** 1000 karakterlik chunk'lar retrieval acisindan avantajli: parca
butun bir cevabi iceriyor, hakem model alaka degerlendirmesini kolay yapiyor.
Ancak ayni buyukluk generation icin dezavantaj: her chunk birden fazla fikir
ve alakasiz bolumler barindiriyor. Model dogru bilgiyi ayiklamak icin dort
uzun metni taramak zorunda kaliyor ve bu sirada baglamdan sapiyor.

**Cikarim:** Chunk boyutunun retrieval optimumu ile generation optimumu ayni
noktada degil. Retrieval metriklerini iyilestirmek sistemin genel kalitesini
iyilestirmeyi garanti etmiyor. Bu, yalnizca F1 veya yalnizca retrieval
skorlarina bakarak optimizasyon yapmanin yaniltici olabilecegini gosteriyor.

## Chunk serisi karari: 800 / k=5

| Metrik | 200/k10 | 500/k5 | 500/k10 | 800/k10 | **800/k5** | 1000/k4 |
|---|---|---|---|---|---|---|
| Ctx Precision | 0.6717 | 0.8048 | 0.7406 | 0.8240 | 0.8528 | **0.8741** |
| Ctx Recall | 0.4881 | 0.6426 | **0.7360** | 0.7227 | 0.7073 | 0.7307 |
| Faithfulness | 0.6656 | 0.7377 | 0.7245 | 0.7232 | **0.7600** | 0.6661 |
| Answer Rel. | 0.6335 | 0.6555 | 0.6812 | 0.6290 | **0.6977** | 0.6182 |
| Retrieval F1 | 0.5653 | 0.7137 | 0.7383 | 0.7701 | 0.7733 | **0.7960** |

800/k5 secildi. Gerekce: dort metrikten ucunde en yuksek deger. 1000/k4
retrieval F1'de 0.023 onde, ancak Faithfulness ve Answer Relevancy'de
0.08-0.09 geride. Hedef dort metrigi birlikte optimize etmek
oldugu icin retrieval F1'i tek basina maksimize etmek dogru olmaz.

---

# Deney serisi 3: Hybrid search (BM25 + dense)

Sabit: chunk=800, overlap=80, k=5. BM25 ve vektor aramasi ayri ayri
calistirilip sonuclar Reciprocal Rank Fusion (RRF) ile birlestirildi.
Her iki yontemden k×2 aday cekilip birlesim sonrasi ilk k tanesi alindi.

| Metrik | Dense (800/k5) | Hybrid | Fark |
|---|---|---|---|
| Context Precision | **0.8528** | 0.8001 | -0.053 |
| Context Recall | **0.7073** | 0.6911 | -0.016 |
| Faithfulness | **0.7600** | 0.7188 | -0.041 |
| Answer Relevancy | **0.6977** | 0.6517 | -0.046 |
| Retrieval F1 | **0.7733** | 0.7416 | -0.032 |

(Answer Relevancy'de 1 NaN: hakem modelin cikti formati bir soruda
ayristirilamadi. Skor 29 soru uzerinden. Diger metrikler temiz.)

**Sonuc: hybrid search bu veri setinde performansi dusurdu.**

**Yorum.** Beklenti, BM25'in embedding'in zayif oldugu alanlari (kod,
kisaltma, ozel isim) kapatmasiydi. Gerceklesmedi. Olasi aciklama: corpus
forum cevaplarindan olusuyor ve yaygin finansal kelimeler ("account",
"bank", "money", "business") her yerde geciyor. BM25 bu kelimelerin
yogunlastigi chunk'lari one cikariyor, ancak yaygin olduklari icin ayirt
edici degiller. Kelime eslesmesi burada anlamsal eslesmeden daha zayif
sinyal uretiyor.

Ikinci etken RRF'in esit agirlikli birlestirme yapmasi. Dense arama tek
basina yeterince iyi calisirken, zayif bir ikinci sinyali esit agirlikla
karistirmak siralamayi bozuyor.

**Karar:** Hybrid search kullanilmiyor. Yalnizca dense (vektor) arama ile
devam edildi. `USE_HYBRID_SEARCH = False`.

---

# Vektor veritabani karsilastirmasi

RAG sistemlerinde genellikle ChromaDB, FAISS, LanceDB gibi hazir vektor
veritabanlari kullaniliyor. Proje kapsaminda NumPy ile exact search tercih edildi. Bu karar
olculerek dogrulandi: ayni 138 chunk ve ayni vektorlerle ChromaDB kurulup
30 sorgu her iki yontemde calistirildi.

| Yontem | Sorgu suresi (ort.) |
|---|---|
| NumPy (exact, brute-force) | 0.203 ms |
| ChromaDB (HNSW) | 4.374 ms |

- NumPy 21 kat daha hizli
- Sonuc ortusme orani: %100 (iki yontem ayni chunk'lari donduruyor)
- ChromaDB indeksleme suresi: 1.07 saniye

Olcum, `retrieve_k = 5` ayariyla yapildi (reranker deneyleri oncesi, chunk
serisi karari sonrasi). Ikinci bir calistirmada NumPy 0.282 ms, ChromaDB
5.167 ms olculdu; oran korundu.

**Yorum.** ANN (yaklasik en yakin komsu) indeksleri, milyonlarca vektor
icinde arama yaparken devreye giren bir optimizasyon. 138 vektorde ise
indeksin kendisi ek yuk olusturuyor: ChromaDB'nin yavasligi HNSW
algoritmasindan degil, katman maliyetinden kaynaklaniyor (SQLite kalicilik,
serilestirme, API cagrisi). Bu olcekte o maliyet, aramanin kendisinden buyuk.

Ayrica exact search kesin sonuc verir; ANN yaklasik calisir. Bu olcekte
yaklasiklik icin bir sebep yok.

**Karar:** NumPy tabanli exact search kullanildi. Karar performans
gerekcesinin yani sira bagimlilik azaltma ve tam kontrol saglama acisindan da
tercih edildi.

---

# Reranker model secimi

Servis uzerinde iki reranker mevcut: `Qwen3-Reranker-8B` (ilk
tercih) ve `bge-reranker-v2-m3`. Ikisi de `/rerank` endpoint'i uzerinden
Cohere uyumlu formatta calisiyor.

Format dogrulamasi icin kucuk bir test yapildi. Sorgu: "How do I deposit a
third party cheque?" Uc aday dokuman verildi.

| Dokuman | Qwen3-Reranker-8B | bge-reranker-v2-m3 |
|---|---|---|
| "The weather in Ankara is cold in winter." | **0.903** (1.) | 0.000017 (3.) |
| "Just have the associate sign the back..." | 0.880 (2.) | 0.023 (2.) |
| "A third party cheque requires endorsement..." | 0.562 (3.) | **0.807** (1.) |

**Qwen3-Reranker-8B alakasiz sonuc uretti.** Konuyla hicbir ilgisi olmayan
hava durumu cumlesini en alakali, dogrudan cevabi iceren cumleyi ise en
alakasiz olarak siralamis. Skorlarin birbirine yakinligi (0.90 / 0.88 / 0.56)
da anlamli bir ayrim yapamadigini gosteriyor.

**bge-reranker-v2-m3 dogru calisti.** Dogrudan cevabi 0.807 ile birinci
sirada, alakasiz cumleyi 0.000017 ile son sirada dondurdu. Skorlar arasindaki
buyukluk farki net bir ayrim isareti.

**Olasi aciklama:** Qwen3-Reranker ailesi ozel bir prompt sablonu bekliyor
(`Instruct: ... Query: ... Document: ...` bicimi). Servis ham metni dogrudan
iletiyor olabilir; model bu durumda anlamli skor uretmiyor.

**Karar:** `bge-reranker-v2-m3` kullanildi. Model secimi olcume dayali olarak
degistirildi.

---

# Deney serisi 4: Reranker

Sabit: chunk=800, overlap=80, hybrid kapali. Model: `bge-reranker-v2-m3`.
Yaklasim: vektor aramasindan retrieve_k aday cekilir, cross-encoder ile
yeniden siralanip top_k tanesi LLM'e verilir.

## retrieve_k = 20 -> top_k = 5

| Metrik | Reranker yok (800/k5) | Reranker (20→5) | Fark |
|---|---|---|---|
| Faithfulness | 0.7600 | **0.7850** | +0.025 |
| Answer Relevancy | 0.6977 | **0.7219** | +0.024 |
| Context Precision | **0.8528** | 0.8081 | -0.045 |
| Context Recall | **0.7073** | 0.6858 | -0.022 |
| Retrieval F1 | **0.7733** | 0.7419 | -0.031 |

**Beklenti gerceklesmedi.** Reranker'in amaci genis bir aday havuzundan
(Recall yuksek) eleme yaparak Precision'i korumakti. Her iki retrieval metrigi
de dustu.

**Buna karsilik generation belirgin iyilesti.** Faithfulness ve Answer
Relevancy ikisi de yaklasik +0.025 artti; her iki fark da gurultu bandinin
(±0.012) uzerinde. Faithfulness 0.7850 ve Answer Relevancy 0.7219, seride
o ana kadarki en yuksek degerler.

**Yorum.** Ragas'in retrieval metrikleri ile uretim kalitesi arasinda bir
ayrisma gozlendi. Reranker, hakem modelin "alakali" olarak isaretledigi
chunk'lari degil, modelin cevap uretmesine fiilen yarayan chunk'lari one
cikariyor olabilir. Bu ikisi her zaman ortusmuyor.

Ayni ayrisma chunk=1000/k=4 deneyinde ters yonde gozlenmisti: orada retrieval
metrikleri yukselirken generation cokmustu. Iki bulgu birlikte, tek basina
retrieval skorlarina bakarak optimizasyon yapmanin yaniltici oldugunu
gosteriyor.

## retrieve_k = 10 -> top_k = 5

| Metrik | 20→5 | 10→5 |
|---|---|---|
| Faithfulness | 0.7850 | 0.7848 |
| Answer Relevancy | 0.7219 | 0.7220 |
| Context Precision | **0.8081** | 0.7924 |
| Context Recall | **0.6858** | 0.6682 |
| Retrieval F1 | **0.7419** | 0.7250 |

Generation metrikleri neredeyse birebir ayni (fark 0.0002 mertebesinde). Bu,
reranker'in her iki aday havuzundan da buyuk olcude ayni chunk'lari sectigini
gosteriyor: 20 aday da 10 aday da ayni ust siralara cikiyor.

Retrieval metrikleri dar havuzda bir miktar daha dusuk. Genis aday havuzu
tercih edildi.

## Reranker karari

| Metrik | Rerankersiz (800/k5) | Reranker (20→5) |
|---|---|---|
| Faithfulness | 0.7600 | **0.7850** |
| Answer Relevancy | 0.6977 | **0.7219** |
| Context Precision | **0.8528** | 0.8081 |
| Context Recall | **0.7073** | 0.6858 |
| Retrieval F1 | **0.7733** | 0.7419 |
| Dort metrik ortalamasi | 0.7545 | 0.7502 |

Dort metrigin duz ortalamasi neredeyse esit. Karar, hangi metriklere oncelik
verildigine bagli.

**Reranker kullanilmasina karar verildi.** Gerekce: Faithfulness ve Answer
Relevancy son kullanicinin dogrudan deneyimledigi ciktiyi olcuyor. Context
Precision ve Recall ara asama metrikleri; kullanici getirilen chunk'larin
isabetiyle degil, aldigi cevabin dogrulugu ve konuyla ilgisiyle ilgileniyor.
Faithfulness dogrudan halusinasyon kontrolu islevi goruyor
ve finansal alanda uydurma cevabin maliyeti yuksek.

Bu karar tartismaya aciktir; her iki konfigurasyonun skorlari raporda birlikte
sunulmaktadir.

Secilen konfigurasyon: `retrieve_k=20`, `top_k=5`, `bge-reranker-v2-m3`.

---

# Deney serisi 5: Prompt muhendisligi

Sabit: chunk=800, retrieve_k=20 → top_k=5, reranker acik.

## Varyant "strict" — kati sadakat kurallari

Amac Faithfulness'i yukseltmekti. Baseline prompt'a su kurallar eklendi:
"Cikarim yapma, genelleme yapma, baglamin belirtmedigi sonuclara ulasmak icin
bilgileri birlestirme." Ayrica "sorulmayan arka plan bilgisi ekleme"
talimati verildi.

| Metrik | baseline | strict | Fark |
|---|---|---|---|
| Faithfulness | **0.7850** | 0.5986 | -0.186 |
| Answer Relevancy | **0.7219** | 0.4720 | -0.250 |
| Context Precision | 0.8081 | **0.8174** | +0.009 |
| Context Recall | **0.6858** | 0.6487 | -0.037 |

**Hedeflenen metrik ters yonde hareket etti.** Faithfulness'i yukseltmek icin
yazilan prompt onu 0.186 dusurdu.

**Mekanizma.** "Bilgi yok" yaniti veren soru sayisi 6'dan 12'ye cikti
(sorularin %40'i). Ornek olarak q_000'de baglam dogru cevabi iceriyor
(ucuncu taraf ceki imzalanip yatirilir), ancak model soyle yanit verdi:

> "Baglam, bunun bir ortaga kesilmis bir cekin isletme hesabina
> yatirilmasina uygulanip uygulanmadigini dogrulamiyor."

Model "cikarim yapma" talimatini asiri yorumlayarak, baglamdaki bilgiyi
soruya baglamak icin gereken en kucuk adimi bile atmayi reddetti.

**Faithfulness'in dusme sebebi.** Ragas bu metrigi "cevaptaki iddialarin
kaci baglamdan destekleniyor" seklinde hesapliyor. Model "baglam bunu
dogrulamiyor", "acikca belirtmiyor" gibi meta-iddialar uretiyor. Bunlar
baglam hakkinda yorumlar; baglamdan cikarilabilir olgusal iddialar degil.
Hakem model bunlari desteklenmemis sayiyor ve skor dusuyor.

**Cikarim:** Asiri kisitlayici prompt, hedefledigi metrigi bozabiliyor.
Halusinasyonu onlemek ile modeli islevsiz kilmak arasindaki denge dar.
Baseline prompt'taki "bilgi yoksa soyle" izni yeterli; buna ek olarak
cikarim yasagi getirmek sistemin cevap uretme yetenegini ortadan kaldiriyor.

## Varyant "cited" — kaynak gosterimi

Farkli bir mekanizma denendi: kisitlama getirmek yerine modelden her iddianin
sonuna hangi baglam blogundan geldigini `[1]`, `[2]` biciminde yazmasi
istendi.

| Metrik | baseline | strict | **cited** |
|---|---|---|---|
| Faithfulness | 0.7850 | 0.5986 | **0.8244** |
| Answer Relevancy | **0.7219** | 0.4720 | 0.7025 |
| Context Precision | 0.8081 | **0.8174** | 0.8173 |
| Context Recall | 0.6858 | 0.6487 | **0.7025** |
| Retrieval F1 | 0.7419 | 0.7233 | 0.7556 |
| Dort metrik ortalamasi | 0.7502 | 0.6342 | **0.7617** |

**Faithfulness projedeki en yuksek degere ulasti** (0.8244, +0.039). Gurultu
bandinin uzerinde, gercek kazanc.

**Mekanizma.** Model her iddianin kaynagini belirtmek zorunda kaldiginda,
baglamda karsiligi olmayan bir sey soylemesi zorlasiyor. Alinti zorunlulugu,
uretim sirasinda surekli baglama donmeyi gerektiriyor.

Answer Relevancy hafif dustu (-0.019, gurultu bandina yakin); muhtemelen
`[1]`, `[2]` isaretlerinin metne karismasindan kaynaklaniyor.

Context Recall artti (+0.017). Retrieval degismedigi icin bu dogrudan bir
retrieval etkisi degil: model daha fazla bloktan alinti yaptigi icin ground
truth iddialarinin daha buyuk bolumu cevapta karsilik buluyor olabilir.

## Prompt deneylerinden cikan sonuc

Iki varyant zit yonde sonuc verdi ve aradaki fark ogretici:

- **strict** modele *ne yapamayacagini* soyledi (cikarim yapma, birlestirme,
  genelleme yapma). Model islevsizlesti, %40 oraninda cevap vermeyi reddetti,
  hedeflenen metrik 0.186 dustu.
- **cited** modele *nasil yapmasi gerektigini* gosterdi (her iddiayi
  kaynagiyla birlikte yaz). Faithfulness 0.039 artti.

Kisitlama getirmek yerine yapi getirmek daha etkili oldu.

---

# Deney serisi 6: Corpus temizligi

Veri setindeki context bloklari birden fazla forum cevabinin birlestirilmis
hali ve birlesme noktalarinda noktalama bozukluklari var. Corpus taramasi:

| Bozukluk | Adet |
|---|---|
| Cift tirnak (`""`) | 104 |
| Tirnak sonrasi bosluksuz harf | 87 |
| Coklu bosluk | 260 |
| Bosluksuz cumle sonu (`.X`) | 25 |

Ornek: `...Before you convert to S-Corp.You don't need to notify the IRS...`

Bu bozukluklar recursive chunking'in `". "` ayiricisini devre disi birakiyor;
algoritma bir alt seviyeye (virgul, bosluk) dusuyor ve chunk sinirlari cumle
ortasindan geciyor.

Uygulanan temizlik: cift tirnaklarin tekile indirilmesi, cumle sonu
noktalamasindan sonra bosluk eklenmesi, tirnak sonrasi bosluk eklenmesi,
coklu bosluklarin teke indirilmesi.

Sonuc: 138 chunk -> 137 chunk. Chunk'larin %85'inin metni degisti.

| Metrik | Temizliksiz | **Temizlikli** | Fark |
|---|---|---|---|
| Faithfulness | 0.8244 | **0.8785** | +0.054 |
| Answer Relevancy | 0.7025 | **0.7659** | +0.063 |
| Context Precision | **0.8173** | 0.7866 | -0.031 |
| Context Recall | **0.7025** | 0.6598 | -0.043 |
| Retrieval F1 | **0.7556** | 0.7176 | -0.038 |
| Dort metrik ortalamasi | 0.7617 | **0.7727** | +0.011 |

**Generation metrikleri belirgin sekilde iyilesti.** Faithfulness 0.8785 ve
Answer Relevancy 0.7659, projedeki en yuksek degerler.

**Yorum.** Chunk sinirlari artik cumle ortasindan gecmiyor. Model butun
cumleler okuyor, parcalanmis ifadelerle ugrasmiyor. Bozuk tirnaklarin
temizlenmesi de metnin okunabilirligini artiriyor.

Retrieval metriklerindeki dusus, chunk sinirlarinin degismesiyle ground truth
eslesmesinin kaymasindan kaynaklaniyor. Ayrica eklenen bosluklar metni
uzattigi icin ayni 800 karaktere bir miktar daha az icerik siginiyor.

**Karar: temizlik uygulanacak.** Gerekce, reranker kararindaki ile ayni:
generation metrikleri kullanicinin dogrudan deneyimledigi ciktiyi olcuyor.
Buradaki fark daha da belirgin (generation +0.05 ve +0.06).

## top_k = 3 (reranker ile)

Hipotez: reranker zaten en iyi adaylari sectigine gore, daha az chunk vermek
gurultuyu azaltip Faithfulness'i yukseltebilir.

| Metrik | top_k=5 | top_k=3 |
|---|---|---|
| Faithfulness | **0.8785** | 0.8492 |
| Answer Relevancy | 0.7659 | **0.7698** |
| Context Precision | 0.7866 | **0.8222** |
| Context Recall | **0.6598** | 0.6095 |
| Retrieval F1 | **0.7176** | 0.7000 |
| Ortalama | **0.7727** | 0.7627 |

**Hipotez dogrulanmadi.** Faithfulness beklenenin tersine dustu (-0.029).
Recall'daki -0.050'lik dusus sebebi acikliyor: uc chunk yetersiz kaliyor,
baglamda cevabin bir bolumu eksik ve model bosluk doldurmaya calisiyor.

Bu, k taramasinda gozlenen kalibin tekrari: cok az baglam Faithfulness'i
dusuruyor (k=3 deneyinde de en dusuk Faithfulness gorulmustu). Yeterli bilgi
olmadan model sadik kalamiyor.

top_k=5 korundu.

## Reranker skor esigi

Reranker skorlarinin dagilimi incelendi: medyan 0.0268, minimum 0.0003,
maksimum 0.9987. Modele verilen chunk'larin yarisi reranker'a gore neredeyse
alakasiz.

Hipotez: sabit top_k yerine skor esigi kullanarak baglam sayisini dinamik
belirlemek. Alakali chunk cok ise cok, az ise az verilir.

Esik = 0.1 uygulandiginda ortalama baglam sayisi 5'ten 2.2'ye dustu
(min 1, max 5).

| Metrik | Esiksiz | Esik 0.1 |
|---|---|---|
| Faithfulness | **0.8785** | 0.8602 |
| Answer Relevancy | 0.7659 | **0.7681** |
| Context Precision | 0.7866 | **0.8046** |
| Context Recall | **0.6598** | 0.5582 |
| Retrieval F1 | **0.7176** | 0.6591 |
| Ortalama | **0.7727** | 0.7478 |

**Hipotez dogrulanmadi.** Context Recall -0.102 ile agir kayip verdi.
Ortalama 2.2 chunk yetersiz kaliyor ve gerekli bilginin onemli bolumu
eleniyor. Faithfulness da dustu (-0.018), top_k=3 deneyindeki mekanizmanin
tekrari.

Precision'daki +0.018 kazanc bu kayiplari karsilamiyor.

**Cikarim:** Reranker skorlarinin mutlak degerleri guvenilir bir alaka olcusu
degil. Dusuk skorlu bir chunk, siralamada geride olsa da cevabin bir parcasini
tasiyabiliyor. Goreli siralama (top_k) mutlak esikten daha saglikli calisiyor.

---

# Final konfigurasyon ve toplam kazanc

| Parametre | Baseline | Final |
|---|---|---|
| Chunk boyutu | 500 | 800 |
| Overlap | 50 | 80 |
| Metin temizligi | Yok | Var |
| retrieve_k | 5 | 20 |
| top_k | 5 | 5 |
| Reranker | Yok | bge-reranker-v2-m3 |
| Hybrid search | Yok | Yok (olculdu, reddedildi) |
| Prompt | baseline | cited |
| Temperature | 0.0 | 0.0 |

| Metrik | Baseline | Final | Fark |
|---|---|---|---|
| Faithfulness | 0.7377 | **0.8785** | +0.141 |
| Answer Relevancy | 0.6555 | **0.7659** | +0.110 |
| Context Precision | 0.8048 | 0.7866 | -0.018 |
| Context Recall | 0.6426 | 0.6598 | +0.017 |

Toplam 14 olcum yapildi; hepsinde NaN sayisi sifir.

## Genel cikarimlar

**1. Retrieval metrikleri ile uretim kalitesi her zaman ayni yone gitmiyor.**
Uc ayri deneyde bu ayrisma gozlendi: chunk=1000/k=4 (retrieval yukseldi,
generation coktu), reranker (retrieval dustu, generation yukseldi), corpus
temizligi (ayni yonde). Yalnizca retrieval skorlarina bakarak optimizasyon
yapmak yaniltici.

**2. Toplam baglam uzunlugu, chunk boyutu ve k'dan daha belirleyici.**
Ikisinin carpimi ~4000 karakter civarinda tutuldugunda en iyi sonuclar alindi.
8000 karakterde "lost in the middle" etkisi gozlendi, 2400 karakterde bilgi
yetersiz kaldi.

**3. Prompt'ta kisitlama getirmek yerine yapi getirmek etkili.** Yasak
listesi (strict) modeli islevsizlestirdi; kaynak gosterimi (cited) hedeflenen
metrigi yukseltti.

**4. Veri kalitesi, parametre ayarindan daha buyuk kazanc sagladi.** Corpus
temizligi tek basina Faithfulness'a +0.054 katti; hicbir parametre degisimi
bu kadar etkili olmadi.

**5. Negatif sonuclar da sonuctur.** On deneyden ucu reddedildi (hybrid
search, top_k=3, skor esigi). Denemeden reddetmek yerine olcup gerekceyle
elemek, mimari kararlarin savunulabilirligini artiriyor.

## Sinirlamalar

- Veri seti 30 ornek iceriyor; istatistiksel guc dusuk.
- Hakem modeli tek (`openai/gpt-oss-120b`); farkli bir hakemle skorlar
  degisebilir.
- Olcum gurultusu ±0.012; bu bandin altindaki farklar yorumlanmadi.
- Context Recall 0.66 seviyesinde kaldi. Denenmemis bir yaklasim: sorgu
  genisletme (query expansion).
- Corpus'ta 30 bagimsiz blok var ve her sorunun cevabi kendi blogunda.
  Retrieval gorevi gercek dunya senaryolarina gore gorece kolay.
