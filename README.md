# RAG Soru-Cevap Sistemi ve Ragas Degerlendirmesi

Uctan uca bir RAG (Retrieval-Augmented Generation) soru-cevap sistemi ve
Ragas kutuphanesi ile hiperparametre optimizasyonu. Sistem hazir RAG
framework'u kullanmadan (LangChain, LlamaIndex vb.) saf Python ile yazildi.

## Sonuclar

Baseline'dan final konfigurasyona kadar 16 farkli konfigurasyon, 17 olcum
yapildi. Tum olcumlerde NaN sayisi sifir.

| Metrik | Baseline | Final | Fark |
|---|---|---|---|
| Faithfulness | 0.7377 | **0.8785** | +0.141 |
| Answer Relevancy | 0.6555 | **0.7659** | +0.110 |
| Context Precision | 0.8048 | 0.7866 | -0.018 |
| Context Recall | 0.6426 | 0.6598 | +0.017 |

Tum olcumler ve karar gerekceleri: `results/benchmark_raporu.md`
Deney sirasinda tutulan notlar: `results/gozlemler.md`

## Final Konfigurasyon

| Parametre | Deger |
|---|---|
| Chunk boyutu | 800 karakter |
| Overlap | 80 karakter |
| Metin temizligi | Acik |
| Retrieval | Dense (vektor), exact search |
| retrieve_k | 20 |
| top_k | 5 |
| Reranker | bge-reranker-v2-m3 |
| Hybrid search | Kapali (olculdu, performansi dusurdu) |
| Prompt | Kaynak gosterimli (`cited`) |
| Temperature | 0.0 |

## Kullanilan Modeller

| Bilesen | Model |
|---|---|
| LLM | Qwen3-Next-80B-A3B-Instruct |
| Embedding | Qwen3-Embedding-8B (4096 boyut) |
| Reranker | bge-reranker-v2-m3 |
| Ragas hakem modeli | openai/gpt-oss-120b |

Modeller OpenAI uyumlu bir API uzerinden servis edilir (ornegin vLLM).
Reranker icin servisin Cohere uyumlu bir `/rerank` endpoint'i sunmasi gerekir.

## Veri Seti

Hugging Face `vibrantlabsai/fiqa`, config `ragas_eval_v3`.
30 finansal soru-cevap ornegi, ~88.400 karakterlik corpus.

## Kurulum

```
powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1
```

Elle kurulum:

```
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

API baglantisi icin `.env.example` dosyasi `.env` adiyla kopyalanip
doldurulmali:

```
LLM_BASE_URL=<api-adresi>
LLM_API_KEY=<token>
```

### Ortam notlari

- **Sertifika:** SSL denetimi yapan aglarda Python HTTPS isteklerinde
  sertifika dogrulama hatasi olusur. `pip-system-certs` paketi Windows
  sertifika deposunu kullanarak bunu cozer.
- **Surum sabitleme:** Ragas ve LangChain surumleri birbirine bagimli;
  `requirements.txt` icindeki kombinasyon dogrulanmistir.

## Proje Yapisi

```
src/        Ana moduller (chunking, embedding, retrieval, generation, evaluation)
scripts/    Calistirilabilir isler (corpus, indeksleme, deney, degerlendirme)
static/     Web arayuzu
data/raw/   Veri setinden cikarilan corpus ve sorular
results/    Deney sonuclari ve gozlem notlari
```

## Calistirma

### Ilk kurulum

```
python scripts\build_corpus.py       # Veri setini indirir
python scripts\build_index.py        # Temizlik + chunk + embedding + indeks
python scripts\test_connection.py    # Baglanti dogrulamasi
```

### Deney calistirma

```
python scripts\run_pipeline.py --retrieve-k 20 --top-k 5 --rerank --prompt cited --name deneyim
python scripts\evaluate.py --name deneyim
```

Sonuclar `results/pipeline_<ad>.json` ve `results/eval_<ad>.json` olarak
kaydedilir; her dosya kullanilan konfigurasyonu icerir.

### Web arayuzu

```
python -m uvicorn src.api:app --port 8000
```

Tarayicida `http://localhost:8000`. Soru sorulur; cevap ve kullanilan baglam
parcalari alaka skorlariyla birlikte gosterilir.

### Yardimci scriptler

```
python scripts\test_chunking.py       # Chunk parametrelerinin etkisi
python scripts\test_embeddings.py     # Embedding ve cache dogrulamasi
python scripts\test_retrieval.py      # Retrieval kalitesi
python scripts\test_reranker.py       # Reranker modeli karsilastirmasi
python scripts\inspect_corpus.py      # Corpus metin bozukluklari
python scripts\inspect_output.py      # Pipeline cikti kontrolu
python scripts\compare_vectordb.py    # NumPy vs ChromaDB (chromadb kurulu olmali)
```

## Mimari Kararlar

**Saf Python.** Hazir RAG framework'leri yerine her katman elle yazildi.
Gerekce: tam kontrol, soyutlama katmani olmadan hata ayiklama kolayligi ve
her parametrenin dogrudan olculebilmesi.

**Exact search.** 137 chunk olcegi icin ANN indeksi gereksiz. ChromaDB ile
karsilastirmali olculdu: NumPy tabanli exact search 18-21 kat daha hizli ve
sonuclar %100 ortusuyor.

**Metin temizligi.** Veri setindeki context bloklari birden fazla forum
cevabinin birlestirilmis hali; birlesme noktalarindaki noktalama
bozukluklari chunking'in dogal cumle ayiricilarini devre disi birakiyordu.
Temizlik tek basina Faithfulness'a +0.054 katti.

**Kaynak gosterimli prompt.** Modelden her iddianin kaynagini belirtmesi
istendi. Faithfulness 0.039 artti. Bunun yerine denenen kisitlayici prompt
("cikarim yapma") ise metrigi 0.186 dusurdu.

**Reranker model degisikligi.** Ilk secenek olan Qwen3-Reranker-8B test
edildiginde alakasiz siralama uretti (bkz. `results/gozlemler.md`);
bge-reranker-v2-m3 kullanildi.
