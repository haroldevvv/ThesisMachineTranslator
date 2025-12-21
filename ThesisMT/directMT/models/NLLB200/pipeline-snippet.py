import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# 1. Load dataset
df = pd.read_csv("english_monolingual_corpus.csv")
english_data = df["english"].dropna().tolist()

# 2. Load forward and back-translation models
tokenizer_en_rin = AutoTokenizer.from_pretrained("en_to_rin_model",src_lang="eng_Latn",tgt_lang="rin_Latn")
model_en_rin = AutoModelForSeq2SeqLM.from_pretrained("en_to_rin_model").half().cuda()

tokenizer_rin_en = AutoTokenizer.from_pretrained("rin_to_en_model",src_lang="rin_Latn",tgt_lang="eng_Latn")
model_rin_en = AutoModelForSeq2SeqLM.from_pretrained("rin_to_en_model").half().cuda()

# 3. Sentence embedding model for semantic filtering
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cuda")

def batch_translate(model, tokenizer, texts, src_lang, tgt_lang, batch_size=128):
    """Batch translation for NLLB-200 with explicit language IDs."""
    outputs = []
    bos = tokenizer.convert_tokens_to_ids(tgt_lang)

    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True,truncation=True, max_length=128).to("cuda")
        generated = model.generate(**inputs, num_beams=5,forced_bos_token_id=bos, max_length=128)
        outputs.extend(tokenizer.batch_decode(generated, skip_special_tokens=True))
    return outputs

# 4. English to Rinconada & Rinconada to English Translation 
rinconada_sents = batch_translate(model_en_rin, tokenizer_en_rin, english_data,"eng_Latn", "rin_Latn")
backtranslations = batch_translate(model_rin_en, tokenizer_rin_en, rinconada_sents,"rin_Latn", "eng_Latn")

# 5. Semantic similarity using embeddings
similarities = []
emb_orig = embedder.encode(english_data, convert_to_tensor=True, normalize_embeddings=True)
emb_back = embedder.encode(backtranslations, convert_to_tensor=True, normalize_embeddings=True)
similarities = torch.sum(emb_orig * emb_back, dim=1).cpu().tolist()

# 6. Filtering
filtered = [rin if sim >= 0.70 else "" for rin, sim in zip(rinconada_sents, similarities)]

# 7. Saving high-quality synthetic data
df = pd.DataFrame({"english": english_data,"synthetic_rinconada": filtered,"similarity": similarities})
df.to_csv("synthetic_rinconada.csv", index=False)
