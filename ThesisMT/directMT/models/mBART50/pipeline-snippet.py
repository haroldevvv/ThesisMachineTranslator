import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# 1. Load dataset
df = pd.read_csv("english_monolingual_corpus.csv")
english_data = df["english"].dropna().tolist()

# 2. Load forward and back-translation models
tokenizer_en_rin = AutoTokenizer.from_pretrained("en_to_rin_model")
model_en_rin = AutoModelForSeq2SeqLM.from_pretrained("en_to_rin_model").half().cuda()

tokenizer_rin_en = AutoTokenizer.from_pretrained("rin_to_en_model")
model_rin_en = AutoModelForSeq2SeqLM.from_pretrained("rin_to_en_model").half().cuda()

# 3. Sentence embedding model for semantic filtering
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cuda")

def batch_translate(model, tokenizer, sentences, tgt_lang=None, batch_size=128):
    """mBART50 batch translation with optional forced BOS language."""
    translations = []
    bos = tokenizer.lang_code_to_id[tgt_lang] if tgt_lang else None

    for i in tqdm(range(0, len(sentences), batch_size)):
        batch = sentences[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True,truncation=True).to("cuda")
        outputs = model.generate(**inputs, num_beams=5, max_length=128,forced_bos_token_id=bos)
        translations.extend(tokenizer.batch_decode(outputs, skip_special_tokens=True))
    return translations

# 4. English to Rinconada & Rinconada to English Translation 
rinconada_sents = batch_translate(model_en_rin, tokenizer_en_rin, english_data)
back_sents = batch_translate(model_rin_en, tokenizer_rin_en, rinconada_sents, tgt_lang="en_XX")

# 5. Semantic similarity using embeddings
emb_orig = embedder.encode(english_data, convert_to_tensor=True, normalize_embeddings=True)
emb_back = embedder.encode(back_sents, convert_to_tensor=True, normalize_embeddings=True)
similarities = torch.sum(emb_orig * emb_back, dim=1).cpu().tolist()

# 6. Filtering 
filtered = [rin if sim >= 0.70 else "" for rin, sim in zip(rinconada_sents, similarities)]

# 7. Save final dataset
df = pd.DataFrame({"english": english_data,"rinconada": filtered,"similarity": similarities})
df.to_csv("synthetic_rinconada.csv", index=False)
