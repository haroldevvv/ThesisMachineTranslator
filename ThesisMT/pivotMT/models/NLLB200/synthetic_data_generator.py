# synthetic_data_generator_tl2rin.py
import os
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Tagalog → Rinconada
model_dir = "/ThesisMT/pivotMT/models/NLLB200/runs/tl_rin_3.5k_15epochs_nllb_offload_run-20251003-165744/final_model"
tokenizer = AutoTokenizer.from_pretrained(model_dir, src_lang="tgl_Latn", tgt_lang="und_Latn")
model = AutoModelForSeq2SeqLM.from_pretrained(model_dir).to(device)

# Rinconada → Tagalog (back-translation)
back_model_dir = "/ThesisMT/pivotMT/models/NLLB200/runs/rin_tl_3.5k_15epochs_nllb_offload_run-20251003-165344/final_model"
back_tokenizer = AutoTokenizer.from_pretrained(back_model_dir, src_lang="und_Latn", tgt_lang="tgl_Latn")
back_model = AutoModelForSeq2SeqLM.from_pretrained(back_model_dir).to(device)

dataset_path = "/ThesisMT/Datasets/generated_synthetic_data/pivot_translation/NLLB200/tl_100K_NLLB200.csv"
df = pd.read_csv(dataset_path)

df = df.rename(columns={"tl": "sentence"})
df = df[df["sentence"].notna()]
df["sentence"] = df["sentence"].astype(str).str.strip()

embedder = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    device="cuda:0" if torch.cuda.is_available() else "cpu"
)

def batch_translate(model, tokenizer, sentences, tgt_lang, max_length=128, batch_size=32):
    translations = []
    for i in tqdm(range(0, len(sentences), batch_size), desc=f"Translating → {tgt_lang}"):
        batch = sentences[i:i+batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        ).to(device)

        outputs = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_lang),
            max_length=max_length,
            num_beams=5,
            early_stopping=True
        )
        decoded = [tokenizer.decode(t, skip_special_tokens=True) for t in outputs]
        translations.extend(decoded)
    return translations

print("Translating Tagalog → Rinconada...")
rinconada_sentences = batch_translate(model, tokenizer, df["sentence"].tolist(), "und_Latn")
print("Back-translating Rinconada → Tagalog...")
back_translated_sentences = batch_translate(back_model, back_tokenizer, rinconada_sentences, "tgl_Latn")

print("Computing semantic similarity...")
sentences = df["sentence"].tolist()
similarities = []
chunk_size = 5000 

for i in tqdm(range(0, len(sentences), chunk_size), desc="Embedding & similarity in chunks"):
    orig_chunk = sentences[i:i+chunk_size]
    back_chunk = back_translated_sentences[i:i+chunk_size]

    emb_orig = embedder.encode(orig_chunk, convert_to_tensor=True, batch_size=32)
    emb_back = embedder.encode(back_chunk, convert_to_tensor=True, batch_size=32)

    emb_orig = F.normalize(emb_orig, p=2, dim=1)
    emb_back = F.normalize(emb_back, p=2, dim=1)
    sims = F.cosine_similarity(emb_orig, emb_back).cpu().tolist()

    similarities.extend(sims)

filtered_rinconada = [rin if sim >= 0.70 else "" for rin, sim in zip(rinconada_sentences, similarities)]

df["rinconada"] = filtered_rinconada
df["similarity"] = similarities
output_path = "/ThesisMT/Datasets/generated_synthetic_data/pivot_translation/NLLB200/generated_synthetic_data_pivot_NLLB200.csv"
df.to_csv(output_path, index=False)

print(f"Synthetic Tagalog-Rinconada dataset completed. Saved to {output_path}")
