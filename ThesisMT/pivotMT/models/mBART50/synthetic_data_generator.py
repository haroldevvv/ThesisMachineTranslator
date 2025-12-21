import os
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, MBart50TokenizerFast
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Tagalog → Rinconada
model_dir = "/ThesisMT/pivotMT/models/mBART50/runs/tl_rin_3.5k_15epochs_mbart50_run-20251003-175907/final_model"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSeq2SeqLM.from_pretrained(model_dir).to(device)

# Rinconada → Tagalog (back-translation)
back_model_dir = "/ThesisMT/pivotMT/models/mBART50/runs/rin_tl_3.5k_20epochs_mbart50_run-20251003-210055/final_model"

try:
    back_tokenizer = AutoTokenizer.from_pretrained(back_model_dir)
except KeyError:
    print(" Custom Rinconada code not found — rebuilding tokenizer manually.")
    back_tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")

if "rin_XX" not in back_tokenizer.lang_code_to_id:
    print("Adding custom language code: rin_XX")
    back_tokenizer.lang_code_to_id["rin_XX"] = len(back_tokenizer.lang_code_to_id)
    back_tokenizer.id_to_lang_code = {v: k for k, v in back_tokenizer.lang_code_to_id.items()}

back_tokenizer.src_lang = "rin_XX"
back_tokenizer.tgt_lang = "tl_XX"
back_model = AutoModelForSeq2SeqLM.from_pretrained(back_model_dir).to(device)

dataset_path = "/ThesisMT/Datasets/generated_synthetic_data/pivot_translation/mBART50/tl_100k.csv"
df = pd.read_csv(dataset_path)

df = df.rename(columns={"tl": "sentence"})
df = df[df["sentence"].notna()]
df["sentence"] = df["sentence"].astype(str).str.strip()

embedder = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    device="cuda:0" if torch.cuda.is_available() else "cpu"
)

def batch_translate(model, tokenizer, sentences, max_length=128, batch_size=32, tgt_lang=None):
    translations = []
    forced_bos = None

    if tgt_lang and hasattr(tokenizer, "lang_code_to_id") and tgt_lang in tokenizer.lang_code_to_id:
        forced_bos = tokenizer.lang_code_to_id[tgt_lang]

    for i in tqdm(range(0, len(sentences), batch_size), desc="Translating in batches"):
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
            max_length=max_length,
            num_beams=5,
            early_stopping=True,
            forced_bos_token_id=forced_bos
        )
        decoded = [tokenizer.decode(t, skip_special_tokens=True) for t in outputs]
        translations.extend(decoded)
    return translations

print("Translating Tagalog → Rinconada...")
rinconada_sentences = batch_translate(model, tokenizer, df["sentence"].tolist(), batch_size=32)
print("Back-translating Rinconada → Tagalog...")
back_translated_sentences = batch_translate(back_model, back_tokenizer, rinconada_sentences, batch_size=32, tgt_lang="tl_XX")

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
output_path = "/ThesisMT/Datasets/generated_synthetic_data/pivot_translation/mBART50/generated_synthetic_pivot_mBART50.csv"
df.to_csv(output_path, index=False, header=True)

print(f" Back-translation with filtering completed. Saved to {output_path}")
