import torch
import pandas as pd
from transformers import MarianTokenizer, MarianMTModel
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Tagalog → Rinconada
model_dir = "/ThesisMT/pivotMT/models/MarianMT/runs/tl_rin_3.5k_25epochs_run-20251003-083523/final_model"
tokenizer = MarianTokenizer.from_pretrained(model_dir)
model = MarianMTModel.from_pretrained(model_dir).half().to(device)  

# Rinconada → Tagalog (back-translation)
back_model_dir = "/ThesisMT/pivotMT/models/MarianMT/runs/rin_tl_3.5k_25epochs_run-20251003-100440/final_model"
back_tokenizer = MarianTokenizer.from_pretrained(back_model_dir)
back_model = MarianMTModel.from_pretrained(back_model_dir).half().to(device)  

dataset_path = "/ThesisMT/Datasets/generated_synthetic_data/pivot_translation/MarianMT/tl_100k_MarianMT.csv"
df = pd.read_csv(dataset_path)
assert "tl" in df.columns, "CSV must have a column named 'tl'"

embedder = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    device="cuda"
)

def batch_translate(model, tokenizer, sentences, max_length=128, batch_size=128, num_beams=5):
    translations = []
    model.eval()
    with torch.no_grad(): 
        for i in tqdm(range(0, len(sentences), batch_size), desc="Translating in batches"):
            batch = sentences[i:i+batch_size]
            inputs = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length
            ).to(model.device)
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=True
            )
            translations.extend(tokenizer.batch_decode(outputs, skip_special_tokens=True))
    return translations

print("Translating Tagalog → Rinconada...")
rinconada_sentences = batch_translate(model, tokenizer, df["tl"].tolist(), batch_size=128, num_beams=5)
print("Back-translating Rinconada → Tagalog...")
back_translated_sentences = batch_translate(back_model, back_tokenizer, rinconada_sentences, batch_size=128, num_beams=5)

print("Computing semantic similarity...")
with torch.no_grad():
    emb_orig = embedder.encode(
        df["tl"].tolist(),
        convert_to_tensor=True,
        batch_size=128,
        normalize_embeddings=True,
        show_progress_bar=True
    )
    emb_back = embedder.encode(
        back_translated_sentences,
        convert_to_tensor=True,
        batch_size=128,
        normalize_embeddings=True,
        show_progress_bar=True
    )

similarities = torch.sum(emb_orig * emb_back, dim=1).cpu().tolist()
filtered_rinconada = [rin if sim >= 0.70 else "" for rin, sim in zip(rinconada_sentences, similarities)]

df["rin"] = filtered_rinconada
df["similarity"] = similarities
output_path = "/ThesisMT/Datasets/generated_synthetic_data/pivot_translation/MarianMT/generated_synthetic_data_pivot_MarianMT.csv"
df.to_csv(output_path, index=False)

print(f"Back-translation with filtering completed. Saved to {output_path}")
