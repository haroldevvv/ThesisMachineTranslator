import torch
import pandas as pd
from transformers import MarianTokenizer, MarianMTModel
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# English → Rinconada
model_dir = "/ThesisMT/directMT/models/MarianMT/runs/en_rin_3.5k_25epochs_run-20250930-235909/final_model"
tokenizer = MarianTokenizer.from_pretrained(model_dir)
model = MarianMTModel.from_pretrained(model_dir).half().to(device)  

# Rinconada → English (back-translation)
back_model_dir = "/ThesisMT/directMT/models/MarianMT/runs/rin_en_3.5k_25epochs_run-20250930-182757/final_model"

back_tokenizer = MarianTokenizer.from_pretrained(back_model_dir)
back_model = MarianMTModel.from_pretrained(back_model_dir).half().to(device)  

dataset_path = "/ThesisMT/Datasets/english_monolingual_data/en_100K.csv"
df = pd.read_csv(dataset_path)
df = df.rename(columns={"en": "sentence"})  

embedder = SentenceTransformer(
    'sentence-transformers/all-MiniLM-L6-v2',
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

def batch_translate_fast(model, tokenizer, sentences, max_length=128, batch_size=128, num_beams=5):
    translations = []
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(sentences), batch_size), desc="Translating in batches"):
            batch = sentences[i:i+batch_size]
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(model.device)
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=True
            )
            translations.extend(tokenizer.batch_decode(outputs, skip_special_tokens=True))
    return translations

print("Translating English → Rinconada...")
rinconada_sentences = batch_translate_fast(model, tokenizer, df['sentence'].tolist(), batch_size=128, num_beams=5)
print("Back-translating Rinconada → English...")
back_translated_sentences = batch_translate_fast(back_model, back_tokenizer, rinconada_sentences, batch_size=128, num_beams=5)
print("Computing semantic similarity...")
with torch.no_grad():
    emb_orig = embedder.encode(
        df['sentence'].tolist(),
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
filtered_rinconada = [
    rin if sim >= 0.70 else "" 
    for rin, sim in zip(rinconada_sentences, similarities)
]

df['rinconada'] = filtered_rinconada
df['similarity'] = similarities
output_path = "/ThesisMT/Datasets/generated_synthetic_data/direct_translation/generated_synthetic_data_direct_MarianMT.csv"
df.to_csv(output_path, index=False, header=True)

print(f"Back-translation with filtering completed. Saved to {output_path}")
