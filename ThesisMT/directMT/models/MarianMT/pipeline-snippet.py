import torch
import pandas as pd
from transformers import MarianTokenizer, MarianMTModel
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# 1. Load dataset
df = pd.read_csv("english_monolingual_corpus.csv")
english_data = df["english"].dropna().tolist()

# 2. Load forward and back-translation models
tokenizer_en_rin = MarianTokenizer.from_pretrained("en_to_rin_model")
model_en_rin = MarianMTModel.from_pretrained("en_to_rin_model").half().cuda()

tokenizer_rin_en = MarianTokenizer.from_pretrained("rin_to_en_model")
model_rin_en = MarianMTModel.from_pretrained("rin_to_en_model").half().cuda()

# 3. Sentence embedding model for semantic filtering
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cuda")

def batch_translate(model, tokenizer, sentences, batch_size=128):
    outputs = []
    for i in tqdm(range(0, len(sentences), batch_size)):
        batch = sentences[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to("cuda")
        translated = model.generate(**inputs, num_beams=5, max_length=128)
        outputs.extend(tokenizer.batch_decode(translated, skip_special_tokens=True))
    return outputs

# 4. English to Rinconada & Rinconada to English Translation 
rinconada_sents = batch_translate(model_en_rin, tokenizer_en_rin, english_data)
backtranslations = batch_translate(model_rin_en, tokenizer_rin_en, rinconada_sents)

# 5. Semantic similarity using embeddings
emb_orig = embedder.encode(english_data, convert_to_tensor=True, normalize_embeddings=True)
emb_back = embedder.encode(backtranslations, convert_to_tensor=True, normalize_embeddings=True)
similarities = torch.sum(emb_orig * emb_back, dim=1).cpu().tolist()

# 6. Filtering
filtered = [rin if sim >= 0.70 else "" for rin, sim in zip(rinconada_sents, similarities)]

# 7. Saving high-quality synthetic data
df = pd.DataFrame({"english": english_data,"synthetic_rinconada": filtered,"similarity": similarities})
df.to_csv("synthetic_rinconada.csv", index=False)
