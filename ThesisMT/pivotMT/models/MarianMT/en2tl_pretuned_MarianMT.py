import torch
import pandas as pd
from datasets import Dataset
from transformers import MarianTokenizer, MarianMTModel
from pathlib import Path
import time

file_path = "/ThesisMT/Datasets/english_monolingual_data/en_100K.csv"
output_file = "/ThesisMT/Datasets/generated_synthetic_data/pivot_translation/MarianMT/tl_100k_MarianMT.csv"
model_name = "Helsinki-NLP/opus-mt-en-tl"  
batch_size = 128

df = pd.read_csv(file_path)  
en_sentences = df["en"].dropna().astype(str).tolist()
print(f"Loaded {len(en_sentences)} English sentences")

dataset = Dataset.from_dict({"text": en_sentences})

print(f"Available devices: {torch.cuda.device_count()} GPU(s)")
tokenizer = MarianTokenizer.from_pretrained(model_name)

model = MarianMTModel.from_pretrained(
    model_name,
    device_map="auto"  
)
model.eval()

@torch.inference_mode()
def translate_batch(batch):
    inputs = tokenizer(
        batch["text"],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128
    ).to(model.device)

    outputs = model.generate(
        **inputs,
        max_length=128
    )

    batch["translation"] = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return batch

out_path = Path(output_file)
with out_path.open("w", encoding="utf-8") as f_out:
    f_out.write("en\ttl\n")  

    start_time = time.time()
    for i in range(0, len(dataset), batch_size):
        batch = dataset[i: i + batch_size]
        translated = translate_batch(batch)
        for en, tl in zip(translated["text"], translated["translation"]):
            f_out.write(f"{en}\t{tl}\n")

        done = i + len(batch["text"])
        elapsed = time.time() - start_time
        speed = done / elapsed
        eta = (len(dataset) - done) / speed
        print(f"[{done}/{len(dataset)}] {speed:.2f} sents/sec | ETA: {eta/60:.1f} min")

print(f"Translation completed! Saved to {output_file}")
