import os
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from pathlib import Path
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

file_path = "/ThesisMT/Datasets/english_monolingual_data/en_100k.csv"
output_file = "/ThesisMT/Datasets/generated_synthetic_data/pivot_translation/NLLB200/tl_100k_NLLB200.csv"
model_name = "facebook/nllb-200-distilled-600M"
src_lang = "eng_Latn"
tgt_lang = "tgl_Latn"
batch_size = 128  

with open(file_path, "r", encoding="utf-8") as f:
    en_sentences = [line.strip() for line in f if line.strip()]

print(f"Loaded {len(en_sentences)} non-empty English sentences")
dataset = Dataset.from_dict({"text": en_sentences})

print(f"Available devices: {torch.cuda.device_count()} GPU(s)")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
tokenizer.src_lang = src_lang
forced_bos_id = tokenizer.convert_tokens_to_ids(tgt_lang)

model = AutoModelForSeq2SeqLM.from_pretrained(
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
        forced_bos_token_id=forced_bos_id,
        max_length=128
    )

    batch["translation"] = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return batch

out_path = Path(output_file)
with out_path.open("w", encoding="utf-8") as f_out:
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
