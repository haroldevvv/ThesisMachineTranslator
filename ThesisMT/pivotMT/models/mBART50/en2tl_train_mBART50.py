import pandas as pd
import torch
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from tqdm import tqdm
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

file_path = "/ThesisMT/Datasets/english_monolingual_data/en_100K.csv"
output_file = "/ThesisMT/Datasets/generated_synthetic_data/pivot_translation/mBART50/tl_100k_mBART50_.csv"

model_dir = "/ThesisMT/pivotMT/models/mBART50/runs/en_tl_3.5k_20epochs_mbart50_run-20251006-093323/final_model"

tokenizer = MBart50TokenizerFast.from_pretrained(model_dir, local_files_only=True)

model = MBartForConditionalGeneration.from_pretrained(
    model_dir,
    local_files_only=True,
    torch_dtype=torch.float16,    
    low_cpu_mem_usage=True        
).to(device)
model.eval()

tokenizer.src_lang = "en_XX"
tokenizer.tgt_lang = "tl_XX"

df = pd.read_csv(file_path)

if "en" in df.columns:
    en_sentences = df["en"].dropna().tolist()
else:
    en_sentences = df.iloc[:, 0].dropna().tolist()

print(f"Loaded {len(en_sentences)} English sentences")

@torch.inference_mode()  
def translate_en_to_tl(sentences, max_length=64, batch_size=128, num_beams=2):
    translations = []

    for i in tqdm(range(0, len(sentences), batch_size), desc="Translating", ncols=100):
        batch = sentences[i:i+batch_size]

        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        ).to(device, non_blocking=True)  

        generated = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.lang_code_to_id["tl_XX"],
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=True
        )

        batch_translations = tokenizer.batch_decode(
            generated, skip_special_tokens=True
        )
        translations.extend(batch_translations)

    return translations

print("Translating English → Tagalog using mBART50...")
tl_sentences = translate_en_to_tl(
    en_sentences, max_length=64, batch_size=128, num_beams=2
)

df_out = pd.DataFrame({"en": en_sentences, "tl": tl_sentences})
df_out.to_csv(output_file, index=False, encoding="utf-8")

print(f" Translation completed! Saved {len(df_out)} sentence pairs to {output_file}")
