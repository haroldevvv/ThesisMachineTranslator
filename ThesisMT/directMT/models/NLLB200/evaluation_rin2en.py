import os
import sys
import torch
import numpy as np
import pandas as pd
import evaluate
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)
from transformers.utils import logging as hf_logging

hf_logging.set_verbosity_error()

os.environ["CUDA_VISIBLE_DEVICES"] = "0"   
os.environ["TOKENIZERS_PARALLELISM"] = "false"

FILE_PATH = "/ThesisMT/Datasets/real_parallel/rin_en_3.5k.csv"
MODEL_DIR = "/ThesisMT/directMT/models/NLLB200/runs/rin_en_3.5k_15epochs_nllb_offload_run-20251001-115421/final_model"

use_cpu = False  
SRC_LANG = "rin_Latn"
TGT_LANG = "eng_Latn"

def prepare_dataset(file_path, tokenizer):
    df = pd.read_csv(file_path)
    assert "rin" in df.columns and "eng" in df.columns, 
    print(f" Loaded {len(df)} parallel sentence pairs from {file_path}")

    dataset = Dataset.from_pandas(df)

    def preprocess(example):
        model_inputs = tokenizer(example["rin"], truncation=True, max_length=128)
        labels = tokenizer(example["eng"], truncation=True, max_length=128)["input_ids"]
        labels = [(l if l != tokenizer.pad_token_id else -100) for l in labels]
        model_inputs["labels"] = labels
        return model_inputs

    tokenized_dataset = dataset.map(
        preprocess,
        batched=False,
        remove_columns=["rin", "eng"],
        num_proc=2  
    )

    return tokenized_dataset.train_test_split(test_size=0.1, seed=42)

bleu = evaluate.load("sacrebleu")
meteor = evaluate.load("meteor")
rouge = evaluate.load("rouge")

def compute_metrics(eval_pred, tokenizer_local):
    preds, labels = eval_pred
    if isinstance(preds, tuple):
        preds = preds[0]

    decoded_preds = tokenizer_local.batch_decode(preds, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer_local.pad_token_id)
    decoded_labels = tokenizer_local.batch_decode(labels, skip_special_tokens=True)

    bleu_result = bleu.compute(predictions=decoded_preds, references=[[l] for l in decoded_labels])
    meteor_result = meteor.compute(predictions=decoded_preds, references=decoded_labels)
    rouge_result = rouge.compute(predictions=decoded_preds, references=decoded_labels)

    return {
        "bleu": bleu_result["score"],
        "meteor": meteor_result.get("meteor", 0.0),
        "rougeL": rouge_result.get("rougeL", 0.0),
    }

def evaluate_model(model_dir, eval_dataset, model_name="", src_lang="rin_Latn", tgt_lang="eng_Latn", use_cpu=False):
    print(" Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, src_lang=src_lang, tgt_lang=tgt_lang)

    if use_cpu:
        print(" Using CPU mode (fallback)...")
        model = AutoModelForSeq2SeqLM.from_pretrained(model_dir, device_map=None)
        model.to("cpu")
    else:
        print(" Using GPU with automatic device mapping and half precision...")
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_dir,
            device_map="auto",                     
            torch_dtype=torch.float16,             
            offload_folder="/tmp/nllb_offload_eval",
            low_cpu_mem_usage=True
        )
        model.eval()

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, label_pad_token_id=-100)

    training_args = Seq2SeqTrainingArguments(
        output_dir="./tmp_eval",
        per_device_eval_batch_size=8,  
        predict_with_generate=True,
        generation_max_length=64,
        report_to="none",
        fp16=True,                     
        dataloader_num_workers=2,
        save_strategy="no",
        no_cuda=False,                 
    )

    def _compute_metrics(eval_pred):
        return compute_metrics(eval_pred, tokenizer)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=_compute_metrics,
    )

    print(" Starting evaluation on GPU...")
    results = trainer.evaluate()

    print(f"\n Evaluation Results for {model_name}:")
    for k, v in results.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
    return results

if __name__ == "__main__":
    print("Preparing dataset...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, src_lang=SRC_LANG, tgt_lang=TGT_LANG)
    split_dataset = prepare_dataset(FILE_PATH, tokenizer)
    eval_dataset = split_dataset["test"]

    print(f"Eval dataset size: {len(eval_dataset)}")

    results = evaluate_model(
        MODEL_DIR,
        eval_dataset,
        model_name="NLLB200 Rin→Eng checkpoint",
        src_lang=SRC_LANG,
        tgt_lang=TGT_LANG,
        use_cpu=use_cpu
    )
