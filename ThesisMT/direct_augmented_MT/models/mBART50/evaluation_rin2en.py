import os
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

FILE_PATH = "/ThesisMT/Datasets/augmented_data/augmented_data_for_direct_translation/rin_en_8.5k_augmented_mBART50"
MODEL_DIR = "/ThesisMT/direct_augmented_MT/models/mBART50/runs/rin_en_3.5k_15epochs_mbart50_run-20251006-115936/final_model"

use_cpu = False   

TGT_LANG = "en_XX"

def prepare_dataset(file_path, tokenizer):
    df = pd.read_csv(file_path)
    assert "rin" in df.columns and "eng" in df.columns, "CSV must have 'rin' and 'eng' columns!"
    print(f"Loaded {len(df)} parallel sentence pairs from {file_path}")

    dataset = Dataset.from_pandas(df)

    def preprocess(example):
        # Rinconada → input
        model_inputs = tokenizer(
            example["rin"],
            truncation=True,
            max_length=128,
            padding="max_length"
        )
        # English → target
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                example["eng"],
                truncation=True,
                max_length=128,
                padding="max_length"
            )["input_ids"]

        labels = [(l if l != tokenizer.pad_token_id and l < tokenizer.vocab_size else -100) for l in labels]
        model_inputs["labels"] = labels
        return model_inputs

    tokenized_dataset = dataset.map(
        preprocess,
        batched=False,
        remove_columns=["rin", "eng"],
        num_proc=1
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

def evaluate_model(model_dir, eval_dataset, model_name="mBART50 Rin→Eng", tgt_lang="en_XX", use_cpu=True):
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)  # safer
    tokenizer.tgt_lang = tgt_lang

    if use_cpu:
        print("Loading model on CPU...")
        model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
        model.to("cpu")
    else:
        print("Loading model on GPU...")
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_dir,
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )

    if hasattr(tokenizer, "lang_code_to_id"):
        model.config.forced_bos_token_id = tokenizer.lang_code_to_id[tgt_lang]

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, label_pad_token_id=-100)

    training_args = Seq2SeqTrainingArguments(
        output_dir="./tmp_eval",
        per_device_eval_batch_size=4 if use_cpu else 8,  
        predict_with_generate=True,
        generation_max_length=64,
        report_to="none",
        fp16=not use_cpu,  
        dataloader_num_workers=0 if use_cpu else 2,
        save_strategy="no",
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

    results = trainer.evaluate()
    print(f"\nEvaluation Results for {model_name}:")
    for k, v in results.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
    return results

if __name__ == "__main__":
    print("Preparing dataset...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=False)
    tokenizer.tgt_lang = TGT_LANG

    split_dataset = prepare_dataset(FILE_PATH, tokenizer)
    eval_dataset = split_dataset["test"]

    print(f"Eval dataset size: {len(eval_dataset)}")

    results = evaluate_model(
        MODEL_DIR,
        eval_dataset,
        model_name="mBART50 Rin→Eng checkpoint",
        tgt_lang=TGT_LANG,
        use_cpu=use_cpu
    )
