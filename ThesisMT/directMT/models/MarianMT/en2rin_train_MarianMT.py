import os
import torch
import pandas as pd
import numpy as np
import evaluate
import nltk
from datetime import datetime
from datasets import Dataset
from transformers import (
    MarianTokenizer,
    MarianMTModel,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback
)

nltk.download("wordnet")
nltk.download("omw-1.4")

file_path = "/ThesisMT/Datasets/real_parallel/rin_en_3.5k.csv"
model_name = "Helsinki-NLP/opus-mt-en-tl"

translation_direction = "eng2rin"

tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name, device_map="auto")

def prepare_dataset(file_path, tokenizer):
    df = pd.read_csv(file_path)

    assert "rin" in df.columns and "eng" in df.columns, \
        "CSV must have 'rin' and 'eng' columns!"

    print(f"Loaded {len(df)} parallel sentence pairs")
    print(f"Training direction: {translation_direction}")

    dataset = Dataset.from_pandas(df)

    def preprocess(example):
        if translation_direction == "rin2eng":
            src_text = example["rin"]
            tgt_text = example["eng"]
        else: 
            src_text = example["eng"]
            tgt_text = example["rin"]

        model_inputs = tokenizer(
            src_text,
            max_length=64,
            truncation=True,
            padding="max_length"
        )
        labels = tokenizer(
            text_target=tgt_text,
            max_length=64,
            truncation=True,
            padding="max_length"
        )["input_ids"]

        labels = [(l if l != tokenizer.pad_token_id else -100) for l in labels]
        model_inputs["labels"] = labels
        return model_inputs

    tokenized_dataset = dataset.map(
        preprocess,
        batched=True,
        remove_columns=["rin", "eng"],
        num_proc=min(4, os.cpu_count())
    )

    return tokenized_dataset.train_test_split(test_size=0.1, seed=42)

bleu = evaluate.load("sacrebleu")
meteor = evaluate.load("meteor")
rouge = evaluate.load("rouge")

def compute_metrics(eval_pred):
    preds, labels = eval_pred
    if isinstance(preds, tuple):
        preds = preds[0]

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    bleu_result = bleu.compute(predictions=decoded_preds, references=[[l] for l in decoded_labels])
    meteor_result = meteor.compute(predictions=decoded_preds, references=decoded_labels)
    rouge_result = rouge.compute(predictions=decoded_preds, references=decoded_labels)

    return {
        "bleu": bleu_result["score"],
        "meteor": meteor_result["meteor"],
        "rouge1": rouge_result["rouge1"],
        "rouge2": rouge_result["rouge2"],
        "rougeL": rouge_result["rougeL"],
        "rougeLsum": rouge_result["rougeLsum"],
    }

def main():
    split_dataset = prepare_dataset(file_path, tokenizer)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    experiment_name = f"{translation_direction}_3.5k_25epochs"
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join("./runs", f"{experiment_name}_run-{timestamp}")
    output_dir = os.path.join(run_dir, "checkpoints")
    log_dir = os.path.join(run_dir, "logs")
    final_model_dir = os.path.join(run_dir, "final_model")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        learning_rate=3e-5,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        num_train_epochs=25,
        weight_decay=0.01,
        label_smoothing_factor=0.1,
        predict_with_generate=True,
        generation_max_length=128,
        logging_strategy="steps",
        logging_steps=50,
        fp16=torch.cuda.is_available(),
        optim="adafactor",
        dataloader_num_workers=2,
        save_safetensors=True,
        load_best_model_at_end=True,
        metric_for_best_model="bleu",
        greater_is_better=True,
        report_to=["tensorboard"],
        logging_dir=log_dir,
        run_name=experiment_name
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=10)]
    )

    trainer.train()

    trainer.save_model(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)

    print(f"Training complete! Final model saved at {final_model_dir}")
    print(f"View logs with: tensorboard --logdir {log_dir}")

def get_split_dataset():
    return prepare_dataset(file_path, tokenizer)

if __name__ == "__main__":
    main()
