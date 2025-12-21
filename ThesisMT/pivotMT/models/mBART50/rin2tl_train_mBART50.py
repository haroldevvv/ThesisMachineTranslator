import os
import torch
import pandas as pd
import numpy as np
import evaluate
import nltk
from datetime import datetime
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback
)

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

nltk.download("wordnet")
nltk.download("omw-1.4")

file_path = "/ThesisMT/Datasets/generated_synthetic_data/pivot_translation/mBART50/tl_3.5k_mBART50.csv"
model_name = "facebook/mbart-large-50-many-to-many-mmt"

torch.cuda.empty_cache()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "garbage_collection_threshold:0.6,max_split_size_mb:128"

SRC_LANG = "rin_XX"  # Rinconada (custom)
TGT_LANG = "tl_XX"   # Tagalog (custom)

tokenizer = AutoTokenizer.from_pretrained(model_name)

if TGT_LANG not in tokenizer.get_vocab():
    tokenizer.add_special_tokens({"additional_special_tokens": [TGT_LANG]})
    print(f"Added custom target language token: {TGT_LANG}")

tokenizer.src_lang = SRC_LANG
tokenizer.tgt_lang = TGT_LANG

model = AutoModelForSeq2SeqLM.from_pretrained(
    model_name,
    #device_map="auto",
    offload_folder="/tmp/mbart50_offload",
    #torch_dtype=torch.float16
)
model.config.use_cache = False

model.resize_token_embeddings(len(tokenizer))

try:
    if hasattr(tokenizer, "lang_code_to_id") and TGT_LANG in tokenizer.lang_code_to_id:
        model.config.forced_bos_token_id = tokenizer.lang_code_to_id[TGT_LANG]
        print(f"Set model.config.forced_bos_token_id to {model.config.forced_bos_token_id}")
except Exception as e:
    print("Could not set forced_bos_token_id:", e)

def prepare_dataset(file_path, tokenizer):
    df = pd.read_csv(file_path)
    assert "tl" in df.columns and "rin" in df.columns, "CSV must have 'tl' and 'rin' columns!"
    print(f"Loaded {len(df)} parallel sentence pairs")

    dataset = Dataset.from_pandas(df)

    def preprocess(example):
        # Rinconada → input (source)
        model_inputs = tokenizer(
            example["rin"],
            padding="max_length",
            truncation=True,
            max_length=128
        )
        # Tagalog → target (labels)
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                example["tl"],
                padding="max_length",
                truncation=True,
                max_length=128
            )["input_ids"]

        labels = [(l if l != tokenizer.pad_token_id else -100) for l in labels]
        model_inputs["labels"] = labels
        return model_inputs

    tokenized_dataset = dataset.map(
        preprocess,
        batched=True,
        remove_columns=["tl", "rin"],
        num_proc=min(4, os.cpu_count() or 1)
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

custom_data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    padding=True,
    return_tensors="pt",
    label_pad_token_id=-100
)

class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        labels = labels.to(device)
        outputs = model(
            input_ids=inputs.get("input_ids", None),
            attention_mask=inputs.get("attention_mask", None),
            labels=labels
        )
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss

def main():
    split_dataset = prepare_dataset(file_path, tokenizer)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]

    experiment_name = "rin_tl_3.5k_20epochs_mbart50"
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
        save_total_limit=4,
        learning_rate=3e-5,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=20,
        weight_decay=0.01,
        label_smoothing_factor=0.1,
        predict_with_generate=True,
        generation_max_length=128,
        generation_num_beams=5,
        logging_strategy="steps",
        logging_steps=50,
        fp16=True,
        gradient_checkpointing=True,
        optim="adafactor",
        dataloader_num_workers=2,
        save_safetensors=True,
        load_best_model_at_end=True,
        metric_for_best_model="bleu",
        greater_is_better=True,
        report_to=["tensorboard"],
        logging_dir=log_dir,
        run_name=experiment_name,
    )

    trainer = CustomSeq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=custom_data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=10)]
    )

    trainer.train()
    trainer.save_model(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)

    print(f"Training completed! Final model saved at {final_model_dir}")
    print(f"View logs with: tensorboard --logdir {log_dir}")

if __name__ == "__main__":
    main()
