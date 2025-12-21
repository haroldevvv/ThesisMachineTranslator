import os
import torch
import numpy as np
import evaluate
from transformers import (
    MarianMTModel,
    MarianTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)

os.environ["CUDA_VISIBLE_DEVICES"] = ""  
os.environ["TOKENIZERS_PARALLELISM"] = "false"

device = torch.device("cpu")
print(" Running evaluation on CPU only...")

def evaluate_model(model_dir, tokenizer_dir, eval_dataset, model_name=""):

    tokenizer = MarianTokenizer.from_pretrained(tokenizer_dir)
    model = MarianMTModel.from_pretrained(model_dir)

    model.to(device)
    model.eval()

    bleu = evaluate.load("sacrebleu")
    meteor = evaluate.load("meteor")
    rouge = evaluate.load("rouge")

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    training_args = Seq2SeqTrainingArguments(
        output_dir="./tmp_eval",
        per_device_eval_batch_size=2,   
        predict_with_generate=True,
        generation_max_length=128,
        fp16=False,                     
        report_to="none",
        dataloader_num_workers=0,
        no_cuda=True,                   
    )

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

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    with torch.no_grad():
        results = trainer.evaluate()

    print(f"\n Evaluation Results for {model_name}:")
    for k, v in results.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")

    return results

if __name__ == "__main__":
    best_model_dir = "/ThesisMT/direct_augmented_MT/models/MarianMT/runs/rin_en_3.5k_20epochs_run-20251006-010342/final_model"
    tokenizer_dir = best_model_dir

    from train_MarianMT import get_split_dataset
    eval_dataset = get_split_dataset()["test"]

    evaluate_model(best_model_dir, tokenizer_dir, eval_dataset, model_name="Best Single Checkpoint (CPU)")
