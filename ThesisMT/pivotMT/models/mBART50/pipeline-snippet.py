# 1. Load forward and back-translation models
tokenizer = AutoTokenizer.from_pretrained(tl_rin_model_dir)
model = AutoModelForSeq2SeqLM.from_pretrained(tl_rin_model_dir).half().cuda()

back_tokenizer = AutoTokenizer.from_pretrained(rin_tl_model_dir)
back_model = AutoModelForSeq2SeqLM.from_pretrained(rin_tl_model_dir).half().cuda()

# 2. Translate Tagalog → Rinconada & Rinconada → Tagalog
if "rin_XX" not in back_tokenizer.lang_code_to_id:
    back_tokenizer.lang_code_to_id["rin_XX"] = len(back_tokenizer.lang_code_to_id)
    back_tokenizer.id_to_lang_code = {v: k for k, v in back_tokenizer.lang_code_to_id.items()}

back_tokenizer.src_lang = "rin_XX"
back_tokenizer.tgt_lang = "tl_XX"

rinconada_sentences = batch_translate(
    model,
    tokenizer,
    df["sentence"].tolist(),  
    batch_size=128,
    num_beams=5
)
back_translated_sentences = batch_translate(
    back_model,
    back_tokenizer,
    rinconada_sentences,
    batch_size=128,
    num_beams=5,
    tgt_lang="tl_XX"          
)
