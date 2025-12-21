# 1. Load forward and back-translation models
tokenizer = AutoTokenizer.from_pretrained(tl_to_rin_model,src_lang="tgl_Latn",tgt_lang="und_Latn")
model = AutoModelForSeq2SeqLM.from_pretrained(tl_to_rin_model).half().cuda()
back_tokenizer = AutoTokenizer.from_pretrained(rin_to_tl_model,src_lang="und_Latn",tgt_lang="tgl_Latn")
back_model = AutoModelForSeq2SeqLM.from_pretrained(rin_to_tl_model).half().cuda()

#Translate Tagalog → Rinconada & Rinconada → Tagalog
rinconada_sentences = batch_translate(
    model,
    tokenizer,
    df["sentence"].tolist(),
    tgt_lang="und_Latn",
    batch_size=128,
    num_beams=5
)

back_translated_sentences = batch_translate(
    back_model,
    back_tokenizer,
    rinconada_sentences,
    tgt_lang="tgl_Latn",
    batch_size=128,
    num_beams=5
)

