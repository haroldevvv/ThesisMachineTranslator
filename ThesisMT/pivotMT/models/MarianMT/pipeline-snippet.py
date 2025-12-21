# 1. Load forward and back-translation models
tl_to_rin_model = MarianMTModel.from_pretrained(tl_rin_model).half().cuda()
tl_to_rin_tokenizer = MarianTokenizer.from_pretrained(tl_rin_model)

rin_to_tl_model = MarianMTModel.from_pretrained(rin_tl_model).half().cuda()
rin_to_tl_tokenizer = MarianTokenizer.from_pretrained(rin_tl_model)

#Translate Tagalog → Rinconada & Rinconada → Tagalog
rinconada_sentences = batch_translate(
    tl_to_rin_model,
    tl_to_rin_tokenizer,
    df["tl"].tolist(),
    batch_size=128,
    num_beams=5
)

back_translated_sentences = batch_translate(
    rin_to_tl_model,
    rin_to_tl_tokenizer,
    rinconada_sentences,
    batch_size=128,
    num_beams=5
)
