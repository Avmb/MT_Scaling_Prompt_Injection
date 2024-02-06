#!/bin/sh

for tgt_lang in de fr ro; do
  python main.py  --model-name "flan-t5-xxl" --log-file "./results/flan-t5/thruthfullqa_en_${tgt_lang}.txt" --lang-pair "en-${tgt_lang}"
  python main.py  --model-name "flan-t5-xxl" --log-file "./results/flan-t5/prefix/thruthfullqa_en_${tgt_lang}.txt" --lang-pair "en-${tgt_lang}" --prefix
done
