#!/usr/bin/env bash   
python main.py  --model-name "ada" \
                --log-file "./results/openai/thruthfullqa_en_fr.txt"  \ 
                --lang-pair "en-fr" \ 
                --few-shot
       
