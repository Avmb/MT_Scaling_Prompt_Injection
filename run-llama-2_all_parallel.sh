#!/bin/sh

few_shot_array=("" "--few-shot")
quantization_array=(None 4-bits 8-bits)
chat_array=("" "-chat")
lang_array=(de fr ro ru)

# $SLURM_ARRAY_TASK_ID from 0 to 47
let I=$SLURM_ARRAY_TASK_ID
let few_shot_array_id=$I%2
let I=$I/2
let quantization_id=$I%3
let I=$I/3
let chat_id=$I%2
let I=$I/2
let lang_id=$I%4

few_shot=${few_shot_array[$few_shot_array_id]}
quantization=${quantization_array[$quantization_id]}
chat=${chat_array[$chat_id]}
lang=${lang_array[$lang_id]}

for model_size in 7b 13b 70b; do
	python main.py --model-name "Llama-2-${model_size}${chat}" --log-file "./results/Llama-2${chat}${few_shot}-quant-${quantization}/thruthfullqa_${lang}_en.txt" --lang-pair "${lang}-en" ${few_shot} --quantization ${quantization} --hf-token-file ../hf_token
	python main.py --model-name "Llama-2-${model_size}${chat}" --log-file "./results/Llama-2${chat}${few_shot}-quant-${quantization}/prefix/thruthfullqa_${lang}_en.txt" --lang-pair "${lang}-en" ${few_shot} --quantization ${quantization} --hf-token-file ../hf_token --prefix

        python main.py --model-name "Llama-2-${model_size}${chat}" --log-file "./results/Llama-2${chat}${few_shot}-quant-${quantization}/thruthfullqa_en_${lang}.txt" --lang-pair "en-${lang}" ${few_shot} --quantization ${quantization} --hf-token-file ../hf_token
        python main.py --model-name "Llama-2-${model_size}${chat}" --log-file "./results/Llama-2${chat}${few_shot}-quant-${quantization}/prefix/thruthfullqa_en_${lang}.txt" --lang-pair "en-${lang}" ${few_shot} --quantization ${quantization} --hf-token-file ../hf_token --prefix
done
