import argparse 
from models import * 
from utils import *
import tqdm
import os 

def add_training_args(parser):
    parser.add_argument("--model-name",type=str,help="choose a model.")
    parser.add_argument("--log-file",type=str,default=None,help="path to save the log")
    parser.add_argument("--few-shot",action="store_true",help="specify if few shot prompt is needed.")
    parser.add_argument("--lang-pair",type=str,help="indicating the language pair, the first one is the source language and the second one is the target language.")
    parser.add_argument("--prefix",action="store_true",help="if use prefix or not")
    parser.add_argument("--quantization",type=str, choices=["None", "4-bits", "8-bits"],default="None",help="quantization option for Llama-2 models")
    parser.add_argument("--hf-token-file",type=str,default=None,help="path to Hugging Face token file")

def get_args():
    parser = argparse.ArgumentParser()
    add_training_args(parser)
    args = parser.parse_args()
    return args 


def main(args): 
    init_logging(args) #all stout will be logged to the log file
    logging.info("start experiment...")
    src_id,tgt_id = args.lang_pair.split("-")[0],args.lang_pair.split("-")[1]
    src_lang,tgt_lang = id2lang[src_id],id2lang[tgt_id] 

    logging.info(f"language pair: {src_lang}-{tgt_lang}")
    
    #set Hugging Face options
    quantization = args.quantization if args.quantization != "None" else None
    hf_token = None
    if args.hf_token_file is not None:
        with open(args.hf_token_file) as hf_token_fs:
            hf_token = hf_token_fs.read().strip()

    #load model and dataset 
    model = get_model(args.model_name,src_lang,tgt_lang,few_shot=True if args.few_shot else False,use_prefix=True if args.prefix else False,
            quantization=quantization, hf_token=hf_token)
    logging.info(f"model parameters: {model.num_params}")
    src_dir = f"truthfullqa/ref_{src_id}.txt"
    src_text = open(src_dir,"r").readlines()
    #construct output file 
    extra_suffix=""
    if args.model_name.startswith("Llama"):
        extra_suffix += "-quant-"+args.quantization
        extra_suffix += "--few-shot" if args.few_shot else ""
    if args.prefix:
        translation_output_dir = f"truthfullqa/prefix/{src_id}_{tgt_id}_output/" + args.model_name + extra_suffix + ".txt"
    else:
        translation_output_dir = f"truthfullqa/noprefix/{src_id}_{tgt_id}_output/" + args.model_name + extra_suffix + ".txt"
    os.makedirs(os.path.dirname(translation_output_dir),exist_ok=True)
    f = open(translation_output_dir,"a")
    for sent in tqdm.tqdm(src_text):
        sent = sent.strip("\n")
        output = model(sent) 
        f.write(output+"\n")
    f.close()
    ref_dir = f"truthfullqa/ref_{tgt_id}.txt" #reference file which contains the golden label. 
    stats = eval(ref_dir,translation_output_dir)
    logging.info(stats)
    logging.info("="*20)
if __name__ == "__main__":
    args = get_args()
    main(args)
