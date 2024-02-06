from matplotlib import pyplot as pp 
from utils import *
import re
import os
import numpy as np 
from scipy.stats import pearsonr
import numpy as np




def parse_info(path):
    info = extract_stats(path)
    instruct_gpt = {"size":[],"accuracy":[],"bleu":[]}
    gpt3 = {"size":[],"accuracy":[],"bleu":[]}
    text_dvnc_2 = {"size":[],"accuracy":[],"bleu":[]}
    text_dvnc_3 = {"size":[],"accuracy":[],"bleu":[]}
    for item in info:
        model_name = item["model_name"]
        if model_name.endswith("001"):

            instruct_gpt["size"].append(item["size"])
            instruct_gpt["accuracy"].append(item["accuracy"])
            instruct_gpt["bleu"].append(item["bleu"])
        elif model_name.endswith("002"):
            text_dvnc_2["size"].append(item["size"])
            text_dvnc_2["accuracy"].append(item["accuracy"])
            text_dvnc_2["bleu"].append(item["bleu"]) 
        elif model_name.endswith("003"):

            text_dvnc_3["size"].append(item["size"])
            text_dvnc_3["accuracy"].append(item["accuracy"])
            text_dvnc_3["bleu"].append(item["bleu"]) 

        else:
            gpt3["size"].append(item["size"])
            gpt3["accuracy"].append(item["accuracy"])
            gpt3["bleu"].append(item["bleu"])

    return instruct_gpt,gpt3,text_dvnc_2,text_dvnc_3


def parse_info_t5(info:dict):
    t5 = {"size":[],"accuracy":[],"bleu":[]}
    for item in info:
        model_name = item["model_name"]
        t5["size"].append(item["size"])
        t5["accuracy"].append(item["accuracy"])
        t5["bleu"].append(item["bleu"])
    return t5


  
def plot_openai(metric:str,src_id,tgt_id):
    log = f"results/openai/prefix/thruthfullqa_{src_id}_{tgt_id}.txt"
    stats = extract_stats(log)
    insgpt,gpt3,text_dvnc_2,text_dvnc_3=parse_info(stats)
 
# print(insgpt)   
def plot(metric:str,src_id,tgt_id):
    pp.xticks([10**i for i in range(8,12)])
    pp.xscale("log")

    pp.plot("size",metric,data=insgpt,marker="+",color="r",label="instruct gpt") ##
    pp.plot("size",metric,data=gpt3,marker="o",color="b",label="gpt 3") ##
    pp.plot("size",metric,data=text_dvnc_2,marker="x",color="y",label="text-davinci-002") ##
    pp.plot("size",metric,data=text_dvnc_3,marker="v",color="k",label="text-davinci-003") ##

    pp.legend()
    pp.xlabel("model size")
    pp.ylabel(metric) ###
    pp.title(f"{id2lang[src_id]}-{id2lang[tgt_id]}:plot of model size vs {metric}") #modify figure path!

    pp.savefig(f"figures/openai/prefix/{metric}_{src_id}_{tgt_id}.jpg")  ###

def plot_t5(metric):
    log_dir = f"results/t5/prefix/truthfullqa_en_"
    de = parse_info_t5(extract_stats(log_dir + "de.txt"))
    fr = parse_info_t5(extract_stats(log_dir + "fr.txt"))
    ro = parse_info_t5(extract_stats(log_dir + "ro.txt"))


    pp.xticks([10**i for i in range(7,11)])
    pp.xscale("log")
    pp.plot("size",metric,data=de,marker="+",color="r",label="German")
    pp.plot("size",metric,data=fr,marker="o",color="b",label="French")
    pp.plot("size",metric,data=ro,marker="x",color="y",label="Romanian")
    pp.legend()
    pp.xlabel("model size")
    pp.ylabel(metric)
    pp.title(f"plot of model size vs {metric}")
    save_dir = "figures/t5/prefix"
    os.makedirs(save_dir,exist_ok=True)
    pp.savefig(f"{save_dir}/{metric}.jpg")
def pearson(info:dict):
    acc = np.array(info["accuracy"]) 
    size = np.array(info["size"]) / 1000000 
    if len(acc)>=2:
        return pearsonr(size,acc)
if __name__ == "__main__":
    log_path = "results/openai/prefix/thruthfullqa_en_ru.txt"
    instruct_gpt,gpt3,text_dvnc_2,text_dvnc_3 = parse_info(log_path)
    
