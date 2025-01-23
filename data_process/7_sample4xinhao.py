import sys, os
import torch
import torch.nn as nn
import csv, random, json
from argparse import ArgumentParser
from tqdm import tqdm
import numpy as np
import pickle, faiss, time
from transformers import AdamW, BertTokenizerFast
from contextlib import suppress

def main(args):
    data_ = []
    with open(args.path_data, "r") as file:
        reader = csv.reader(file, delimiter='\01')
        for index, row in tqdm(enumerate(reader)):
            data_.append(row)

    print(len(data_))
    

    with open(args.path_new, 'w') as file_new:
        write_new = csv.writer(file_new, delimiter='\01')
        
        for index, row in tqdm(enumerate(data_)):
            if index == 0:
                write_new.writerow(row)
            
            if index >= 1:
                prob = random.random()
                if prob <= 0.1:
                    write_new.writerow(row)
    
            
    

if __name__ == "__main__":
    # parameter
    parser = ArgumentParser()
    parser.add_argument('--path_data', type=str, default="/nfs/volume-93-2/meilang/myprojects/poi_contextual_rerank/dataset/process/5_sampled_userlog_with_level_test_with_candidate.csv")

    parser.add_argument('--path_new', type=str, default="/nfs/volume-93-2/meilang/myprojects/poi_contextual_rerank/dataset/process/5_sampled_userlog_with_level_test_with_candidate_sample4xinhao.csv")


    # run
    parser.add_argument('--sample_prob', type=float, default=0.1)

    args = parser.parse_args()
    print(args)

    
    # main
    main(args)