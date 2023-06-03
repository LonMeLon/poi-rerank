import sys, os
import torch
import torch.nn as nn
import csv, random, json
from argparse import ArgumentParser
from tqdm import tqdm
import numpy as np
import pickle, faiss
from transformers import AdamW, BertTokenizerFast
from contextlib import suppress

import model


def load_data2list(file_path, begin_row, list_cols):
    poi_set = []
    with open(file_path, "r") as file:
        reader = csv.reader(file, delimiter='\01')
        for index, row in tqdm(enumerate(reader)):
            if index >= begin_row:
                list_attr = [row[col] for col in list_cols]
                poi_set.append(list_attr)
              
    return poi_set

def load_geohash(geohash_path):
    dict_geohash = {}
    with open(geohash_path, "r") as file:
        reader = csv.reader(file)
        for row in reader:
            hashcode = row[0]
            dict_geohash.setdefault(hashcode, len(dict_geohash))
    return dict_geohash


def geo_spec_tok(geohash):
    return ''.join(['['+cd+']' for cd in geohash])


def load_checkpoint(checkpoint_load, model):
    checkpoint = torch.load(checkpoint_load, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])


def save_obj(data, path,):
    file = open(path, 'wb')
    pickle.dump(data, file)
    file.close()

def load_obj(path):
    file = open(path, 'rb')
    return pickle.load(file)


class OurDataset(torch.utils.data.Dataset):
    def __init__(self, userlog_data):
        self.userlog_data = userlog_data
        
    def __len__(self):
        return len(self.userlog_data)

    def __getitem__(self, index):
        return self.userlog_data[index]


def main(args):
    # poi 
    list_poi_set = load_data2list(args.poi_path, 1, [i for i in range(4)])
    print("list_poi_set", len(list_poi_set))
    
    # geohash
    dict_geohash = load_geohash(args.geohash_path)

    # data loader
    data_poi_set = OurDataset(list_poi_set)
    dataloader_ = torch.utils.data.DataLoader(
        data_poi_set, 
        batch_size=args.batch_size,
        num_workers=4, 
        pin_memory=True,
    )
    
    
    # tokenizer
    tokenizer = BertTokenizerFast.from_pretrained(args.plm_path)
    print(len(tokenizer))
    geo_special_tokens_dict = ['['+gcd+']' for gcd in dict_geohash]
    tokenizer.add_tokens(geo_special_tokens_dict)
    print(len(tokenizer))

    # model
    itmodel = model.Bert_geo.from_pretrained(args.plm_path, args)
    itmodel.resize_token_embeddings(len(tokenizer))
    load_checkpoint(args.checkpoint_path, itmodel)
    itmodel = itmodel.cuda()
    itmodel.eval()

    # infer
    poi_ids = []
    infer_poi_vecs = []
    
    # batch
    for idx_,  batch_ in tqdm(enumerate(dataloader_)):
        
        batch_poids = []
        batch_poi_context = []
        
        this_batch_size = len(batch_[0])
        for kkkkk in range(this_batch_size):
            poid_ = batch_[0][kkkkk]
            name_ = batch_[1][kkkkk]
            address_ = batch_[2][kkkkk]
            geohash_ = batch_[3][kkkkk]

            batch_poids.append(poid_)
            batch_poi_context.append(
                geo_spec_tok(geohash_) + '[SEP]' + \
                name_ + '[SEP]' + \
                address_
            )

        batch_poi_context = tokenizer(batch_poi_context, padding=True, return_tensors='pt')

        with torch.no_grad():
            poi_ids += batch_poids

            batch_poi_emb = itmodel.cls_(
                batch_poi_context['input_ids'].cuda(), 
                batch_poi_context['attention_mask'].cuda(), 
                batch_poi_context['token_type_ids'].cuda(),
            )
            infer_poi_vecs.append(batch_poi_emb.cpu().numpy())


    # concat
    infer_poi_vecs = np.concatenate(infer_poi_vecs, 0)
    print(len(poi_ids))
    print(infer_poi_vecs.shape)

    
    # faiss index and save
    isExists = os.path.exists(args.index_path)
    if not isExists:
        os.makedirs(args.index_path)

    save_obj(poi_ids, args.index_path + '/poi_ids2vecs.list',)
    save_obj(infer_poi_vecs, args.index_path + '/infer_poi_vecs.numpy', )
    
    print('finished')
    

if __name__ == "__main__":
    
    # parameter
    parser = ArgumentParser()

    # data
    # poi
    parser.add_argument('--poi_path', type=str, default="/nfs/volume-93-2/meilang/myprojects/poi_contextual_rerank/dataset/process/1_poi_need_attr.csv")
    # geohash
    parser.add_argument('--geohash_path', type=str, default='/nfs/volume-93-2/meilang/myprojects/poi_pretrain_dense/ColBERT/1_data_process/contextual_mapping_all_city/1_geohash_code.csv')
    

    # model
    parser.add_argument('--plm_path', type=str, default='/nfs/volume-93-2/meilang/myprojects/poi_pretrain_dense/download_pretrained_model/bert-base-chinese/')

    # checkpoint
    parser.add_argument('--checkpoint_path', type=str, default='checkpoint/temp0.1/model-5.ck')
    # index
    parser.add_argument('--index_path', type=str, default='index/' + parser.parse_args().checkpoint_path)

    
    # run
    parser.add_argument('--batch_size', type=int, default=128)
    
    args = parser.parse_args()
    print(args)

    
    # main
    main(args)