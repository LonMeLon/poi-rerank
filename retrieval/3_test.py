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

import model


def load_data2dict(file_path, begin_row, id_col, other_list_cols):
    poi_set = {}
    with open(file_path, "r") as file:
        reader = csv.reader(file, delimiter='\01')
        for index, row in tqdm(enumerate(reader)):
            if index >= begin_row:
                poi_id = row[id_col]
                other_list_attr = [row[col] for col in other_list_cols]
                poi_set[poi_id] = other_list_attr
              
    return poi_set

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


def eval_recall_topk(pos_index, top_recall_index, cutoff_list, ):
    assert len(pos_index) == len(top_recall_index)

    # dict metric
    dict_metric_full = {}
    for cutoff in cutoff_list:
        dict_metric_full[cutoff] = []

    # each record
    for i in tqdm(range(len(pos_index))):
        pos_res = pos_index[i]
        top_res = top_recall_index[i]

        # rank rel mark
        rel_mark = []
        for tid in top_res:
            if tid == pos_res:
                rel_mark.append(1)
            else:
                rel_mark.append(0)

        for cutoff in cutoff_list:
            # cutoff rank
            cutoff_rel_mark = rel_mark[:cutoff] 
            # eval
            result_metric = 0
            if np.mean(cutoff_rel_mark) > 0:
                result_metric = 1
            dict_metric_full[cutoff].append(result_metric)

    return dict_metric_full

class OurDataset(torch.utils.data.Dataset):
    def __init__(self, userlog_data):
        self.userlog_data = userlog_data
        
    def __len__(self):
        return len(self.userlog_data)

    def __getitem__(self, index):
        return self.userlog_data[index]


def main(args):
    # prepare
    # poi 
    dict_poi_set = load_data2dict(args.poi_path, 1, 0, [i for i in range(4)])
    print(len(dict_poi_set), )
    # goehash
    dict_geohash = load_geohash(args.geohash_path)
    # query, geohash, clk_poiid, rec_poi_list_id
    test_data = load_data2list(args.test_clicklog_path, 1, [i for i in range(8)])
    print(len(test_data), )
    test_data = OurDataset(test_data)


    # data
    dataloader_ = torch.utils.data.DataLoader(
        test_data, 
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
    load_checkpoint(args.load_checkpoint_path, itmodel)
    itmodel = itmodel.cuda()
    itmodel.eval()


    # infer
    query_pos_ids = []
    query_infer_vecs = []


    # batch
    for idx_,  batch_ in tqdm(enumerate(dataloader_)):
        
        batch_poids = []
        batch_query_context = []
        
        # query, geohash, clk_poiid, filter_rec_poi_list_id, sess_time_list, sess_query_list, filter_sess_poilist_list_id, start_poiid
        this_batch_size = len(batch_[0])
        for kkkkk in range(this_batch_size):
            query_ = batch_[0][kkkkk]
            geohash_ = batch_[1][kkkkk]
            clk_poiid_ = batch_[2][kkkkk]

            batch_poids.append(clk_poiid_)
            batch_query_context.append(
                geo_spec_tok(geohash_) + '[SEP]' + \
                query_
            )

        batch_query_context = tokenizer(batch_query_context, padding=True, return_tensors='pt')

        with torch.no_grad():
            query_pos_ids += batch_poids

            batch_query_emb = itmodel.cls_(
                batch_query_context['input_ids'].cuda(), 
                batch_query_context['attention_mask'].cuda(), 
                batch_query_context['token_type_ids'].cuda(),
            )
            query_infer_vecs.append(batch_query_emb.cpu().numpy())


    # concat
    
    query_infer_vecs = np.concatenate(query_infer_vecs, 0)
    print(len(query_pos_ids))
    print(query_infer_vecs.shape)
    
    
    # faiss index search
    infer_poi_vecs = load_obj(args.load_index_path + '/infer_poi_vecs.numpy')
    faiss_index = faiss.IndexFlatIP(infer_poi_vecs.shape[-1])
    faiss_index.add(infer_poi_vecs)
    
    singlegpu = faiss.StandardGpuResources()
    faiss_index = faiss.index_cpu_to_gpu(singlegpu, 0, faiss_index)
    start = time.time()
    _, topk_res = faiss_index.search(query_infer_vecs, args.topk)
    print(topk_res.shape)
    end = time.time()
    print('search time', (end - start))

    # save result
    poi_ids2vecs = load_obj(args.load_index_path + '/poi_ids2vecs.list')
    topk_recall = []
    for i in range(topk_res.shape[0]):
        one_topk_res = topk_res[i]
        topk_recall.append([poi_ids2vecs[index] for index in one_topk_res])

    
    # evaluation
    dict_metric_full = eval_recall_topk(query_pos_ids, topk_recall, [5, 10, 20, 50, 100])
    for cutoff in dict_metric_full:
        all_res = dict_metric_full[cutoff]
        means = np.mean(all_res)
        print('Recall@{}:{}'.format(cutoff, means))
    

if __name__ == "__main__":
    # parameter
    parser = ArgumentParser()

    # data
    # poi
    parser.add_argument('--poi_path', type=str, default="/nfs/volume-93-2/meilang/myprojects/poi_contextual_rerank/dataset/process/1_poi_need_attr.csv")

    # geohash
    parser.add_argument('--geohash_path', type=str, default='/nfs/volume-93-2/meilang/myprojects/poi_pretrain_dense/ColBERT/1_data_process/contextual_mapping_all_city/1_geohash_code.csv')

    # clicklog
    parser.add_argument('--test_clicklog_path', type=str, default="/nfs/volume-93-2/meilang/myprojects/poi_contextual_rerank/dataset/process/5_sampled_userlog_with_level_test.csv")
    

    # model
    parser.add_argument('--plm_path', type=str, default='/nfs/volume-93-2/meilang/myprojects/poi_pretrain_dense/download_pretrained_model/bert-base-chinese/')


    # checkpoint
    parser.add_argument('--load_checkpoint_path', type=str, default='checkpoint/dynamic_temp0.1/model-4.ck')
    # index
    parser.add_argument('--load_index_path', type=str, default='index/checkpoint/temp0.1/model-5.ck')

    
    # run
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--topk', type=int, default=100)
    
    args = parser.parse_args()
    print(args)

    
    # main
    main(args)