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

sys.path.append('/nfs/volume-93-2/meilang/myprojects/poi_contextual_rerank/model/retrieval_for_rerank/')

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


class OurDataset(torch.utils.data.Dataset):
    def __init__(self, userlog_data):
        self.userlog_data = userlog_data
        
    def __len__(self):
        return len(self.userlog_data)

    def __getitem__(self, index):
        return self.userlog_data[index]


def main(args):
    # poi 
    dict_poi_set = load_data2dict(args.poi_path, 1, 0, [i for i in range(4)])
    list_poi_full = list(dict_poi_set.keys())
    print(len(dict_poi_set), len(list_poi_full))
    # goehash d
    dict_geohash = load_geohash(args.geohash_path)
    
    
    ### poi vec
    infer_poi_vecs = load_obj(args.load_index_path + '/infer_poi_vecs.numpy')
    faiss_index = faiss.IndexFlatIP(infer_poi_vecs.shape[-1])
    print('faiss_index', faiss_index, )
    faiss_index.add(infer_poi_vecs)
    singlegpu = faiss.StandardGpuResources()
    faiss_index = faiss.index_cpu_to_gpu(singlegpu, 0, faiss_index)
    ### poid
    poi_ids2vecs = load_obj(args.load_index_path + '/poi_ids2vecs.list')
    print(infer_poi_vecs.shape, len(poi_ids2vecs))
    

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



    title_ = []
    data_ = []
    sess_length_ = []
    sess_query_context_ = []
    with open(args.path_data, "r") as file:
        reader = csv.reader(file, delimiter='\01')
        for index, row in tqdm(enumerate(reader)):
            if index == 0:
                print(row)
                title_ = row

            if index >= 1:
                data_.append(row)
                
                query, geohash, clk_poiid, rec_poi_list_id, sess_time_list, sess_query_list, sess_poilist_list_id, start_poiid = row


                reform_sess_query_list = json.loads(sess_query_list)
                assert query == reform_sess_query_list[-1]

                
                sess_length_.append(len(reform_sess_query_list))

                for qqqqq in reform_sess_query_list:
                    sess_query_context_.append(
                        geo_spec_tok(geohash) + '[SEP]' + \
                        qqqqq
                    )
                
                if (index + 1) % 100000 == 0:
                    print(index + 1)

    print(len(data_), len(sess_length_), len(sess_query_context_))



    sess_query_context_emb = []

    if len(sess_query_context_) / args.batch_size > int(len(sess_query_context_) / args.batch_size):
        batch_num = int(len(sess_query_context_) / args.batch_size) + 1
    else:
        batch_num = int(len(sess_query_context_) / args.batch_size)
    print('batch_num', batch_num)

    for bid in tqdm(range(batch_num)):
        start_ = bid * args.batch_size
        end_ = (bid + 1) * args.batch_size

        input_ = sess_query_context_[start_:end_]
        input_ = tokenizer(input_, padding=True, return_tensors='pt')
        with torch.no_grad():
            input_emb_ = itmodel.cls_(
                input_['input_ids'].cuda(), 
                input_['attention_mask'].cuda(), 
                input_['token_type_ids'].cuda(),
            )
            sess_query_context_emb.append(input_emb_.cpu().numpy())


    sess_query_context_emb = np.concatenate(sess_query_context_emb, 0)
    print('sess_query_context_emb', sess_query_context_emb.shape)


    start = time.time()
    _, session_topk_res = faiss_index.search(sess_query_context_emb, args.topk)
    end = time.time()
    print('search', end - start)

    position_ = [0] + list(np.cumsum(sess_length_))
    assert position_[-1] == sum(sess_length_) == len(sess_query_context_)

    
    
    with open(args.path_new, 'w') as file_new:
        write_new = csv.writer(file_new, delimiter='\01')
        
        print(title_)
        write_new.writerow(title_ + ['sess_candidate_poilist'])
        
        for index, row in tqdm(enumerate(data_)):
            this_session_topk_res = session_topk_res[position_[index] : position_[index+1]]
            assert this_session_topk_res.shape[0] == sess_length_[index]
            
            this_sess_candidate_poilist = []
            for iiiii in range(this_session_topk_res.shape[0]):
                topk_candidate = list(this_session_topk_res[iiiii])
                topk_candidate = [poi_ids2vecs[cand] for cand in topk_candidate]
                assert len(topk_candidate) == args.topk
                
                this_sess_candidate_poilist.append(topk_candidate)

            this_sess_candidate_poilist = json.dumps(this_sess_candidate_poilist)


            write_new.writerow(row + [this_sess_candidate_poilist])
                
    print(index)



if __name__ == "__main__":
    # parameter
    parser = ArgumentParser()
    parser.add_argument('--path_data', type=str, default="5_sampled_userlog_with_level_train.csv")

    parser.add_argument('--path_new', type=str, default="5_sampled_userlog_with_level_train_with_candidate.csv")


    # poi
    parser.add_argument('--poi_path', type=str, default="/nfs/volume-93-2/meilang/myprojects/poi_contextual_rerank/dataset/process/1_poi_need_attr.csv")

    # geohash
    parser.add_argument('--geohash_path', type=str, default='/nfs/volume-93-2/meilang/myprojects/poi_pretrain_dense/ColBERT/1_data_process/contextual_mapping_all_city/1_geohash_code.csv')

    # model
    parser.add_argument('--plm_path', type=str, default='/nfs/volume-93-2/meilang/myprojects/poi_pretrain_dense/download_pretrained_model/bert-base-chinese/')

    # checkpoint
    parser.add_argument('--load_checkpoint_path', type=str, default='/nfs/volume-93-2/meilang/myprojects/poi_contextual_rerank/model/retrieval_for_rerank/checkpoint/dynamic_temp0.1/model-4.ck')
    # index
    parser.add_argument('--load_index_path', type=str, default='/nfs/volume-93-2/meilang/myprojects/poi_contextual_rerank/model/retrieval_for_rerank/index/checkpoint/temp0.1/model-5.ck')

    # run
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--topk', type=int, default=100)
    


    args = parser.parse_args()
    print(args)

    
    # main
    main(args)