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
    

def sample_neg_not_overlap(sample_num, all_candidate, had_items_set):
    res = []
    while len(res) < sample_num:
        sub_samples = random.sample(all_candidate, sample_num)
        for rand_id in sub_samples:
            if len(res) == sample_num:
                break
            if rand_id not in had_items_set:
                res.append(rand_id) 
    
    assert len(res) == sample_num
    return res


def load_checkpoint(checkpoint_load, model):
    print('loading checkpoint', checkpoint_load)
    checkpoint = torch.load(checkpoint_load, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])

def save_checkpoint(checkpoint_path, model, epoch, ):
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    
    checkpoint = {}
    checkpoint['model_state_dict'] = model.state_dict()
    torch.save(checkpoint, checkpoint_path + '/model-' + str(epoch) + '.ck')

def save_obj(data, path,):
    file = open(path, 'wb')
    pickle.dump(data, file)
    file.close()

def load_obj(path):
    file = open(path, 'rb')
    return pickle.load(file)


# evaluation
def evaluate_top(pos_index, top_retrieval_index, cutoff_list, metric_list):
    assert len(top_retrieval_index) == len(pos_index)

    # dict metric
    dict_metric_full = {}
    for metric in metric_list:
        dict_metric_full[metric] = {}
        for cutoff in cutoff_list:
            dict_metric_full[metric][cutoff] = []

    # each record
    for i in tqdm(range(len(top_retrieval_index))):
        top_res = top_retrieval_index[i]
        pos_res = pos_index[i]
        # rank rel mark
        rel_mark = []
        for tid in top_res:
            if tid == pos_res:
                rel_mark.append(1)
            else:
                rel_mark.append(0)

        # metric / cutoff
        for metric in metric_list:
            for cutoff in cutoff_list:
                # cutoff rank
                cutoff_rel_mark = rel_mark[:cutoff] # 
                # eval
                result_metric = 0
                if metric == 'mrr':
                    for index, rel in enumerate(cutoff_rel_mark):
                        if rel == 1:
                            result_metric = 1 / (1 + index)
                            break
                if metric == 'ndcg':
                    dcg = np.sum(cutoff_rel_mark / np.log2(1 + np.arange(1, len(cutoff_rel_mark) + 1)))
                    sorted_cutoff_rel_mask = sorted(cutoff_rel_mark, reverse=True)
                    norm = np.sum(sorted_cutoff_rel_mask / np.log2(1 + np.arange(1, len(sorted_cutoff_rel_mask) + 1)))
                    if norm > 0:
                        result_metric = dcg / norm

                if metric == 'sr':
                    if np.mean(cutoff_rel_mark) > 0:
                        result_metric = 1

                dict_metric_full[metric][cutoff].append(result_metric)


    return dict_metric_full



class OurDataset(torch.utils.data.Dataset):
    def __init__(self, userlog_data):
        self.userlog_data = userlog_data

        for iii in tqdm(range(len(self.userlog_data))):
            # query, geohash, clk_poiid, filter_rec_poi_list_id, sess_time_list, sess_query_list, filter_sess_poilist_list_id, start_poiid, sess_candidate_poilist
            #self.userlog_data[iii][5] = json.loads(self.userlog_data[iii][5])
            self.userlog_data[iii][8] = json.loads(self.userlog_data[iii][8])
        
    def __len__(self):
        return len(self.userlog_data)

    def __getitem__(self, index):
        return self.userlog_data[index]



def main(args):
    # prepare
    # poi 
    dict_poi_set = load_data2dict(args.poi_path, 1, 0, [i for i in range(4)])
    list_poi_full = list(dict_poi_set.keys())
    print(len(dict_poi_set), len(list_poi_full))
    # goehash d
    dict_geohash = load_geohash(args.geohash_path)
    # query, geohash, clk_poiid, filter_rec_poi_list_id, sess_time_list, sess_query_list, filter_sess_poilist_list_id, start_poiid, sess_candidate_poilist
    test_data = load_data2list(args.test_data_path, 1, [i for i in range(9)])
    print(len(test_data))
    test_data = OurDataset(test_data)
    
    
    # ----- all train data
    dataloader_ = torch.utils.data.DataLoader(
        test_data, 
        batch_size=args.batch_size,
        num_workers=4, 
        pin_memory=True,
        collate_fn=lambda x:x,
    )

    # tokenizer
    tokenizer = BertTokenizerFast.from_pretrained(args.plm_path)
    print(len(tokenizer))
    geo_special_tokens_dict = ['['+gcd+']' for gcd in dict_geohash]
    tokenizer.add_tokens(geo_special_tokens_dict)
    print(len(tokenizer))


    
    # ----- model load
    itmodel = model.Our_actor.from_pretrained(args.plm_path, args)
    itmodel.resize_token_embeddings(len(tokenizer))
    load_checkpoint(args.load_checkpoint_path, itmodel)
    itmodel = itmodel.cuda()
    itmodel.eval()


    
    query_pos_ids = []
    query_rerank_poids = []
    # batch
    for idx_,  batch_ in tqdm(enumerate(dataloader_)):
        batch_pos_poids = []
        batch_candiate_poids = []

        query_poi_context_ = []


        if (idx_ + 1) * args.batch_size > 10000:
            break

        this_batch_size = len(batch_)
        for kkkkk in range(this_batch_size):
            query_ = batch_[kkkkk][0]
            q_geohash = batch_[kkkkk][1]
            clk_poiid = batch_[kkkkk][2]
            clicked_query_candidate_poilist = batch_[kkkkk][8][-1]

            assert len(clicked_query_candidate_poilist) == 100

            clicked_query_candidate_poilist = clicked_query_candidate_poilist[0:args.candidate_num]

            
            batch_pos_poids.append(clk_poiid)
            batch_candiate_poids.append([clk_poiid] + clicked_query_candidate_poilist)

            _, pos_poi_name, pos_poi_address, pos_poi_geohash = dict_poi_set[clk_poiid]
            query_poi_context_.append(
                geo_spec_tok(q_geohash) + '[SEP]' + \
                query_ + '[SEP]' + \
                geo_spec_tok(pos_poi_geohash) + '[SEP]' + \
                pos_poi_name + '[SEP]' + \
                pos_poi_address
            )
            for candi_poid in clicked_query_candidate_poilist:
                _, candi_poi_name, candi_poi_address, candi_poi_geohash = dict_poi_set[candi_poid]
                query_poi_context_.append(
                    geo_spec_tok(q_geohash) + '[SEP]' + \
                    query_ + '[SEP]' + \
                    geo_spec_tok(candi_poi_geohash) + '[SEP]' + \
                    candi_poi_name + '[SEP]' + \
                    candi_poi_address
                )
        
        query_poi_context_ = tokenizer(query_poi_context_, padding=True, return_tensors='pt')
        
        with torch.no_grad():
            scores_ = itmodel.score_query_poi(
                query_poi_context_['input_ids'].cuda(), 
                query_poi_context_['attention_mask'].cuda(), 
                query_poi_context_['token_type_ids'].cuda(),
            ).reshape(-1).reshape(this_batch_size, 1 + args.candidate_num).detach().cpu().numpy()

        
        
        query_pos_ids += batch_pos_poids

        for bbbbb in range(this_batch_size):
            pairs_poi_scores = list(
                zip(
                    batch_candiate_poids[bbbbb], 
                    list(scores_[bbbbb])
                )
            )
            sorted_pairs_poi_scores = sorted(pairs_poi_scores, key=lambda x:x[1], reverse=True)
            sorted_poids = [pair[0] for pair in sorted_pairs_poi_scores]
            
            query_rerank_poids.append(sorted_poids)


    dict_metric_full = evaluate_top(query_pos_ids, query_rerank_poids, [1, 3, 5, 10], ['mrr', 'ndcg', ])
    for metric in dict_metric_full:
        for cutoff in dict_metric_full[metric]:
            all_res = dict_metric_full[metric][cutoff]
            means = np.mean(all_res)
            print('{}@{}:{}'.format(metric, cutoff, means))



if __name__ == "__main__":
    # parameter
    parser = ArgumentParser()
    
    # data
    # poi
    parser.add_argument('--poi_path', type=str, default="/nfs/volume-93-2/meilang/myprojects/poi_contextual_rerank/dataset/process/1_poi_need_attr.csv")
    # clicklog
    parser.add_argument('--test_data_path', type=str, default="/nfs/volume-93-2/meilang/myprojects/poi_contextual_rerank/dataset/process/5_sampled_userlog_with_level_test_with_candidate.csv")
    # geohash
    parser.add_argument('--geohash_path', type=str, default='/nfs/volume-93-2/meilang/myprojects/poi_pretrain_dense/ColBERT/1_data_process/contextual_mapping_all_city/1_geohash_code.csv')
    

    # model
    parser.add_argument('--plm_path', type=str, default='/nfs/volume-93-2/meilang/myprojects/poi_pretrain_dense/download_pretrained_model/bert-base-chinese/')

    # checkpoint
    parser.add_argument('--load_checkpoint_path', type=str, default='checkpoint/major_actor-v0-23999.ck')

    
    # run
    # batch size
    parser.add_argument('--batch_size', type=int, default=4)
    # batch size
    parser.add_argument('--candidate_num', type=int, default=50)
    

    args = parser.parse_args()
    print(args)

    
    # main
    main(args)