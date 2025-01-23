# filter chinese query / POI
# filter user log with existing poi
# filter user log with searching the close start position (distance < 100 m)


# coding:utf8

import os, json, time, pickle
import csv
csv.field_size_limit(500 * 1024 * 1024)
import string
import random
import pandas as pd
from datetime import datetime
from argparse import ArgumentParser
from tqdm import tqdm
from transformers import BertTokenizer, BertTokenizerFast
import matplotlib as mpl

def save_obj(data, path):
    file = open(path, 'wb')
    pickle.dump(data, file)
    file.close()

def load_obj(path):
    file = open(path, 'rb')
    return pickle.load(file)

def calculate_reform(toks_one, toks_two):
    # count
    dict_toks = {}
    for tok in toks_one:
        dict_toks.setdefault(tok, [0, 0])
        dict_toks[tok][0] += 1
    for tok in toks_two:
        dict_toks.setdefault(tok, [0, 0])
        dict_toks[tok][1] += 1
    # compare
    reform_effort = 0
    for tok in dict_toks:
        reform_effort += abs(dict_toks[tok][1] - dict_toks[tok][0])
    
    return reform_effort

def stat_(args):
    
    tokenizer = BertTokenizerFast.from_pretrained(args.vocab_path)

    poi_set = {}
    with open(args.path_poi, 'r') as poi_file:
        reader_poi = csv.reader(poi_file, delimiter='\01')
        for index, row in tqdm(enumerate(reader_poi)):
            if index >= 1:
                # poi_id
                poid, name, address, geohash = row

                # dict:  3 type
                poi_set[poid] = (name, address, geohash)
        
        print('poi_set', len(poi_set))

    gap_time_list_all = []
    gap_query_len_list_all = []
    
    total_count_ = 0
    sess_len_count_ = {}


    sess_unclick_rank_all = []

    dict_clicked_query_freq = {}
    dict_sess_query_candidate_poi = {}

    with open(args.path_filter_log, mode='r') as file:
        reader = csv.reader(file, delimiter='\01')
        for index, row in tqdm(enumerate(reader)):
            if index == 0:
                print(row)
            if index >= 1:
                #rec_poi_list_id = json.loads(rec_poi_list_id)
                #filter_rec_poi_list_id = json.dumps(filter_rec_poi_list_id)
                query, geohash, clk_poiid, rec_poi_list_id, sess_time_list, sess_query_list, sess_poilist_list_id, start_poiid = row

                #rec_poi_list_id = json.loads(rec_poi_list_id)
                #sess_time_list = json.loads(sess_time_list)
                sess_query_list = json.loads(sess_query_list)
                #sess_poilist_list_id = json.loads(sess_poilist_list_id)

                #if query != sess_query_list[-1]:
                #    print(query, sess_query_list)
                #    print(json.loads(sess_time_list))

                #assert query == sess_query_list[-1]

                #dict_clicked_query_freq[query] = 1 + dict_clicked_query_freq.get(query, 0)
                '''
                for sess_i, sessqqq in enumerate(sess_query_list):
                    dict_sess_query_candidate_poi.setdefault(sessqqq, {})
                    # click tag
                    dict_sess_query_candidate_poi[sessqqq][clk_poiid] = 1 + dict_sess_query_candidate_poi[sessqqq].get(clk_poiid, 0)
                '''



                #sess_query_list_tokens = tokenizer(sess_query_list)['input_ids']
                #sess_query_list_tokens = [qtoks[1:-1] for qtoks in sess_query_list_tokens]
                '''
                print(index)
                print(query, sess_query_list)
                print(sess_query_list_tokens)
                print(rec_poi_list_id)
                print(sess_poilist_list_id[-1])

                sess_reform_gap = [
                    calculate_reform(sess_query_list_tokens[iii], sess_query_list_tokens[iii+1])
                    for iii in range(len(sess_query_list_tokens) - 1)
                ]
                print(sess_reform_gap)

                if index >= 50:
                    break
                '''
                '''
                if len(sess_poilist_list_id) >= 2:
                    for iiiii in range(len(sess_poilist_list_id) - 1):
                        for i, rec_id in enumerate(sess_poilist_list_id[iiiii]):
                            if clk_poiid == rec_id:
                                sess_unclick_rank_all.append(i + 1)
                                break
                '''

                '''
                for i, rec_id in enumerate(rec_poi_list_id):
                    if clk_poiid == rec_id:
                        click_ranks.append(i + 1)
                        break
                '''
                
                length_sess = len(sess_query_list)
                sess_len_count_.setdefault(length_sess, 0)
                sess_len_count_[length_sess] += 1
                total_count_ += 1

                #if length_sess >= 50:
                #    print(sess_query_list)
                
                '''
                if len(sess_time_list) >= 2:
                    count_ += 1
                    gap_query_len_list = [
                        (len(sess_query_list[iii+1])-len(sess_query_list[iii])) 
                        for iii in range(len(sess_query_list) - 1)
                    ]
                    gap_time_list = [
                        ( int(sess_time_list[iii+1]) - int(sess_time_list[iii]) ) / 1000
                        for iii in range(len(sess_time_list) - 1)
                    ]
                    
                    gap_query_len_list_all += gap_query_len_list
                    gap_time_list_all += gap_time_list
                '''
    '''
    print(len(dict_sess_query_candidate_poi))
    save_obj(dict_sess_query_candidate_poi, '4_stat_sess_query_candidate_poi.dict')
    
    candidate_length = {}
    for qqqqq in dict_sess_query_candidate_poi:
        candidate_length.setdefault(len(dict_sess_query_candidate_poi[qqqqq]), 0)
    
    sorted_length = sorted(candidate_length.items(), key=lambda x:x[0], reverse=True)
    print(sorted_length[0:20])
    print(sorted_length[-20:])
    '''
    '''
    print(len(dict_clicked_query_freq))
    save_obj(dict_clicked_query_freq, '4_stat_clicked_query_freq.dict')
    
    sorted_queries = sorted(dict_clicked_query_freq.items(), key=lambda x:x[1], reverse=True)


    f_query = open(args.path_stat_queries, "w")
    w_query = csv.writer(f_query, delimiter='\01')
    for pair in sorted_queries:
        #query, num = pair
        w_query.writerow(pair)
    f_query.close()
    '''
    
    
    
    for key in sess_len_count_:
        sess_len_count_[key] = sess_len_count_[key] / total_count_
    resss = list(sess_len_count_.items())
    print(total_count_)
    resss = sorted(resss, key=lambda x:x[0])
    
    total_length = 0
    for iiiii, pair in enumerate(resss):
        print(pair[0], round(pair[1] * 100, 2))
        total_length += pair[0] * pair[1]
        print(iiiii + 1, total_length / sum([resss[kkk][1] for kkk in range(iiiii + 1)]))
        print(iiiii + 1, total_length)
    
    
    #print(count_, len(gap_query_len_list_all), len(gap_time_list_all))
    #save_obj(gap_query_len_list_all, '4_gap_query_len_list_all')
    #save_obj(gap_time_list_all, '4_gap_time_list_all')
    #print(len(click_ranks))
    #save_obj(click_ranks, '4_click_ranks_all')

    #print(len(sess_unclick_rank_all))
    #save_obj(sess_unclick_rank_all, '4_sess_unclick_rank_all')



if __name__ == "__main__":
    
    # parameter
    parser = ArgumentParser()
    parser.add_argument('--vocab_path', type=str, default="/nfs/volume-93-2/meilang/myprojects/poi_pretrain_dense/download_pretrained_model/bert-base-chinese/", help='')
    parser.add_argument('--path_filter_log', type=str, default="3_userlog_need_filtering.csv", help='')
    parser.add_argument('--path_stat_queries', type=str, default="4_stat_query_freq.csv")
    parser.add_argument('--path_poi', type=str, default="1_poi_need_attr.csv")
    
    args = parser.parse_args()


    # function to sample data
    stat_(args)