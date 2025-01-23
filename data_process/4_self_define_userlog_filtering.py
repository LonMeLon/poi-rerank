# filter chinese query / POI
# filter user log with existing poi
# filter user log with searching the close start position (distance < 100 m)


# coding:utf8

import os, json
import csv
csv.field_size_limit(500 * 1024 * 1024)
import string
import random
import pandas as pd
from datetime import datetime
from argparse import ArgumentParser
from tqdm import tqdm
from transformers import BertTokenizer, BertTokenizerFast

def time_stat(start_time, end_time):
    return ((end_time - start_time).seconds)


def tensorize_text(text, tokenizer):
    return 0

def tensorize_category(cate, dict_cate):
    return dict_cate[cate]

def tensorize_geohash(geohash, dict_geohash):
    return [dict_geohash[char] for char in geohash]

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


def sample_split_data(args):
    
    tokenizer = BertTokenizerFast.from_pretrained(args.vocab_path)

    poi_set = {}
    poi_file = open(args.path_poi, 'r')
    reader_poi = csv.reader(poi_file, delimiter='\01')
    for index, row in tqdm(enumerate(reader_poi)):
        if index >= 1:
            # poi_id
            poid, name, address, geohash = row

            # dict:  3 type
            poi_set[poid] = (name, address, geohash)
    
    print('poi_set', len(poi_set))
    poi_file.close()


    count = [0, 0, 0]

    file = open(args.path_click_log, "r")
    reader = csv.reader(file, delimiter='\01')

    count___ = 0


    f_filter = open(args.path_filter_log, 'w')
    w_filter = csv.writer(f_filter, delimiter='\01')

    tokenizer = BertTokenizerFast.from_pretrained(args.vocab_path)
    # define filtering patterns (userlog_data)
    ### session length: 1-10
    ### time gap: 0.5-5
    ### query reformulation gap: 1-10 (add, delete)
    ### session rec list cut off: 10
    length_bound = (1, 5)
    time_gap_bound = (1, 15) 
    reform_gap = (1, 10)
    rec_cutoff = 10
    assert length_bound[0] < length_bound[1]
    assert time_gap_bound[0] < time_gap_bound[1]
    assert reform_gap[0] < reform_gap[1]

    for index, row in tqdm(enumerate(reader)):
        if index == 0:
            print(row)
            w_filter.writerow(row)
            
        if index >= 1:
            if (index + 1) % 100000 == 0:
                print(index, count)

            query, geohash, clk_poiid, rec_poi_list_id, sess_time_list, sess_query_list, sess_poilist_list_id, start_poiid = row


            # convert str to list format
            rec_poi_list_id_ccccc = json.loads(rec_poi_list_id)
            sess_time_list_ccccc = json.loads(sess_time_list)
            sess_query_list_ccccc = json.loads(sess_query_list)
            sess_poilist_list_id_ccccc = json.loads(sess_poilist_list_id)
            assert len(sess_time_list_ccccc) == len(sess_query_list_ccccc) == len(sess_poilist_list_id_ccccc)
        

            # filter
            ### session length: 
            if not (len(sess_query_list_ccccc) >= length_bound[0] and len(sess_query_list_ccccc) <= length_bound[1]):
                continue

            ### time gap: 
            min_sess_time_gap = sum(time_gap_bound) / 2
            max_sess_time_gap = sum(time_gap_bound) / 2
            if len(sess_query_list_ccccc) >= 2:
                sess_time_gap = [
                    ( int(sess_time_list_ccccc[iii+1]) - int(sess_time_list_ccccc[iii]) ) / 1000
                    for iii in range(len(sess_time_list_ccccc) - 1)
                ]
                min_sess_time_gap = min(sess_time_gap)
                max_sess_time_gap = max(sess_time_gap)
            if not (min_sess_time_gap >= time_gap_bound[0] and max_sess_time_gap <= time_gap_bound[1]):
                continue

            ### query reformulation gap: 
            min_sess_reform_gap = sum(reform_gap) / 2
            max_sess_reform_gap = sum(reform_gap) / 2
            if len(sess_query_list_ccccc) >= 2:
                sess_query_list_tokens = tokenizer(sess_query_list_ccccc)['input_ids']
                sess_query_list_tokens = [qtoks[1:-1] for qtoks in sess_query_list_tokens]
                sess_reform_gap = [
                    calculate_reform(sess_query_list_tokens[iii], sess_query_list_tokens[iii+1])
                    for iii in range(len(sess_query_list_tokens) - 1)
                ]
                min_sess_reform_gap = min(sess_reform_gap)
                max_sess_reform_gap = max(sess_reform_gap)
            if not (min_sess_reform_gap >= reform_gap[0] and max_sess_reform_gap <= reform_gap[1]):
                continue

            ### session rec list cut off: 
            rec_poi_list_id_ccccc = rec_poi_list_id_ccccc[0:rec_cutoff]
            sess_poilist_list_id_ccccc = [rec_list[0:rec_cutoff] for rec_list in sess_poilist_list_id_ccccc]


            filter_rec_poi_list_id = json.dumps(rec_poi_list_id_ccccc)
            filter_sess_poilist_list_id = json.dumps(sess_poilist_list_id_ccccc)

            w_filter.writerow([query, geohash, clk_poiid, filter_rec_poi_list_id, sess_time_list, sess_query_list, filter_sess_poilist_list_id, start_poiid])
            count[1] += 1
                    
    file.close() 
    f_filter.close()
    print(count, count___)

def sample_data(args):
    sample_split_data(args)

if __name__ == "__main__":
    
    # parameter
    parser = ArgumentParser()
    parser.add_argument('--vocab_path', type=str, default="/nfs/volume-93-2/meilang/myprojects/poi_pretrain_dense/download_pretrained_model/bert-base-chinese/", help='')
    
    parser.add_argument('--path_filter_log', type=str, default="4_self_define_userlog_filtering.csv", help='')
    
    parser.add_argument('--path_poi', type=str, default="1_poi_need_attr.csv")
    parser.add_argument('--path_click_log', type=str, default="3_userlog_need_filtering.csv")
    

    parser.add_argument('--least_rec_poi_num', type=int, default=10)

    

    args = parser.parse_args()


    # function to sample data
    sample_data(args)