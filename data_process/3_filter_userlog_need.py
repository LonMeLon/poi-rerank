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

    for index, row in tqdm(enumerate(reader)):
        if index == 0:
            print(row)
            w_filter.writerow(row)
            
        if index >= 1:
            if (index + 1) % 100000 == 0:
                print(index, count)

            query, geohash, clk_poiid, rec_poi_list_id, sess_time_list, sess_query_list, sess_poilist_list_id, start_poiid = row


            # clk_poiid;start_poiid
            if clk_poiid not in poi_set:
                continue
            if start_poiid not in poi_set:
                continue

            # sess_query_list
            fffff_sess_query_list = json.loads(sess_query_list)
            if query not in fffff_sess_query_list:
                continue
            if query != fffff_sess_query_list[-1]:
                continue
            
            
            fffff_sess_time_list = json.loads(sess_time_list)
            min_gap_time_list = 0
            if len(fffff_sess_time_list) >= 2:
                gap_time_list = [
                    ( int(fffff_sess_time_list[iii+1]) - int(fffff_sess_time_list[iii]) ) / 1000
                    for iii in range(len(fffff_sess_time_list) - 1)
                ]
                min_gap_time_list = min(gap_time_list)
            if min_gap_time_list < 0:
                continue


            # rec_poi_list_id
            fffff_rec_poi_list_id = json.loads(rec_poi_list_id)
            filter_rec_poi_list_id = [rec_poid for rec_poid in fffff_rec_poi_list_id if rec_poid in poi_set]
            if len(filter_rec_poi_list_id) <= 2:
                continue

            # sess query in
            #token_sess_query_list = tokenizer(json.loads(sess_query_list), padding=True)['input_ids']
            fffff_sess_query_list = json.loads(sess_query_list)
            token_sess_query_list = tokenizer(fffff_sess_query_list)['input_ids']
            min_len_token_sess_query_list = min([len(fsq) for fsq in token_sess_query_list])
            if min_len_token_sess_query_list <= 2 + 0:
                continue

            # sess_poilist_list_id
            fffff_sess_poilist_list_id = json.loads(sess_poilist_list_id)
            filter_sess_poilist_list_id = []
            for one_poilist_id in fffff_sess_poilist_list_id:
                filter_sess_poilist_list_id.append([poid__ for poid__ in one_poilist_id if poid__ in poi_set])
            min_len_filter_sess_poilist_list_id = min([len(fspl) for fspl in filter_sess_poilist_list_id])
            if min_len_filter_sess_poilist_list_id <= 2:
                continue
            

            filter_rec_poi_list_id = json.dumps(filter_rec_poi_list_id)
            filter_sess_poilist_list_id = json.dumps(filter_sess_poilist_list_id)

            #assert type(filter_rec_poi_list_id) == type(sess_time_list) == type(sess_query_list) == type(filter_sess_poilist_list_id)

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
    
    parser.add_argument('--path_filter_log', type=str, default="3_userlog_need_filtering.csv", help='')
    
    parser.add_argument('--path_poi', type=str, default="1_poi_need_attr.csv")
    parser.add_argument('--path_click_log', type=str, default="2_userlog_need.csv")
    

    parser.add_argument('--least_rec_poi_num', type=int, default=10)

    

    args = parser.parse_args()


    # function to sample data
    sample_data(args)