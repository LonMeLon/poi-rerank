import random
random.seed(0)
from collections import Counter, defaultdict
import torch
import csv, random, json
from tqdm import tqdm
from torch.utils.data import Dataset
import numpy as np
from transformers import AdamW, BertTokenizerFast

from difflib import get_close_matches,SequenceMatcher
def get_sim_q(q,q_database,q_len_dict,max_num=2):
    q_list = get_close_matches(q, q_database, n=5, cutoff=0.3)
    score = []
    for match in q_list:
        sim = SequenceMatcher(None, q, match).ratio()
        spe = (q_len_dict[match]-q_len_dict[q])/q_len_dict[q]
        score.append(sim+spe)
    score = np.array(score)
    idx =  np.argsort(-score)
    try:
        sim_q = list(np.array(q_list)[idx[:max_num]])
    except:
        sim_q = q_list[:1]
    return sim_q


# 导入geohash
def load_geohash(geohash_path=r'data\1_geohash_code.csv'):
    dict_geohash = {}
    with open(geohash_path, "r") as file:
        reader = csv.reader(file)
        for row in reader:
            hashcode = row[0]
            dict_geohash.setdefault(hashcode, len(dict_geohash))
    return dict_geohash

#导入poi
def load_data2dict(file_path, begin_row, id_col, other_list_cols):
    poi_set = {}
    with open(file_path, "r", encoding='utf-8') as file:
        reader = csv.reader(file, delimiter='\01')
        for index, row in tqdm(enumerate(reader)):
            if index >= begin_row:
                poi_id = row[id_col]
                other_list_attr = [row[col] for col in other_list_cols]
                poi_set[poi_id] = other_list_attr

    return poi_set

#导入csv文件数据集
def load_data2list(file_path, begin_row, list_cols):
    poi_set = []
    with open(file_path, "r",encoding='utf-8') as file:
        reader = csv.reader(file, delimiter='\01')
        for index, row in tqdm(enumerate(reader)):
            if index >= begin_row:
                list_attr = [row[col] for col in list_cols]
                poi_set.append(list_attr)

    return poi_set

#处理原始数据，特别是json格式
class OurDataset(torch.utils.data.Dataset):
    def __init__(self, userlog_data):
        self.userlog_data = userlog_data

        for iii in tqdm(range(len(self.userlog_data))):
            # query, geohash, clk_poiid, filter_rec_poi_list_id, sess_time_list, sess_query_list, filter_sess_poilist_list_id, start_poiid, sess_candidate_poilist
            self.userlog_data[iii][3] = json.loads(self.userlog_data[iii][3])
            self.userlog_data[iii][5] = json.loads(self.userlog_data[iii][5])
            self.userlog_data[iii][6] = json.loads(self.userlog_data[iii][6])
            self.userlog_data[iii][8] = json.loads(self.userlog_data[iii][8])


    def __len__(self):
        return len(self.userlog_data)

    def __getitem__(self, index):
        return self.userlog_data[index]
    

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


def geo_spec_tok(geohash):
    return ''.join(['['+cd+']' for cd in geohash])
    
def get_query_database(data,tokenizer):
    """
    原论文只用训练集的query作database
    """
    q_database=[]
    q_len_dict = {}
    q_geo_dict= {}
    for sample in data:
        q = sample[0]
        geohash = sample[1]
        # q = q + '[sep]' + geo_spec_tok(geohash)
        q_split = tokenizer.tokenize(q)
        q_len_dict[q] = len(q_split)
        q_geo_dict[q] = geohash
        q_database.append(q)
    return q_database,q_len_dict,q_geo_dict


def get_doc_text(doc_poi,dict_poi_set):
    """
        获取poi点的doc文本表达
    """
    _, doc_poi_name, doc_poi_address, doc_poi_geohash = dict_poi_set[doc_poi]
    return  geo_spec_tok(doc_poi_geohash) + "[SEP]" + \
                    doc_poi_name + "[SEP]" + \
                    doc_poi_address


#  label, num_neg, click_doc, next_q, previous_q, qd_pairs, docs, simq = split_3gen_2w(line)
#  其中docs是1*pos+num_neg*neg    qd_pairs是历史记录的qd，这里需要改成只用q
def make_train_data(tofile,train_data,q_database,q_len_dict,dict_poi_set,q_geo_dict,list_poi_full,neg_sample_num=2):
    # query, geohash, clk_poiid, filter_rec_poi_list_id, sess_time_list, sess_query_list, 
    # filter_sess_poilist_list_id, start_poiid, sess_candidate_poilist
    with open(tofile, "w",encoding="utf8") as fw:
        for sample in tqdm(train_data):
            q = sample[0]
            geohash = sample[1]
            query = q + '[SEP]' + geo_spec_tok(geohash)
            sim_q = get_sim_q(q,q_database,q_len_dict)[-1] 
            sim_q = sim_q +'[SEP]'+geo_spec_tok(q_geo_dict[sim_q])
            next_d = "[empty_d]"
            next_q1 = "[empty_q]"
            next_q2 = "[empty_q]"
            previous_q1 = "[empty_q]"
            previous_q2 = "[empty_q]"
            sess_query_list = sample[5]
            if len(sess_query_list)>=2:
                previous_q1 = sess_query_list[-2]
            if len(sess_query_list)>=3:
                previous_q2 = sess_query_list[-3]

            click_poi = sample[2]
            click_doc = get_doc_text(click_poi,dict_poi_set)
            
            #历史信息
            history = ""
            title = "[empty_d]" #前面都没有点击
            for sess_q in sess_query_list:
                history += sess_q + "\t" + title + "\t"
            
            history += query + "\t"


            query_click_doc = [click_poi]
            query_unclick_doc = []

            query_unclick_doc = sample_neg_not_overlap(neg_sample_num, sample[8][-1], query_click_doc)
            if list_poi_full is not None:
                query_unclick_doc += sample_neg_not_overlap(neg_sample_num, list_poi_full, query_click_doc)
            
            gen_labels = next_q1 + "\t" + next_q2 + "\t" + click_doc + "\t" + next_d  + "\t" + previous_q2 + "\t" + previous_q1 + "\t" + sim_q
            # print(query_unclick_doc)
            for click in query_click_doc:
                d_click = get_doc_text(click,dict_poi_set)
                d_click = history + "====" + d_click
                unclick_seq = ""
                for unclick in query_unclick_doc:
                    unclick_seq += "\t" + get_doc_text(unclick,dict_poi_set)
                unclick_cnt = len(query_unclick_doc)
                # training data example (w=2)
                # label(0/1)训练时候label无用   num_neg(neg doc num)   
                # click_doc(current clicked doc)  next_doc(next clicked doc)    
                #  next_q1  next_q2 previous_q1 previous_q2 qd_pairs(history+current query)   
                # docs(candidates)    simq(supplemental query) 
                fw.write("1" + "\t" + str(unclick_cnt)+ "\t" + gen_labels +  "\t" + d_click  + unclick_seq + "\n")


# testing data example
# label(0/1)   next_q    click_doc(current clicked doc)   previous_q  qd_pairs(history+current query+candidate doc)
def make_test_data(tofile,test_data,q_database,q_len_dict,dict_poi_set,q_geo_dict):
    with open(tofile, "w",encoding="utf8") as fw:
        for sample in tqdm(test_data):
            q = sample[0]
            geohash = sample[1]
            query = q + '[SEP]' + geo_spec_tok(geohash)
            sim_q = get_sim_q(q,q_database,q_len_dict)[-1] 
            sim_q = sim_q +'[SEP]'+geo_spec_tok(q_geo_dict[sim_q])
            next_q = "[empty_q]"
            sess_query_list = sample[5]
            first_q = sess_query_list[0]
            click_poi = sample[2]
            click_doc = get_doc_text(click_poi,dict_poi_set)
            gen_labels = next_q + "\t" + click_doc + "\t" + sim_q

            history = ""
            title = "[empty_d]" #前面都没有点击
            for sess_q in sess_query_list:
                history += sess_q + "\t" + title + "\t"
            history += query + "\t"

            title = click_doc
            s = "1" + "\t" + gen_labels + "\t" + history + title + "\n"
            fw.write(s)
            neg_list = sample[8][-1][:50]  #测试的时候选前50
            pos_list=[click_poi]
            for neg_poi in neg_list:
                title = get_doc_text(neg_poi,dict_poi_set)
                if neg_poi in pos_list:
                    s = "1" + "\t" + gen_labels + "\t" + history + title + "\n"
                    fw.write(s)
                else:
                    s = "0" + "\t" + gen_labels + "\t" + history + title + "\n"
                    fw.write(s)


def sample_test(data,num):
    idx = [i for i in range(len(data))]
    idx = random.sample(idx,num)
    test_data=[]
    for i in idx:
        test_data.append(data[i])
    return test_data

if __name__ == '__main__':
    dict_geohash = load_geohash('/nfs/volume-93-2/meilang/myprojects/poi_pretrain_dense/ColBERT/1_data_process/contextual_mapping_all_city/1_geohash_code.csv')
    dict_poi_set = load_data2dict('/nfs/volume-93-2/meilang/myprojects/poi_contextual_rerank/dataset/process/1_poi_need_attr.csv', 1, 0, [i for i in range(4)])
    list_poi_full = list(dict_poi_set.keys())

    print('loading data...')
    train_data = load_data2list('/nfs/volume-93-2/meilang/myprojects/poi_contextual_rerank/dataset/process/5_sampled_userlog_with_level_train_with_candidate.csv', 1, [i for i in range(9)])
    train_data = OurDataset(train_data)

    test_data = load_data2list('/nfs/volume-93-2/meilang/myprojects/poi_contextual_rerank/dataset/process/5_sampled_userlog_with_level_test_with_candidate.csv', 1, [i for i in range(9)])
    test_data = OurDataset(test_data)
    print('end loading...')

    tokenizer = BertTokenizerFast.from_pretrained('/nfs/volume-93-2/meilang/myprojects/poi_pretrain_dense/download_pretrained_model/bert-base-chinese/')
    geo_special_tokens_dict = ['[' + gcd + ']' for gcd in dict_geohash]
    tokenizer.add_tokens(geo_special_tokens_dict)

    q_database,q_len_dict,q_geo_dict = get_query_database(train_data,tokenizer)

    print('making training data...')
    make_train_data('./data/poi/train.txt',train_data,q_database,q_len_dict,dict_poi_set,q_geo_dict,list_poi_full,neg_sample_num=2)
    print('finish...')

    print('making testing data...')
    # test_data = sample_test(test_data, 2000)
    make_test_data("./data/poi/test.txt",test_data,q_database,q_len_dict,dict_poi_set,q_geo_dict)
    print('finish...')