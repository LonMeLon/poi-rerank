"""

new utils for poi-search       updated on 2023.04.11

"""

from collections import Counter, defaultdict
import torch
from transformers import BertTokenizerFast
import csv, random, json
from tqdm import tqdm
from torch.utils.data import Dataset
import numpy as np



reform_type = ["generalize", "exploration", "exploiation", "new task"]
stops = []  # 可以考虑后续加停用词
def get_type(prev, nexts, tokenizer):
    prev_tokens = tokenizer.tokenize(prev, add_special_tokens=False)
    nexts_tokens = tokenizer.tokenize(nexts, add_special_tokens=False)
    prevCounter, nextsCounter = Counter(), Counter()
    for token in prev_tokens:
        if token not in stops:
            prevCounter[token] += 1
    for token in nexts_tokens:
        if token not in stops:
            nextsCounter[token] += 1
    common = prevCounter & nextsCounter
    retained = common
    removed = prevCounter - common
    added = nextsCounter - common
    if retained:
        if removed and added:
            return 1
        elif removed:
            return 0
        else:
            return 2
    else:
        return 3

# 导入geohash
def load_geohash(geohash_path=r'data\1_geohash_code.csv'):
    dict_geohash = {}
    with open(geohash_path, "r") as file:
        reader = csv.reader(file)
        for row in reader:
            hashcode = row[0]
            dict_geohash.setdefault(hashcode, len(dict_geohash))
    return dict_geohash


# dict_geohash = load_geohash()

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

#处理原始数据，特别是json格式，并计算reform type
class OurDataset(torch.utils.data.Dataset):
    def __init__(self, userlog_data):
        self.userlog_data = userlog_data

        tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')

        for iii in tqdm(range(len(self.userlog_data))):
            # query, geohash, clk_poiid, filter_rec_poi_list_id, sess_time_list, sess_query_list, filter_sess_poilist_list_id, start_poiid, sess_candidate_poilist
            self.userlog_data[iii][3] = json.loads(self.userlog_data[iii][3])
            self.userlog_data[iii][5] = json.loads(self.userlog_data[iii][5])
            self.userlog_data[iii][6] = json.loads(self.userlog_data[iii][6])
            self.userlog_data[iii][8] = json.loads(self.userlog_data[iii][8])

            # get reform label
            len_sess_q_list = len(self.userlog_data[iii][5])
            reform_type = []
            for i in range(len_sess_q_list - 1):
                reform_type.append(get_type(self.userlog_data[iii][5][i], self.userlog_data[iii][5][i + 1], tokenizer))
            self.userlog_data[iii].append(reform_type)

    def __len__(self):
        return len(self.userlog_data)

    def __getitem__(self, index):
        return self.userlog_data[index]


# 获取数据集所有token，为vocab类做准备
def get_init_tokens(dict_geohash,train_data,list_poi_full,dict_poi_set,tokenizer):
    all_tokens = []
    geo_special_tokens = ['['+gcd+']' for gcd in dict_geohash] #geo tokens
    for sample in train_data:  # query tokens
        history_querys = sample[5]
        for q in history_querys:
            all_tokens += tokenizer.tokenize(q,add_special_tokens=False)
    # poi tokens(doc tokens)
    for poi in list_poi_full:
        address = dict_poi_set[poi][2]+dict_poi_set[poi][1]
        all_tokens += tokenizer.tokenize(address,add_special_tokens=False)
    all_tokens = list(set(all_tokens))
    all_tokens.extend(['[sep]'])
    all_tokens += geo_special_tokens
    return all_tokens



#训练数据dataset构建
"""
对于训练集中的每个会话S，我们将第一个查询以外的查询视为当前查询，然后对候选文档进行排序
"""
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

def get_pair_data(train_data, neg_sample_num=2,list_poi_full=None):
    # train_data
    # query, geohash, clk_poiid, filter_rec_poi_list_id, sess_time_list, sess_query_list,
    # filter_sess_poilist_list_id, start_poiid, sess_candidate_poilist, reform_type
    pair_data = []
    for qid in range(len(train_data)):
        newsample = dict()
        samplelist = train_data[qid]
        pos_list = [samplelist[2]]
        # 抽取2个hard  再从总的里random 2个
        neg_list = sample_neg_not_overlap(neg_sample_num, samplelist[8][0], pos_list)
        if list_poi_full is not None:
            neg_list += sample_neg_not_overlap(neg_sample_num, list_poi_full, pos_list)
        # if not neg_list or not pos_list or len(samplelist[9]) == 0:
        #     continue
        newsample['query_id'] = qid  # 和模型输入没啥关系
        newsample['geohash'] = samplelist[1]
        newsample['current_query'] = samplelist[0]
        newsample['history'] = {'sess_q_list': samplelist[5][:-1], 'sess_poi_list': samplelist[6][:-1]}
        newsample['reform_type'] = samplelist[-1]
        newsample['pos_candidate'] = pos_list[-1]
        newsample['neg_candidate'] = neg_list
        pair_data.append(newsample)
    return pair_data

class PairHQCNDataset(Dataset):
    def __init__(self, max_query_length, max_doc_length, dataset, vocab, dict_poi_set,tokenizer,history_num=3):
        self.max_query_length = max_query_length
        self.max_doc_length = max_doc_length
        self._dataset = dataset
        self.vocab = vocab
        self._total_data = len(dataset)
        self.dict_poi_set = dict_poi_set
        self.tokenizer = tokenizer
        self.history_num = history_num

    def __len__(self):
        return self._total_data

    def __getitem__(self, idx):
        # queries_ids包括[history queryies, current query] 长度为sl
        sample = self._dataset[idx]
        guid = sample['query_id']  #当前(d+,d-)对应哪个q
        hq, hpc, hnc = list(), list(), list()
        #遍历历史， 改为长度不定的输入
        history = sample['history']
        sess_q_list = history['sess_q_list'] #不包含current query
        his_len = len(sess_q_list)
        for i in range(his_len):
            sess_q = sess_q_list[i]
            hq.append(sess_q)
        if his_len==0:
            hq.append('[pad]')
        q = sample['current_query']
        geohash = sample['geohash']
        hq.append(self.geo_spec_tok(geohash) + '[sep]' + q)
        hpc.append(self._get_doc_text(sample['pos_candidate']))
        for neg_poi in sample['neg_candidate']:
            hnc.append(self._get_doc_text(neg_poi))
        wss_label = sample['reform_type']
        if wss_label == []:
            wss_label = [3]   #对于一个session只有一个query特殊处理（复制） ---new task
        else:
            wss_label = [wss_label[-1]]
        s_index = len(hq) - 1

        if 'retained' in sample:
            retained, added, removed = sample['retained'], sample['added'], sample['removed']
        else:
            retained, added, removed  = [], [], []

        paded_retained, paded_added, paded_removed = self._reform_pad(retained, added, removed)

        paded_q, paded_pd, paded_nd = self._ids_pad(hq, hpc, hnc)
        batch = {'queries_ids': np.asarray(paded_q), 'documents_pos_ids': np.asarray(paded_pd), 'documents_neg_ids': np.asarray(paded_nd),
        'retain_label': np.asarray(paded_retained), 'add_label': np.asarray(paded_added), 'remove_label': np.asarray(paded_removed),
        'wss_label':np.asarray(wss_label), 'guid': guid, 's_index': s_index, 'query_id':str(guid)+'_q', 'doc_id':str(guid)+'_d'}

        return batch

    def geo_spec_tok(self,geohash):
        return ''.join(['[' + cd + ']' for cd in geohash])

    def _get_doc_text(self,doc_poi):
        """
        获取poi点的doc文本表达
        """
        _, doc_poi_name, doc_poi_address, doc_poi_geohash = self.dict_poi_set[doc_poi]
        return  self.geo_spec_tok(doc_poi_geohash) + '[sep]' + \
                    doc_poi_name + '[sep]' + \
                    doc_poi_address

    def _ids_pad(self, hq, hpc, hnc):
        pad_id = self.vocab.get_id(self.vocab.pad_token)
        padded_q, padded_pd, padded_nd = list(), list(), list()
        for q in hq:
            # tokens = q.split()
            tokens = self.tokenizer.tokenize(q)
            wid = [self.vocab.get_id(token) for token in tokens]
            wid = wid[:self.max_query_length]
            wid = wid + [pad_id] * (self.max_query_length - len(tokens))
            padded_q.append(wid)
        padded_q = padded_q[-self.history_num:]
        while len(padded_q) < self.history_num:
            padded_q.append([pad_id] * self.max_query_length)
        padded_q = [np.asarray(item) for item in padded_q]

        for d in hpc:
            # tokens = d.split()
            tokens = self.tokenizer.tokenize(d)
            wid = [self.vocab.get_id(token) for token in tokens]
            wid = wid[:self.max_doc_length]
            wid = wid + [pad_id] * (self.max_doc_length - len(tokens))
            padded_pd.append(wid)
        padded_pd = padded_pd[-self.history_num:]
        while len(padded_pd) < self.history_num:
            padded_pd.append([pad_id] * self.max_doc_length)
        padded_pd = [np.asarray(item) for item in padded_pd]

        for d in hnc:
            # tokens = d.split()
            tokens = self.tokenizer.tokenize(d)
            wid = [self.vocab.get_id(token) for token in tokens]
            wid = wid[:self.max_doc_length]
            wid = wid + [pad_id] * (self.max_doc_length - len(tokens))
            padded_nd.append(wid)
        padded_nd = padded_nd[-self.history_num:]
        while len(padded_nd) < self.history_num:
            padded_nd.append([pad_id] * self.max_doc_length)
        padded_nd = [np.asarray(item) for item in padded_nd]

        return padded_q, padded_pd, padded_nd

    def _reform_pad(self, retain, add, remove):
        pad_id = self.vocab.get_id(self.vocab.pad_token)

        retain = retain[:self.max_query_length]
        retain = retain + [0] * (self.max_query_length - len(retain))

        add = add[:self.max_query_length]
        add = add + [0] * (self.max_query_length - len(add))

        remove = remove[:self.max_query_length]
        remove = remove + [0] * (self.max_query_length - len(remove))

        return retain, add, remove


#测试数据集构建，从候选集中抽取前50个，注意其中可能已经有正样本 (这里包含了只有一个query的数据)
# 抽取候选集前50 点击label是1 没点击是0
def get_test_data(data,idx=None):
    test_data=[]
    if idx == None:
        idx = [i for i in range(len(data))]
    for qid in idx:
        newsample = dict()
        samplelist = data[qid]
        pos_list = [samplelist[2]]
        neg_list = samplelist[8][0][:50]  #测试的时候选前50
        newsample['query_id'] = qid  #和模型输入没啥关系
        newsample['geohash'] = samplelist[1]
        newsample['current_query'] = samplelist[0]
        newsample['history'] = {'sess_q_list':samplelist[5][:-1],'sess_poi_list':samplelist[6][:-1]}
        try:
            newsample['reform_type'] = samplelist[-1]
        except:
            newsample['reform_type'] = 3 #新任务
        newsample['candidate'] = pos_list[-1]
        newsample['label']=1 #点击
        test_data.append(newsample)
        for neg in neg_list:
            newsample = newsample.copy()
            newsample['candidate'] = neg
            newsample['label']=0
            if neg in pos_list:
                newsample['label']=1
            test_data.append(newsample)
    return test_data


class HQCNDataset(Dataset):
    def __init__(self, max_query_length, max_doc_length, dataset, vocab, dict_poi_set, tokenizer, history_num=3):
        self.max_query_length = max_query_length
        self.max_doc_length = max_doc_length
        self._dataset = dataset
        self.vocab = vocab
        self._total_data = len(dataset)
        self.dict_poi_set = dict_poi_set
        self.tokenizer = tokenizer
        self.history_num = history_num

    def __len__(self):
        return self._total_data

    def __getitem__(self, idx):

        sample = self._dataset[idx]
        guid = sample['query_id']
        hq, hc = list(), list()
        history = sample['history']
        sess_q_list = history['sess_q_list']  # 不包含current query
        his_len = len(sess_q_list)
        if his_len==0:
            hq.append('[pad]')
        for i in range(his_len):
            sess_q = sess_q_list[i]
            hq.append(sess_q)
        q = sample['current_query']
        geohash = sample['geohash']
        hc.append(self._get_doc_text(sample['candidate']))
        hq.append(self.geo_spec_tok(geohash) + '[sep]' + q)

        wss_label = sample['reform_type']
        if wss_label == []:
            wss_label = [3]  # 对于一个session只有一个query特殊处理（复制）
        else:
            wss_label = [wss_label[-1]]
        label = sample['label']
        s_index = len(hq) - 1
        paded_q, paded_d = self._ids_pad(hq, hc)

        batch = {'queries_ids': np.asarray(paded_q), 'documents_ids': np.asarray(paded_d),
                 'wss_label': np.asarray(wss_label), 'guid': guid, 's_index': s_index, 'query_id': str(guid) + '_q',
                 'doc_id': str(guid) + '_d', 'label': label}
        # print('batch',batch)
        return batch

    def geo_spec_tok(self,geohash):
        return ''.join(['[' + cd + ']' for cd in geohash])

    def _get_doc_text(self,doc_poi):
        """
        获取poi点的doc文本表达
        """
        _, doc_poi_name, doc_poi_address, doc_poi_geohash = self.dict_poi_set[doc_poi]
        return  self.geo_spec_tok(doc_poi_geohash) + '[sep]' + \
                    doc_poi_name + '[sep]' + \
                    doc_poi_address

    def _ids_pad(self, hq, hc):
        pad_id = self.vocab.get_id(self.vocab.pad_token)
        padded_q, padded_d = list(), list()
        for q in hq:
            # tokens = q.split()
            tokens = self.tokenizer.tokenize(q)
            wid = [self.vocab.get_id(token) for token in tokens]
            wid = wid[:self.max_query_length]
            wid = wid + [pad_id] * (self.max_query_length - len(tokens))
            padded_q.append(wid)
        padded_q = padded_q[-self.history_num:]
        while len(padded_q) < self.history_num:
            padded_q.append([pad_id] * self.max_query_length)
        padded_q = [np.asarray(item) for item in padded_q]

        for d in hc:
            # tokens = d.split()
            tokens = self.tokenizer.tokenize(d)
            wid = [self.vocab.get_id(token) for token in tokens]
            wid = wid[:self.max_doc_length]
            wid = wid + [pad_id] * (self.max_doc_length - len(tokens))
            padded_d.append(wid)
        padded_d = padded_d[-self.history_num:]
        while len(padded_d) < self.history_num:
            padded_d.append([pad_id] * self.max_doc_length)
        padded_d = [np.asarray(item) for item in padded_d]

        return padded_q, padded_d

    def _reform_pad(self, retain, add, remove):
        pad_id = self.vocab.get_id(self.vocab.pad_token)

        retain = retain[:self.max_query_length]
        retain = retain + [0] * (self.max_query_length - len(retain))

        add = add[:self.max_query_length]
        add = add + [0] * (self.max_query_length - len(add))

        remove = remove[:self.max_query_length]
        remove = remove + [0] * (self.max_query_length - len(remove))

        return retain, add, remove