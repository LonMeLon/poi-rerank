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

class OurDataset(torch.utils.data.Dataset):
    def __init__(self, userlog_data):
        self.userlog_data = userlog_data

        for iii in tqdm(range(len(self.userlog_data))):
            # query, geohash, clk_poiid, filter_rec_poi_list_id, sess_time_list, sess_query_list, filter_sess_poilist_list_id, start_poiid, sess_candidate_poilist
            self.userlog_data[iii][5] = json.loads(self.userlog_data[iii][5])
            self.userlog_data[iii][8] = json.loads(self.userlog_data[iii][8])
        
    def __len__(self):
        return len(self.userlog_data)

    def __getitem__(self, index):
        return self.userlog_data[index]



def main(args):
    # initalize
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    torch.distributed.init_process_group(backend="nccl")


    # prepare
    # poi 
    dict_poi_set = load_data2dict(args.poi_path, 1, 0, [i for i in range(4)])
    list_poi_full = list(dict_poi_set.keys())
    print(args.local_rank, len(dict_poi_set), len(list_poi_full))
    # goehash d
    dict_geohash = load_geohash(args.geohash_path)
    # query, geohash, clk_poiid, filter_rec_poi_list_id, sess_time_list, sess_query_list, filter_sess_poilist_list_id, start_poiid, sess_candidate_poilist
    train_data = load_data2list(args.train_clicklog_path, 1, [i for i in range(9)])
    print(args.local_rank, len(train_data))
    train_data = OurDataset(train_data)
    
    
    # ----- all train data
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
    train_dataloader = torch.utils.data.DataLoader(
        train_data, 
        sampler=train_sampler, 
        batch_size=args.batch_size,
        num_workers=4, 
        pin_memory=True,
        collate_fn=lambda x:x,
    )

    # tokenizer
    tokenizer = BertTokenizerFast.from_pretrained(args.plm_path)
    print(args.local_rank, len(tokenizer))
    geo_special_tokens_dict = ['['+gcd+']' for gcd in dict_geohash]
    tokenizer.add_tokens(geo_special_tokens_dict)
    print(args.local_rank, len(tokenizer))


    
    # ----- model load
    itmodel = model.bert_rerank.from_pretrained(args.plm_path, args)
    itmodel.resize_token_embeddings(len(tokenizer))

    itmodel = itmodel.to(device)
    itmodel.train()

    itmodel = nn.parallel.DistributedDataParallel(
        itmodel, 
        device_ids=[args.local_rank],
        output_device=args.local_rank,
    )

    # optimizer
    optimizer = AdamW(filter(lambda p: p.requires_grad, itmodel.parameters()), lr=args.learning_rate, eps=1e-8)
    scaler = torch.cuda.amp.GradScaler() if args.mix_amp == 1 else None
    autocast = torch.cuda.amp.autocast if args.mix_amp == 1 else suppress


    # training
    for epoch in range(args.epoch):
        # shuffle
        train_sampler.set_epoch(epoch)
        
        # save
        if args.local_rank == 0:
            print('save checkpoint', args.local_rank)
            save_checkpoint(args.save_checkpoint_path, itmodel.module, epoch) 
        
        # batch
        for idx_,  batch_ in tqdm(enumerate(train_dataloader)):
            assert args.local_rank == torch.distributed.get_rank()
            
            query_poi_context_ = []
            query_poi_label_ = []
            
            batch_had_poids = {}

            
            this_batch_size = len(batch_)
            for kkkkk in range(this_batch_size):
                query_ = batch_[kkkkk][0]
                q_geohash = batch_[kkkkk][1]
                clk_poiid = batch_[kkkkk][2]
                
                sess_query_list = batch_[kkkkk][5]
                clicked_query_candidate_poilist = batch_[kkkkk][8][-1]

                assert query_ == sess_query_list[-1]
                assert len(clicked_query_candidate_poilist) == 100


                # click poi
                batch_had_poids.setdefault(clk_poiid, 0)

                _, pos_poi_name, pos_poi_address, pos_poi_geohash = dict_poi_set[clk_poiid]
                query_poi_context_.append(
                    geo_spec_tok(q_geohash) + '[SEP]' + \
                    '[SEP]'.join(sess_query_list) + '[SEP]' + \
                    geo_spec_tok(pos_poi_geohash) + '[SEP]' + \
                    pos_poi_name + '[SEP]' + \
                    pos_poi_address
                )
                query_poi_label_.append(1)

                # rand neg poi
                set_rand_neg_poid = sample_neg_not_overlap(args.rand_neg_num_per, list_poi_full, batch_had_poids)
                assert len(set_rand_neg_poid) == args.rand_neg_num_per
                for rand_neg_poid in set_rand_neg_poid:
                    batch_had_poids.setdefault(rand_neg_poid, 0)
                    _, rand_neg_poi_name, rand_neg_poi_address, rand_neg_poi_geohash = dict_poi_set[rand_neg_poid]
                    query_poi_context_.append(
                        geo_spec_tok(q_geohash) + '[SEP]' + \
                        '[SEP]'.join(sess_query_list) + '[SEP]' + \
                        geo_spec_tok(rand_neg_poi_geohash) + '[SEP]' + \
                        rand_neg_poi_name + '[SEP]' + \
                        rand_neg_poi_address
                    )
                    query_poi_label_.append(0)

                # hard neg poi
                set_hard_neg_poid = sample_neg_not_overlap(args.hard_neg_num_per, clicked_query_candidate_poilist, batch_had_poids)
                assert len(set_hard_neg_poid) == args.hard_neg_num_per
                for hard_neg_poid in set_hard_neg_poid:
                    batch_had_poids.setdefault(hard_neg_poid, 0)
                    _, hard_neg_poi_name, hard_neg_poi_address, hard_neg_poi_geohash = dict_poi_set[hard_neg_poid]
                    query_poi_context_.append(
                        geo_spec_tok(q_geohash) + '[SEP]' + \
                        '[SEP]'.join(sess_query_list) + '[SEP]' + \
                        geo_spec_tok(hard_neg_poi_geohash) + '[SEP]' + \
                        hard_neg_poi_name + '[SEP]' + \
                        hard_neg_poi_address
                    )
                    query_poi_label_.append(0)
                    
            assert len(query_poi_context_) == len(query_poi_label_) == this_batch_size * (1 + args.rand_neg_num_per + args.hard_neg_num_per)
            
            query_poi_context_ = tokenizer(query_poi_context_, padding=True, return_tensors='pt')

            with autocast():
                # score
                scores_ = itmodel.module.score(
                    query_poi_context_['input_ids'].to(device), 
                    query_poi_context_['attention_mask'].to(device), 
                    query_poi_context_['token_type_ids'].to(device),
                )
                scores_ = torch.sigmoid(scores_.reshape(-1))
                labels_scores = torch.FloatTensor(query_poi_label_).to(scores_.device)
                
                # loss
                loss = itmodel.module.criterion(scores_, labels_scores.reshape(-1))
                
            # backward
            if scaler is not None:
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(itmodel.parameters(), 2.0)
                scaler.step(optimizer)  
                scaler.update()
            else:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(itmodel.parameters(), 2.0)
                optimizer.step()


            # print
            if (idx_ + 1) % 1000 == 0 and args.local_rank == 0:
                print(epoch, args.local_rank, (idx_ + 1) * args.batch_size)
                print('rank:{}, loss:{}, '.format(args.local_rank, loss.item(),))
            

if __name__ == "__main__":
    # parameter
    parser = ArgumentParser()
    

    # multi gpu
    parser.add_argument('--local_rank', default=0, type=int, help='node rank for distributed training')
    # apex
    parser.add_argument('--mix_amp', default=0, type=int,)


    # data
    # poi
    parser.add_argument('--poi_path', type=str, default="/nfs/volume-93-2/meilang/myprojects/poi_contextual_rerank/dataset/process/1_poi_need_attr.csv")
    # clicklog
    parser.add_argument('--train_clicklog_path', type=str, default="/nfs/volume-93-2/meilang/myprojects/poi_contextual_rerank/dataset/process/5_sampled_userlog_with_level_train_with_candidate.csv")
    # geohash
    parser.add_argument('--geohash_path', type=str, default='/nfs/volume-93-2/meilang/myprojects/poi_pretrain_dense/ColBERT/1_data_process/contextual_mapping_all_city/1_geohash_code.csv')
    

    # model
    parser.add_argument('--plm_path', type=str, default='/nfs/volume-93-2/meilang/myprojects/poi_pretrain_dense/download_pretrained_model/bert-base-chinese/')

    # checkpoint
    parser.add_argument('--save_checkpoint_path', type=str, default='checkpoint/')

    
    # run
    # learning rate
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    # batch size
    parser.add_argument('--batch_size', type=int, default=16)
    # epoch
    parser.add_argument('--epoch', type=int, default=10)
    # rand neg num per click
    parser.add_argument('--rand_neg_num_per', type=int, default=2)
    # hard neg num per click
    parser.add_argument('--hard_neg_num_per', type=int, default=2)

    
    args = parser.parse_args()
    print(args)

    
    # main
    main(args)