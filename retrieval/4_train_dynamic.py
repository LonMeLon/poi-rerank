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


class OurDataset(torch.utils.data.Dataset):
    def __init__(self, userlog_data):
        self.userlog_data = userlog_data
        
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
    # query, geohash, clk_poiid, rec_poi_list_id
    train_data = load_data2list(args.train_clicklog_path, 1, [i for i in range(4)])
    print(args.local_rank, len(train_data))
    train_data = OurDataset(train_data)



    # Faiss gpu
    ### poi vec
    infer_poi_vecs = load_obj(args.load_index_path + '/infer_poi_vecs.numpy')
    faiss_index = faiss.IndexFlatIP(infer_poi_vecs.shape[-1])
    print('faiss_index', faiss_index, args.local_rank)
    faiss_index.add(infer_poi_vecs)
    singlegpu = faiss.StandardGpuResources()
    faiss_index = faiss.index_cpu_to_gpu(singlegpu, args.local_rank, faiss_index)

    infer_poi_vecs = torch.from_numpy(infer_poi_vecs).to(device)
    ### poid
    poi_ids2vecs = load_obj(args.load_index_path + '/poi_ids2vecs.list')
    dict_poid2vecid = {}
    dict_vecid2poid = {}
    for index, poid in enumerate(poi_ids2vecs):
        dict_poid2vecid[poid] = index
        dict_vecid2poid[index] = poid
    print(infer_poi_vecs.shape, len(dict_poid2vecid), len(dict_vecid2poid))
    
    
    # ----- all train data
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
    train_dataloader = torch.utils.data.DataLoader(
        train_data, 
        sampler=train_sampler, 
        batch_size=args.batch_size,
        num_workers=4, 
        pin_memory=True,
    )

    # tokenizer
    tokenizer = BertTokenizerFast.from_pretrained(args.plm_path)
    print(args.local_rank, len(tokenizer))
    geo_special_tokens_dict = ['['+gcd+']' for gcd in dict_geohash]
    tokenizer.add_tokens(geo_special_tokens_dict)
    print(args.local_rank, len(tokenizer))


    
    # ----- model load
    itmodel = model.Bert_geo.from_pretrained(args.plm_path, args)
    itmodel.resize_token_embeddings(len(tokenizer))
    load_checkpoint(args.load_checkpoint_path, itmodel)
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
        
        # batch
        for idx_,  batch_ in tqdm(enumerate(train_dataloader)):
            assert args.local_rank == torch.distributed.get_rank()
            ### query
            batch_query_context = []
            ### click poi 
            batch_pos_poids = []
            ### neg poi
            batch_had_poids = {}
            ##### rand neg poi
            batch_rand_neg_poids = []

            
            this_batch_size = len(batch_[0])
            for kkkkk in range(this_batch_size):
                query = batch_[0][kkkkk]
                geohash = batch_[1][kkkkk]
                clk_poiid = batch_[2][kkkkk]
        
                # query
                batch_query_context.append(
                    geo_spec_tok(geohash) + '[SEP]' + \
                    query
                )
                # click poi
                batch_had_poids.setdefault(clk_poiid, 0)

                batch_pos_poids.append(clk_poiid)
                # rand neg poi
                set_rand_neg_poid = sample_neg_not_overlap(args.rand_neg_num_per, list_poi_full, batch_had_poids)
                assert len(set_rand_neg_poid) > 0
                
                for rand_neg_poid in set_rand_neg_poid:
                    batch_had_poids.setdefault(rand_neg_poid, 0)
                    
                    batch_rand_neg_poids.append(rand_neg_poid)

            ### query
            batch_query_context = tokenizer(batch_query_context, padding=True, return_tensors='pt')


            with autocast():
                # query emb
                query_context_emb = itmodel.module.cls_(
                    batch_query_context['input_ids'].to(device), 
                    batch_query_context['attention_mask'].to(device), 
                    batch_query_context['token_type_ids'].to(device),
                )
                # hard neg poi
                batch_hard_neg_poid2index = []
                _, ranks_dynamic = faiss_index.search(query_context_emb.detach().cpu().numpy(), 50)
                for bii in range(ranks_dynamic.shape[0]):
                    #hard_indexes = np.random.choice(ranks_dynamic[bii], args.hard_neg_num_per, replace=False)
                    #batch_hard_neg_poid2index += list(hard_indexes)

                    hard_indexes = list(ranks_dynamic[bii])
                    batch_hard_neg_poid2index += sample_neg_not_overlap(args.hard_neg_num_per, hard_indexes, batch_had_poids)

                
                pos_poi_context_emb = infer_poi_vecs[[dict_poid2vecid[pos_poid_] for pos_poid_ in batch_pos_poids]]
                rand_neg_poi_context_emb = infer_poi_vecs[[dict_poid2vecid[rand_id_] for rand_id_ in batch_rand_neg_poids]]
                hard_neg_poi_context_emb = infer_poi_vecs[batch_hard_neg_poid2index]

                # score
                scores_pos = itmodel.module.score_pair(query_context_emb, pos_poi_context_emb)  
                scores_inbatch_neg = itmodel.module.score_inbatch_neg(query_context_emb, pos_poi_context_emb)
                scores_rand_neg = itmodel.module.score_all(query_context_emb, rand_neg_poi_context_emb)
                scores_hard_neg = (itmodel.module.score_pair(query_context_emb.repeat(1, args.hard_neg_num_per).reshape(-1, query_context_emb.shape[1]), hard_neg_poi_context_emb)).reshape(-1, args.hard_neg_num_per) # hard neg: [batch, hard_neg_num_per]  
                
                
                scores = torch.cat(
                    [
                        scores_pos.reshape(-1, 1), 
                        scores_inbatch_neg, 
                        scores_rand_neg,
                        scores_hard_neg,
                    ], 
                    dim=1
                )
                labels_scores = torch.zeros(scores.shape[0], dtype=torch.long).to(scores.device)
                
                # loss
                loss = itmodel.module.criterion(scores / args.temp, labels_scores) 
                
                pos_neg_rand_diff = scores_pos.mean() - torch.cat([scores_inbatch_neg,scores_rand_neg,], dim=1).mean()
                pos_neg_hard_diff = scores_pos.mean() - scores_hard_neg.mean()


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
                print('rank:{}, loss:{}, pos_neg_rand_diff:{}, pos_neg_hard_diff:{}'.format(args.local_rank, loss.item(), pos_neg_rand_diff.item(), pos_neg_hard_diff.item()))

        
        # save
        if args.local_rank == 0:
            print('save checkpoint', args.local_rank)
            save_checkpoint('checkpoint/dynamic_temp0.1/', itmodel.module, epoch + 1)

if __name__ == "__main__":
    # parameter
    parser = ArgumentParser()
    

    # multi gpu
    parser.add_argument('--local_rank', default=0, type=int, help='node rank for distributed training')
    # apex
    parser.add_argument('--mix_amp', default=1, type=int, )


    # data
    # poi
    parser.add_argument('--poi_path', type=str, default="/nfs/volume-93-2/meilang/myprojects/poi_contextual_rerank/dataset/process/1_poi_need_attr.csv")
    # clicklog
    parser.add_argument('--train_clicklog_path', type=str, default="/nfs/volume-93-2/meilang/myprojects/poi_contextual_rerank/dataset/process/5_all_userlog_with_level.csv")
    # geohash
    parser.add_argument('--geohash_path', type=str, default='/nfs/volume-93-2/meilang/myprojects/poi_pretrain_dense/ColBERT/1_data_process/contextual_mapping_all_city/1_geohash_code.csv')
    

    # model
    parser.add_argument('--plm_path', type=str, default='/nfs/volume-93-2/meilang/myprojects/poi_pretrain_dense/download_pretrained_model/bert-base-chinese/')

    # checkpoint
    parser.add_argument('--save_checkpoint_path', type=str, default='checkpoint/dynamic_temp0.1/')
    parser.add_argument('--load_checkpoint_path', type=str, default='checkpoint/temp0.1/model-5.ck')
    # index
    parser.add_argument('--load_index_path', type=str, default='index/checkpoint/temp0.1/model-5.ck')

    
    # run
    # contrastive temp
    parser.add_argument('--temp', type=float, default=0.1)
    # learning rate
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    # batch size
    parser.add_argument('--batch_size', type=int, default=64)
    # epoch
    parser.add_argument('--epoch', type=int, default=5)
    # rand neg num per click
    parser.add_argument('--rand_neg_num_per', type=int, default=1)
    # hard neg num per click
    parser.add_argument('--hard_neg_num_per', type=int, default=4)

    
    args = parser.parse_args()
    print(args)

    
    # main
    main(args)