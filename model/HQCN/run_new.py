from __future__ import absolute_import, division, print_function

from argparse import ArgumentParser
import glob
import logging
import os
import random
import json

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
#from tensorboardX import SummaryWriter
from tqdm import tqdm, trange

#from pytorch_transformers import (WEIGHTS_NAME, BertConfig, BertTokenizer)
from Models import HQCN
import random
from torch.optim import AdamW
#from pytorch_transformers import WarmupLinearSchedule
from utils import *
from scipy.special import softmax
from vocab import Vocab
from evaluate import Map, mrr, ndcg
import copy
import pickle
from contextlib import suppress

logger = logging.getLogger(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def save_checkpoint(checkpoint_path, model, epoch, ):
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    checkpoint = {}
    checkpoint['model_state_dict'] = model.state_dict()
    torch.save(checkpoint, checkpoint_path + '/model-' + str(epoch) + '.ck')

def evaluate(args, eval_dataset, model, vocab, batch_size=64):

    eval_dataloader = DataLoader(eval_dataset,batch_size=batch_size)

    # Eval!
    maps, mrrs_1, mrrs_3, mrrs_5, mrrs_10, ndcg1, ndcg3, ndcg5, ndcg10, count = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    for batch in tqdm(eval_dataloader
            # , desc="Evaluating"
                      ):
        model.eval()
        eval_guids = batch['guid']
        # gt = [1] + [0] * neg_num
        gt = batch['label'].to(device)
        with torch.no_grad():
            inputs = {'queries': batch['queries_ids'].to(device),
                      'documents': batch['documents_ids'].to(device),
                      'can_index': batch['s_index'].to(device),
                      'wss_label': batch['wss_label'].to(device)}

            pred, _ = model(**inputs)

            mrrs_1 += mrr(gt, pred, 1)
            mrrs_3 += mrr(gt, pred, 3)
            mrrs_5 += mrr(gt, pred, 5)
            mrrs_10 += mrr(gt, pred, 10)

            ndcg1 += ndcg(1)(gt, pred)
            ndcg3 += ndcg(3)(gt, pred)
            ndcg5 += ndcg(5)(gt, pred)
            ndcg10 += ndcg(10)(gt, pred)

            count += 1

    print('*' * 100)
    print('MAP ', 1.0 * maps / count)
    print('MRR@1 ', 1.0 * mrrs_1 / count)
    print('MRR@3 ', 1.0 * mrrs_3 / count)
    print('MRR@5 ', 1.0 * mrrs_5 / count)
    print('MRR@10 ', 1.0 * mrrs_10 / count)
    print('NDCG@1 ', 1.0 * ndcg1 / count)
    print('NDCG@3 ', 1.0 * ndcg1 / count)
    print('NDCG@5 ', 1.0 * ndcg5 / count)
    print('NDCG@10 ', 1.0 * ndcg10 / count)
    print('*' * 100)

    return

def save_checkpoint(checkpoint_path, model, epoch, ):
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    checkpoint = {}
    checkpoint['model_state_dict'] = model.state_dict()
    torch.save(checkpoint, checkpoint_path + '/model-' + str(epoch) + '.ck')

def train(args, train_dataset, eval_dataset, model, vocab, alpha=0.8):
    """ Train the model """
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    model = model.to(device)
    model.train()

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    # no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs


    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    args.warmup_steps = int(t_total * args.warmup_portion)
    #scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)


    # Train!
    tr_loss=.0
    global_step=0
    for epoch in range(args.epoch):

        # save
        if args.local_rank == 0:
            print('save checkpoint', args.local_rank)
            save_checkpoint(args.save_checkpoint_path, model, epoch)

            # batch
            for idx_, batch in tqdm(enumerate(train_dataloader)):
                model.train()
                guids = batch['guid']
                inputs = {'queries': batch['queries_ids'].to(device),
                          'can_index': batch['s_index'].to(device),
                          'wss_label': batch['wss_label'].to(device)}

                inputs['documents'] = batch['documents_pos_ids'].to(device)
                pos_score, loss_reform = model(**inputs)
                inputs['documents'] = batch['documents_neg_ids'].to(device)
                neg_score, _ = model(**inputs)
                # print(pos_score, neg_score)
                label = torch.ones(pos_score.size()).to(device)
                crit = nn.MarginRankingLoss(margin=1)
                loss = crit(pos_score, neg_score, Variable(label, requires_grad=False))
                loss = alpha * loss + (1 - alpha) * loss_reform
                loss.backward()
                # backward
                tr_loss += loss.item()
                if (idx_ + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    #scheduler.step()  # Update learning rate schedule
                    model.zero_grad()
                    global_step += 1

                # print
                if (idx_ + 1) % 400 == 0 and args.local_rank == 0:
                    print(epoch, (idx_ + 1) * args.batch_size)
                    print('global_step:{}, tr_loss / global_step:{}, '.format(global_step, tr_loss / global_step))
                    print('rank:{}, loss:{}, '.format(args.local_rank, loss.item(), ))
                #if (global_step+1) % 800 == 0:
                #    evaluate(args, eval_dataset, model, vocab, batch_size=51) # 50个候选集+click
            evaluate(args, eval_dataset, model, vocab, batch_size=51) # 50个候选集+click




if __name__ == "__main__":
    print(torch.__version__)

    # parameter
    parser = ArgumentParser()

    # multi gpu
    parser.add_argument('--local_rank', default=0, type=int, help='node rank for distributed training')
    # apex
    parser.add_argument('--mix_amp', default=0, type=int, )

    # data
    # poi
    parser.add_argument('--poi_path', type=str, default="/nfs/volume-93-2/meilang/myprojects/poi_contextual_rerank/dataset/process/1_poi_need_attr.csv")
    # clicklog
    parser.add_argument('--train_clicklog_path', type=str, default="/nfs/volume-93-2/meilang/myprojects/poi_contextual_rerank/dataset/process/5_sampled_userlog_with_level_train_with_candidate.csv")
    
    parser.add_argument('--test_clicklog_path', type=str, default="/nfs/volume-93-2/meilang/myprojects/poi_contextual_rerank/dataset/process/5_sampled_userlog_with_level_test_with_candidate.csv")

    # geohash
    parser.add_argument('--geohash_path', type=str, default='/nfs/volume-93-2/meilang/myprojects/poi_pretrain_dense/ColBERT/1_data_process/contextual_mapping_all_city/1_geohash_code.csv')

    # model
    parser.add_argument('--plm_path', type=str, default='/nfs/volume-93-2/meilang/myprojects/poi_pretrain_dense/download_pretrained_model/bert-base-chinese/')

    # checkpoint
    parser.add_argument('--save_checkpoint_path', type=str, default='checkpoint/')

    # run
    # learning rate
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    # batch size
    parser.add_argument('--batch_size', type=int, default=48)
    # epoch
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_portion", default=0.1, type=float,
                        help="Linear warmup over warmup_steps (=t_total * warmup_portion). override warmup_steps ")
    # rand neg num per click
    parser.add_argument('--rand_neg_num_per', type=int, default=2)
    parser.add_argument("--history_num", default=5, type=int, required=False,
                        help="number of history turns to concat")

    parser.add_argument("--max_query_length", default=60, type=int, required=False,
                        help="max length of query")
    parser.add_argument("--max_doc_length", default=120, type=int, required=False,
                        help="max length of document")
    parser.add_argument("--inner_dim", default=200, type=int, required=False,
                        help="max length of document")
    parser.add_argument("--n_layers", default=6, type=int, required=False,
                        help="layers of transformer")
    parser.add_argument("--n_head", default=8, type=int, required=False,
                        help="head number")
    parser.add_argument("--dropout", default=0.1, type=float,
                        help="dropout rate ")
    parser.add_argument("--weight_decay", default=1e-8, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--embed_size", default=256, type=int,
                        help="The size of word embedding")

    # args = parser.parse_known_args()[0]  #for notebook
    args = parser.parse_args()
    args.max_seq_length = max(args.max_query_length, args.max_doc_length)

    print(args)


    train_data = load_data2list(args.train_clicklog_path, 1, [i for i in range(9)])
    train_data = OurDataset(train_data)

    test_data = load_data2list(args.test_clicklog_path, 1, [i for i in range(9)])
    test_data = OurDataset(test_data)

    dict_geohash = load_geohash(args.geohash_path)
    dict_poi_set = load_data2dict(args.poi_path, 1, 0, [i for i in range(4)])
    list_poi_full = list(dict_poi_set.keys())

    tokenizer = BertTokenizerFast.from_pretrained(args.plm_path)
    geo_special_tokens_dict = ['[' + gcd + ']' for gcd in dict_geohash]
    tokenizer.add_tokens(geo_special_tokens_dict)
    print('get all token ing...')
    try:
        f = open("all_tokens.txt", "r",encoding='utf-8')
        lll = f.readlines()
        all_tokens = []
        for l in lll:
            all_tokens.append(l.replace('\n', ''))
    except:
        all_tokens = get_init_tokens(dict_geohash,train_data,list_poi_full,dict_poi_set,tokenizer)
        str = '\n'
        f=open("all_tokens.txt","w",encoding='utf-8')
        f.write(str.join(all_tokens))
        f.close()
    print('finished get all token...')


    vocab = Vocab(initial_tokens=all_tokens)
    vocab.randomly_init_embeddings(embed_dim=args.embed_size)


    #训练数据集
    pair_data = get_pair_data(train_data,args.rand_neg_num_per,list_poi_full)
    PairDataSet = PairHQCNDataset(max_query_length=args.max_query_length, max_doc_length=args.max_doc_length, dataset=pair_data, vocab=vocab,
                                  dict_poi_set=dict_poi_set, tokenizer=tokenizer, history_num=args.history_num)

    d_k = args.embed_size // args.n_head
    model = HQCN(vocab, d_word_vec=args.embed_size, d_model=args.embed_size, d_inner=args.inner_dim,
                 n_layers=args.n_layers,n_head=args.n_head, d_k=d_k, d_v=d_k, dropout=args.dropout, n_position=args.max_seq_length)


    # Test
    print(len(train_data), len(test_data))
    idx = [i for i in range(len(test_data))]
    #idx = random.sample(idx, 1000)
    test_data = get_test_data(test_data, idx[0:10000])
    TestDataSet = HQCNDataset(max_query_length=args.max_query_length, max_doc_length=args.max_doc_length, dataset=test_data, vocab=vocab,
                              dict_poi_set=dict_poi_set, tokenizer=tokenizer, history_num=args.history_num)

    # Train
    print('train!')
    train(args, PairDataSet, TestDataSet, model, vocab, alpha=0.8)
    '''
    train_dataloader = DataLoader(TestDataSet, batch_size=51)
    for idx_, batch in tqdm(enumerate(train_dataloader)):
        print(idx_)
        print(batch['label'])
        break
    '''