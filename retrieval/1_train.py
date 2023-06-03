import sys, os
import torch
import torch.nn as nn
import csv, random, json
from argparse import ArgumentParser
from tqdm import tqdm
import numpy as np
import pickle, faiss, math
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

def save_checkpoint(checkpoint_path, name, model, ):
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    torch.save(
        {
            'model_state_dict': model.state_dict(),
        }, 
        checkpoint_path + name,
    )

def save_obj(data, path,):
    file = open(path, 'wb')
    pickle.dump(data, file)
    file.close()

def load_obj(path):
    file = open(path, 'rb')
    return pickle.load(file)

def eval_ranking(cutoff_rel_mark, metric):
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
    return result_metric


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


def gumbel_max_sample(
        candi_poids,
        softmax_prob,
        sample_num,
    ):
        assert len(candi_poids) == softmax_prob.shape[0]

        unif = torch.rand(softmax_prob.size()).to(softmax_prob.device)  # [num_samples_per_query, ranking_size]

        EPS = 1e-20
        gumbel = -torch.log(-torch.log(unif + EPS) + EPS) 

        batch_logits = (softmax_prob.log() + gumbel)  # / temperature

        _, batch_indices = torch.sort(batch_logits, dim=0, descending=True)

        batch_indices = list(batch_indices.detach().cpu().numpy())
        
        return batch_indices[0:sample_num]

    
def click_model_simulate(
    softmax_prob,
    action_poids,
):
    obs_prob = torch.tensor([max(0.01, 1 / (i + 1)) for i in range(len(action_poids))]).to(softmax_prob.device)

    click_prob = torch.mul(obs_prob, softmax_prob)

    rand_prob = torch.rand((obs_prob.shape[0],)).to(softmax_prob.device)
    
    click_log = (rand_prob < click_prob) + 0

    return list(click_log.detach().cpu().numpy())


def tokenize_query_action(
    batch_state, batch_action,
    dict_poi_set, tokenizer,
):
    query_action_context, mask_query_action = [], []

    batch_max_action_len = max([len(action) for action in batch_action])

    for bb in range(len(batch_state)):
        bb_sess_q_geohash, bb_current_query = batch_state[bb][0:2]
        bb_current_candi_poids = batch_action[bb]

        for cc in range(len(bb_current_candi_poids)):
            cc_bb_poid = bb_current_candi_poids[cc]
            query_action_context.append(
                geo_spec_tok(bb_sess_q_geohash) + '[SEP]' + \
                bb_current_query + '[SEP]' + \
                geo_spec_tok(dict_poi_set[cc_bb_poid][3]) + '[SEP]' + \
                dict_poi_set[cc_bb_poid][1] + '[SEP]' + \
                dict_poi_set[cc_bb_poid][2]
            )
            mask_query_action.append(1)
        
        for _ in range(len(bb_current_candi_poids), batch_max_action_len):
            query_action_context.append('')
            mask_query_action.append(0)
    
    query_action_context = tokenizer(query_action_context, padding=True, return_tensors='pt')
    for key in ['input_ids', 'attention_mask', 'token_type_ids']:
        query_action_context[key] = query_action_context[key].reshape(len(batch_state), batch_max_action_len, -1)
    mask_query_action = torch.LongTensor(mask_query_action).reshape(len(batch_state), batch_max_action_len,)

    return query_action_context, mask_query_action

    


def get_action_from_state(
    sess_query_geohash, current_query, current_candi_poids,
    # prefer_vec,
    dict_poi_set, tokenizer,
    major_actor,
    device,
    max_or_gumbel,
):
    # current_query_candi_poids_context
    current_query_candi_poids_context = []
    for cpoid in current_candi_poids:
        current_query_candi_poids_context.append(
            geo_spec_tok(sess_query_geohash) + '[SEP]' \
            + current_query + '[SEP]' + \
            geo_spec_tok(dict_poi_set[cpoid][3]) + '[SEP]' + \
            dict_poi_set[cpoid][1] + '[SEP]' + \
            dict_poi_set[cpoid][2]
        )
    
    # tokenize
    current_query_candi_poids_context = tokenizer(current_query_candi_poids_context, padding=True, return_tensors='pt')
    #prefer_vec = torch.FloatTensor(prefer_vec)

    with torch.no_grad():
        # cal scores 
        current_scores_candi = major_actor.module.score_query_poi(
            # token
            current_query_candi_poids_context['input_ids'].to(device), 
            current_query_candi_poids_context['attention_mask'].to(device), 
            current_query_candi_poids_context['token_type_ids'].to(device), 
            # prefer vec
            #prefer_vec.reshape(1, -1).repeat(len(current_candi_poids), 1).to(device),
        ).reshape(-1)

        # action
        if max_or_gumbel == 'gumbel':
            current_action_poids_indices = gumbel_max_sample(
                current_candi_poids, 
                torch.softmax(current_scores_candi, dim=0), 
                args.action_topk,
            )
        
        if max_or_gumbel == 'max':
            _, batch_indices = torch.sort(torch.softmax(current_scores_candi, dim=0), dim=1, descending=True)
            batch_indices = list(batch_indices.detach().cpu().numpy())
            current_action_poids_indices = batch_indices[0:args.action_topk]


        scores_current_action_poids = current_scores_candi[current_action_poids_indices]
        current_action_poids = [current_candi_poids[ind] for ind in current_action_poids_indices]

        return current_action_poids, scores_current_action_poids


def get_reward_vec(
    current_action_poids, scores_current_action_poids,
    current_query, next_reform_query,
    tokenizer
):
    click_feedback = click_model_simulate(
        torch.softmax(scores_current_action_poids, dim=0),
        current_action_poids,
    )
    click_reward = eval_ranking(click_feedback, 'mrr')
    
    reform_effort = calculate_reform(
        tokenizer(current_query)['input_ids'],
        tokenizer(next_reform_query)['input_ids'],
    )
    reform_reward = 2.0 / (1 + math.exp(reform_effort))

    return [click_reward, reform_reward]
    






class OurDataset(torch.utils.data.Dataset):
    def __init__(self, userlog_data):
        self.userlog_data = userlog_data

        for iii in tqdm(range(len(self.userlog_data))):
            # query, geohash, clk_poiid, filter_rec_poi_list_id, sess_time_list, sess_query_list, filter_sess_poilist_list_id, start_poiid, sess_candidate_poilist
            for json_num in [5, 6, 8]:
                self.userlog_data[iii][json_num] = json.loads(self.userlog_data[iii][json_num])
        
    def __len__(self):
        return len(self.userlog_data)

    def __getitem__(self, index):
        return self.userlog_data[index]


def main(args):
    # initalize
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    torch.distributed.init_process_group(backend="nccl")
    torch.distributed.barrier()


    # data prepare
    # poi 
    dict_poi_set = load_data2dict(args.poi_path, 1, 0, [i for i in range(4)])
    print(args.local_rank, len(dict_poi_set))
    # goehash d
    dict_geohash = load_geohash(args.geohash_path)
    # query, geohash, clk_poiid, filter_rec_poi_list_id, sess_time_list, sess_query_list, filter_sess_poilist_list_id, start_poiid, sess_candidate_poilist
    train_data = load_data2list(args.train_clicklog_path, 1, [i for i in range(9)])
    print(args.local_rank, len(train_data))
    # dataset
    train_data = OurDataset(train_data)
    
    torch.distributed.barrier()
    
    # ----- all train data
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
    train_dataloader = torch.utils.data.DataLoader(
        train_data, 
        sampler=train_sampler, 
        batch_size=1,
        num_workers=4, 
        pin_memory=True,
        collate_fn=lambda x:x,
    )

    torch.distributed.barrier()

    # tokenizer
    tokenizer = BertTokenizerFast.from_pretrained(args.plm_path)
    print(args.local_rank, len(tokenizer))
    geo_special_tokens_dict = ['['+gcd+']' for gcd in dict_geohash]
    tokenizer.add_tokens(geo_special_tokens_dict)
    print(args.local_rank, len(tokenizer))

    torch.distributed.barrier()


    
    # model: actor-critic initialize
    ### major actor
    major_actor = model.Our_actor.from_pretrained(args.plm_path, args)
    major_actor.resize_token_embeddings(len(tokenizer))
    load_checkpoint(args.load_checkpoint_path, major_actor)
    major_actor = major_actor.to(device)
    major_actor.train()
    major_actor = nn.parallel.DistributedDataParallel(major_actor, device_ids=[args.local_rank], output_device=args.local_rank,)
    ### target actor
    target_actor = model.Our_actor.from_pretrained(args.plm_path, args)
    target_actor.resize_token_embeddings(len(tokenizer))
    load_checkpoint(args.load_checkpoint_path, target_actor)
    target_actor = target_actor.to(device)
    target_actor.train()
    target_actor = nn.parallel.DistributedDataParallel(target_actor, device_ids=[args.local_rank], output_device=args.local_rank,)
    ### major critic
    major_critic = model.Our_critic.from_pretrained(args.plm_path, args)
    major_critic.resize_token_embeddings(len(tokenizer))
    major_critic = major_critic.to(device)
    major_critic.train()
    major_critic = nn.parallel.DistributedDataParallel(major_critic, device_ids=[args.local_rank], output_device=args.local_rank,)
    ### target critic
    target_critic = model.Our_critic.from_pretrained(args.plm_path, args)
    target_critic.resize_token_embeddings(len(tokenizer))
    target_critic = target_critic.to(device)
    target_critic.train()
    target_critic = nn.parallel.DistributedDataParallel(target_critic, device_ids=[args.local_rank], output_device=args.local_rank,)
    
    
    # optimizer
    optimizer_actor = AdamW(filter(lambda p: p.requires_grad, major_actor.parameters()), lr=args.learning_rate, eps=1e-8)
    optimizer_critic = AdamW(filter(lambda p: p.requires_grad, major_critic.parameters()), lr=args.learning_rate, eps=1e-8)
    
    scaler = torch.cuda.amp.GradScaler() if args.mix_amp == 1 else None
    
    
    
    autocast = torch.cuda.amp.autocast if args.mix_amp == 1 else suppress

    torch.distributed.barrier()
    

    # replay_buffer
    replay_buffer = [] # (state, action, reward, next state)
    critic_criterion  = nn.MSELoss()

    training_batch_num = 0

    # training
    for epoch in range(args.epoch):
        # shuffle
        train_sampler.set_epoch(epoch)
        
        # save
        torch.distributed.barrier()

        '''
        if args.local_rank == 0 and training_batch_num % 100 == 0:
            print('save checkpoint, epoch {}, rank {}'.format(epoch, args.local_rank))
            save_checkpoint(args.save_checkpoint_path, '/major_actor-v' + str(epoch) + '-' + str(training_batch_num) + '.ck', major_actor.module, )
            save_checkpoint(args.save_checkpoint_path, '/major_critic-v' + str(epoch) + '-' + str(training_batch_num) + '.ck', major_critic.module, )
        '''
        

        for idx_,  batch_ in tqdm(enumerate(train_dataloader)):
            assert args.local_rank == torch.distributed.get_rank()
            
            batch_len = len(batch_)
            
            # step action for each record in batch
            for x in range(batch_len):
                x_q_geohash = batch_[x][1]
                x_clk_poiid = batch_[x][2]
                
                x_sess_query = batch_[x][5]
                x_sess_candi_poids = batch_[x][8]

                assert len(x_sess_query) == len(x_sess_candi_poids)

                # step
                #prefer_ = random.random()
                #prefer_vec = [prefer_, 1 - prefer_]

                for step in range(len(x_sess_query)):
                    step_x_query = x_sess_query[step]
                    step_x_candi_poids = x_sess_candi_poids[step][0:args.cutoff]
                    if x_clk_poiid not in step_x_candi_poids:
                        step_x_candi_poids += [x_clk_poiid]
                        
                    next_step_x_reform_query = x_sess_query[min(step + 1, len(x_sess_query) - 1)]
                    next_step_x_candi_poids = x_sess_candi_poids[min(step + 1, len(x_sess_query) - 1)][0:args.cutoff]

                    instance_state = [x_q_geohash, step_x_query, step_x_candi_poids]
                    instance_action, instance_action_score = get_action_from_state(
                        x_q_geohash, step_x_query, step_x_candi_poids,
                        #prefer_vec,
                        dict_poi_set, tokenizer,
                        major_actor,
                        device,
                        'gumbel',
                    )
                    instance_reward_vec = get_reward_vec(
                        instance_action, instance_action_score,
                        step_x_query, next_step_x_reform_query,
                        tokenizer
                    )
                    instance_next_state = [x_q_geohash, next_step_x_reform_query, next_step_x_candi_poids]
                    
                    replay_buffer.append([instance_state, instance_action, instance_reward_vec, instance_next_state, ])#prefer_vec])


            # update with replay_buffer
            if len(replay_buffer) >= args.batch_size:
                batch_replay = random.sample(replay_buffer, args.batch_size)
                #training_batch_num += 1

                batch_state = []
                batch_action = []
                batch_reward_vec = []
                batch_next_state = []
                #batch_prefer_vec = []

                batch_major_policy_action = []
                batch_target_policy_next_action = []

                # batch
                for memory in batch_replay:
                    m_state = memory[0] # [sess_query_geohash, current_query, current_candi_poids]
                    m_action = memory[1] # current_action_poids
                    m_reward_vec = memory[2] # [click_reward, reform_reward]
                    m_next_state = memory[3] # [sess_query_geohash, next_reform_query, next_candi_poids]
                    #m_prefer_vec = memory[4]

                    batch_state.append(m_state)
                    batch_action.append(m_action)
                    batch_reward_vec.append(m_reward_vec)
                    batch_next_state.append(m_next_state)
                    #batch_prefer_vec.append(m_prefer_vec)

                    m_major_policy_action, _ = get_action_from_state(
                        m_state[0], m_state[1], m_state[2],
                        #m_prefer_vec,
                        dict_poi_set, tokenizer,
                        major_actor,
                        device,
                        'gumbel',
                    )

                    m_target_policy_next_action, _ = get_action_from_state(
                        m_next_state[0], m_next_state[1], m_next_state[2],
                        #m_prefer_vec,
                        dict_poi_set, tokenizer,
                        target_actor,
                        device,
                        'gumbel',
                    )

                    batch_major_policy_action.append(m_major_policy_action)
                    batch_target_policy_next_action.append(m_target_policy_next_action)


                
                # tokenize
                batch_query_action_context, batch_mask_query_action = tokenize_query_action(
                    batch_state, batch_action,
                    dict_poi_set, tokenizer,
                )
                batch_next_query_action_context, batch_next_mask_query_action = tokenize_query_action(
                    batch_next_state, batch_target_policy_next_action,
                    dict_poi_set, tokenizer,
                )
                batch_query_action_context_policy, batch_mask_query_action_policy = tokenize_query_action(
                    batch_state, batch_major_policy_action,
                    dict_poi_set, tokenizer,
                )

                with autocast():
                    # Critic loss
                    batch_Qvals = major_critic.module.Q_query_action(
                        batch_query_action_context['input_ids'].to(device), 
                        batch_query_action_context['attention_mask'].to(device), 
                        batch_mask_query_action.to(device),
                        #torch.FloatTensor(batch_prefer_vec).to(device),
                    )
                    with torch.no_grad():
                        batch_next_Q = target_critic.module.Q_query_action(
                            batch_next_query_action_context['input_ids'].to(device), 
                            batch_next_query_action_context['attention_mask'].to(device), 
                            batch_next_mask_query_action.to(device),
                            #torch.FloatTensor(batch_prefer_vec).to(device),
                        )
                    batch_Qprime = torch.FloatTensor(batch_reward_vec).to(device) + args.gamma * batch_next_Q

                    critic_loss = critic_criterion(batch_Qvals, batch_Qprime)
                    

                    # Actor loss
                    policy_loss = - major_critic.module.Q_query_action(
                        batch_query_action_context_policy['input_ids'].to(device), 
                        batch_query_action_context_policy['attention_mask'].to(device), 
                        batch_mask_query_action_policy.to(device),
                        #torch.FloatTensor(batch_prefer_vec).to(device),
                    )
                    
                    batch_prefer = torch.FloatTensor([args.prefer, 1 - args.prefer]).to(device)
                    batch_prefer = batch_prefer.reshape(1, -1).repeat(policy_loss.shape[0], 1)
                    policy_loss = torch.mul(policy_loss, batch_prefer).sum(1)

                    policy_loss = policy_loss.mean()

                # backward
                if scaler is not None:
                    print('critic_loss', critic_loss.item())
                    print('policy_loss', policy_loss.item())

                    # update critic
                    optimizer_critic.zero_grad()
                    scaler.scale(critic_loss).backward()
                    scaler.unscale_(optimizer_critic)
                    torch.nn.utils.clip_grad_norm_(major_critic.parameters(), 2.0)
                    scaler.step(optimizer_critic)  
                    scaler.update()

                    # update actor
                    optimizer_actor.zero_grad()
                    scaler.scale(policy_loss).backward()
                    scaler.unscale_(optimizer_actor)
                    torch.nn.utils.clip_grad_norm_(major_actor.parameters(), 2.0)
                    scaler.step(optimizer_actor)  
                    scaler.update()
                else:
                    # update critic
                    optimizer_critic.zero_grad()
                    critic_loss.backward()
                    torch.nn.utils.clip_grad_norm_(major_critic.parameters(), 2.0)
                    optimizer_critic.step()

                    # update actor
                    optimizer_actor.zero_grad()
                    policy_loss.backward()
                    torch.nn.utils.clip_grad_norm_(major_actor.parameters(), 2.0)
                    optimizer_actor.step()

                # update target networks
                with torch.no_grad():
                    for target_param, param in zip(target_actor.parameters(), major_actor.parameters()):
                        target_param.data.copy_(param.data * args.tau + target_param.data * (1.0 - args.tau))
            
                    for target_param, param in zip(target_critic.parameters(), major_critic.parameters()):
                        target_param.data.copy_(param.data * args.tau + target_param.data * (1.0 - args.tau))


                # print
                if (idx_ + 1) % 2000 == 0 and args.local_rank == 0:
                    print(epoch, args.local_rank, (idx_ + 1) * args.batch_size, )
                    print('rank:{}, policy loss:{}, critic_loss:{}, '.format(args.local_rank, policy_loss.item(), critic_loss.item(),))

                    print('save checkpoint, epoch {}, rank {}'.format(epoch, args.local_rank))
                    save_checkpoint(args.save_checkpoint_path, '/major_actor-v' + str(epoch) + '-' + str(idx_) + '.ck', major_actor.module, )
                    save_checkpoint(args.save_checkpoint_path, '/major_critic-v' + str(epoch) + '-' + str(idx_) + '.ck', major_critic.module, )



            

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

    parser.add_argument('--load_checkpoint_path', type=str, default='/nfs/volume-93-2/meilang/myprojects/poi_contextual_rerank/model/click_query_bert_rerank/checkpoint/model-9.ck')

    
    # run
    # learning rate
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    # batch size
    parser.add_argument('--batch_size', type=int, default=8)
    # epoch
    parser.add_argument('--epoch', type=int, default=7)
    # action topk
    parser.add_argument('--action_topk', type=int, default=10)
    # cutoff topk
    parser.add_argument('--cutoff', type=int, default=20)
    # gamma in Rl
    parser.add_argument('--gamma', type=float, default=0.9)
    # tau in Rl
    parser.add_argument('--tau', type=float, default=0.01)
    # prefer
    parser.add_argument('--prefer', type=float, default=0.5)
    
    

    
    args = parser.parse_args()
    print(args)

    
    # main
    main(args)