from distutils.command.config import config
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from transformers import (
    BertForPreTraining, 
    PreTrainedModel,
    BertModel, 
    BertGenerationEncoder, 
    BertGenerationDecoder, 
    EncoderDecoderModel,
    EncoderDecoderConfig,
)

from transformers.models.bert.modeling_bert import BertForPreTrainingOutput

import torch.nn.functional as F

from typing import Callable


class Our_actor(BertForPreTraining):
    def __init__(self, config, data_args, ):

        super().__init__(config)

        self.args = data_args
        self.criterion = nn.BCELoss()

        #self.mlp_score = nn.Linear(config.hidden_size + 2, 1)

        self.mlp_ = nn.Linear(config.hidden_size, 1)
    
        # weight initialized
        self.init_weights()

    
    def score_query_poi(self, 
        # token [batch, ]
        input_ids_context, attention_mask_context, token_type_ids_context,
        # prefer vec [batch, ]
        # prefer_vec,
    ):
        # [batch, dim]
        context_feature = self.bert(
            input_ids=input_ids_context, 
            attention_mask=attention_mask_context, 
            token_type_ids=token_type_ids_context,
        )[1]

        # [batch, 2]
        '''
        prefer_context_feature = torch.cat(
            [context_feature, prefer_vec],
            dim=1,
        )
        '''

        # score
        score_query_poi = self.mlp_(context_feature)#prefer_context_feature)

        return score_query_poi

    
    
class Our_critic(BertForPreTraining):
    def __init__(self, config, data_args, ):

        super().__init__(config)

        self.args = data_args
        self.criterion = nn.BCELoss()

        #self.mlp_Q = nn.Linear(config.hidden_size + 2, 2)
        self.mlp_Q = nn.Linear(config.hidden_size, 2)
        
    
        # weight initialized
        self.init_weights()

    def Q_query_action(
        self, 
        # history [batch, batch_max_action_len, max_seq]
        input_ids_query_action, attention_mask_query_action, 
        # history rec browser mask [batch, batch_max_action_len]
        mask_query_action,
        # prefer vec
        #prefer_vec,
    ):
        b_size, batch_max_action_len, _ = input_ids_query_action.shape
            
        # session + action tok [CLS] emb;
        ### [batch * batch_max_action_len, hid_dim]
        query_action_qp_emb = self.bert(
            input_ids=input_ids_query_action.reshape(b_size * batch_max_action_len, -1), 
            attention_mask=attention_mask_query_action.reshape(b_size * batch_max_action_len, -1), 
        )[1]
        ### [batch, batch_max_action_len, hid_dim]
        query_action_qp_emb = query_action_qp_emb.reshape(b_size, batch_max_action_len, -1) 


        cls_tok = (101 * torch.ones((b_size, 1))).long().to(query_action_qp_emb.device)
        cls_emb = self.bert.embeddings.word_embeddings(cls_tok) 
        cls_mask = torch.ones((b_size, 1)).long().to(cls_emb.device)


        cls_query_action_qp_emb = torch.cat(
            [cls_emb, query_action_qp_emb], 
            dim=1,
        )
        cls_query_action_attention_mask = torch.cat(
            [cls_mask, mask_query_action], 
            dim=1,
        )

        # Q-value
        ### [batch, dim]
        cls_query_action_feature = self.bert(
            inputs_embeds=cls_query_action_qp_emb, 
            attention_mask=cls_query_action_attention_mask, 
        )[1]

        # [batch, 2]
        '''
        prefer_cls_query_action_feature = torch.cat(
            [cls_query_action_feature, prefer_vec],
            dim=1,
        )
        '''

        # Q-value
        Q_query_action = self.mlp_Q(cls_query_action_feature)#prefer_cls_query_action_feature)

        return Q_query_action
            
    
    

            