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
from torch_scatter import scatter_max, segment_csr, scatter_max
from torch_scatter import scatter

from typing import Callable


class bert_rerank(BertForPreTraining):
    def __init__(self, config, data_args, ):

        super().__init__(config)

        #print("config:\n", config)

        self.args = data_args
        self.criterion = nn.BCELoss()

        self.mlp_ = nn.Linear(config.hidden_size, 1)
    
        # weight initialized
        self.init_weights()


    def score(self, 
        input_ids_context, 
        attention_mask_context, 
        token_type_ids_context,
    ):
        score = self.mlp_(
            self.bert(
                input_ids=input_ids_context, 
                attention_mask=attention_mask_context, 
                token_type_ids=token_type_ids_context,
            )[1]
        )

        return score

    '''
    def score_pair(self, Q, D):
        return torch.mul(Q, D).sum(1)
    
    def score_inbatch_neg(self, Q, D):
        assert Q.shape == D.shape
        bsize = Q.shape[0]
        return (torch.matmul(Q, D.T)).flatten()[1: ].view(bsize - 1, bsize + 1)[: ,: -1].reshape(bsize, bsize - 1) 
    
    def score_all(self, Q, D):
        return torch.matmul(Q, D.T) 

    
    def cls_(self, 
        input_ids_context, 
        attention_mask_context, 
        token_type_ids_context,
    ):
        # text feature
        Text_emb = self.bert(
            input_ids=input_ids_context, 
            attention_mask=attention_mask_context,
            token_type_ids=token_type_ids_context,
        )[1]
        # cosine sim
        Text_emb = F.normalize(Text_emb, p=2, dim=1)
        
        return Text_emb
    
    
    def __init__(self, config, data_args, tokenizer,):
        
        
        
        
        super().__init__(config)
        
        self.data_args = data_args
        self.tokenizer_size = len(tokenizer)

        self.score_mlp = nn.Sequential(
            nn.Linear(config.hidden_size, 1),
        )
        self.BCELoss = nn.BCELoss()

        # weight initialized
        self.init_weights()
        
    def forward(
        # basic
        self, labels, input_ids, 
        input_ids_request_poi, attention_mask_request_poi, token_type_ids_request_poi,
        labels_request_poi,
    ):
        batch_size, med_size, last_size = input_ids_request_poi.shape
        scores_request_poi = self.score_mlp(
            self.bert(
                input_ids=input_ids_request_poi.reshape(-1, last_size), 
                attention_mask=attention_mask_request_poi.reshape(-1, last_size), 
                token_type_ids=token_type_ids_request_poi.reshape(-1, last_size),
            )[1]
        )

        scores_request_poi = torch.sigmoid(scores_request_poi.reshape(-1))
        
        loss_request_poi = self.BCELoss(scores_request_poi, labels_request_poi.reshape(-1))

        return BertForPreTrainingOutput(
            loss=loss_request_poi,
        )
    '''