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


class Bert_geo(BertForPreTraining):
    def __init__(self, config, data_args, ):

        super().__init__(config)

        #print("config:\n", config)

        self.args = data_args
        self.criterion = nn.CrossEntropyLoss()

        # weight initialized
        self.init_weights()

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

