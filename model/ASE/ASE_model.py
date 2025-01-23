import torch
import torch.nn as nn
import torch.nn.init as init
from transformers.models.bart.modeling_bart import BartClassificationHead

class ASEModel(nn.Module):
    def __init__(self, bart, tokenizer):
        super(ASEModel, self).__init__()

        # bart model
        self.model = bart
        self.tokenizer = tokenizer

        # uncertainty parameters
        self.loss_weights = nn.Parameter(torch.ones(4))

        # classification head provided by huggingface.
        # (input_dim, inner_dim, num_classes, dropout)
        self.classification_head = BartClassificationHead(
            768, 768, 1, 0.1
        )
        init.xavier_normal_(self.classification_head.dense.weight)
        init.xavier_normal_(self.classification_head.out_proj.weight)

        self.classifier = nn.Linear(768, 1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(0.1)

    def generation_forward(self, encoder_input_ids, encode_attention_mask, labels):
        
        # obtain generation loss
        # bart will automatically shift the labels to get decoder_input_ids

        bart_inputs = {'input_ids': encoder_input_ids, 'attention_mask': encode_attention_mask, 
        'output_hidden_states': True, 'labels': labels}
        bart_outputs = self.model(**bart_inputs)
        
        loss = bart_outputs.loss

        return loss

    def weighted_loss(self, loss, index):
        # obtain uncertainty loss
        return (loss / (self.loss_weights[index] * 2)) + (self.loss_weights[index] + 1).log()
    
    def hinge_loss(self, scores, margin, mask):
        # obtain hinge ranking loss
        loss = torch.nn.functional.relu(margin - (torch.unsqueeze(scores[:, 0], -1) - scores[:, 1:]) * mask)
        return torch.mean(loss)

    def forward(self, batch_data, is_test=False):

        if is_test:
            # only use encoder while inferencing.
            encoder_input_ids_rank = batch_data["encoder_input_ids_rank"]
            encode_attention_mask_rank = batch_data["encode_attention_mask_rank"]
            eos_position = batch_data["eos_position"]

            bart_inputs = {'input_ids': encoder_input_ids_rank, 'attention_mask': encode_attention_mask_rank,
            'output_hidden_states': True}

            bart_outputs = self.model.model(**bart_inputs)
            encoder_hidden = bart_outputs.encoder_last_hidden_state

            # the ouput of [CLS].
            classification_head_token = (eos_position == 1)

            # go through MLP.
            eos_hidden = encoder_hidden[classification_head_token,:]
            y_pred = self.classification_head(eos_hidden).squeeze(1)

            return y_pred
        else:
            scores = []
            gen_losses = []
            loss_mask = None
            for batch in batch_data:
                encoder_input_ids_rank = batch["encoder_input_ids_rank"]
                encode_attention_mask_rank = batch["encode_attention_mask_rank"]
                encoder_input_ids_gen_fq = batch["encoder_input_ids_gen_fq"]
                encoder_input_ids_gen_cd = batch["encoder_input_ids_gen_cd"]
                encoder_input_ids_gen_sq = batch["encoder_input_ids_gen_sq"]
                eos_position = batch["eos_position"]
                next_q_labels = batch["next_q_labels"]
                click_doc_labels = batch["click_doc_labels"]
                previous_q_labels = batch["previous_q_labels"]
                simq_labels = batch["simq_labels"]
                loss_mask = batch["loss_mask"]

                # Ranking

                bart_inputs = {'input_ids': encoder_input_ids_rank, 'attention_mask': encode_attention_mask_rank, 
                'output_hidden_states': True}

                bart_outputs = self.model.model(**bart_inputs)
                encoder_hidden = bart_outputs.encoder_last_hidden_state

                # the ouput of [CLS].
                classification_head_token = (eos_position == 1)

                eos_hidden = encoder_hidden[classification_head_token,:]
                y_pred = self.classification_head(eos_hidden)
                scores.append(y_pred)

                # Generation Losses of three tasks.

                gen_loss1 = self.generation_forward(encoder_input_ids_gen_fq, encode_attention_mask_rank, next_q_labels)
                gen_loss2 = self.generation_forward(encoder_input_ids_gen_cd, encode_attention_mask_rank, click_doc_labels)
                gen_loss3 = self.generation_forward(encoder_input_ids_gen_sq, encode_attention_mask_rank, simq_labels)

                gen_loss = self.weighted_loss(gen_loss1, 1) + self.weighted_loss(gen_loss2, 2) + self.weighted_loss(gen_loss3, 3)

                gen_losses.append(gen_loss)
            
            # Ranking Loss

            batch_scores = torch.cat(scores, dim = -1)
            ranking_loss = self.hinge_loss(batch_scores, 1, loss_mask)
            w_ranking_loss = self.weighted_loss(ranking_loss, 0)

            return sum(gen_losses)/len(gen_losses), w_ranking_loss
