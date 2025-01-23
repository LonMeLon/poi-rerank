import linecache
from torch.utils.data import Dataset
import numpy as np
import copy

# Test dataset for AOL
class TestFileDataset(Dataset):
    def __init__(self, filename, tokenizer):
        super(TestFileDataset, self).__init__()
        self._filename = filename
        self._max_seq_length = 128
        self._tokenizer = tokenizer
        with open(filename, "r",encoding='utf8') as f:
            self._total_data = len(f.readlines())
    
    def check_length(self, pairlist):
        '''
        Make sure the current session sequence has even numbers of behaviors.
        '''
        assert len(pairlist) % 2 == 0
        max_seq_length = self._max_seq_length - 3
        if len(pairlist) == 2:
            while len(pairlist[0]) + len(pairlist[1]) + 2 > max_seq_length:
                if len(pairlist[0]) > len(pairlist[1]):
                    pairlist[0].pop(0)
                else:
                    pairlist[1].pop(-1)
        else:
            q_d_minimum_length = 0
            for i in range(len(pairlist)):
                q_d_minimum_length += len(pairlist[i]) + 1
            if q_d_minimum_length > max_seq_length:
                pairlist.pop(0)
                pairlist.pop(0)
                pairlist = self.check_length(pairlist)
        return pairlist
    
    def anno_encoder(self, all_qd_toks):
        '''
        Get input ids of encoder.
        '''

        # position of the [CLS] token
        eos_position = 0

        encode_attention_mask = [1] * len(all_qd_toks)
        assert len(all_qd_toks) <= self._max_seq_length
        while len(all_qd_toks) < self._max_seq_length:
            all_qd_toks.append("<pad>")
            encode_attention_mask.append(0)
        assert len(all_qd_toks) == len(encode_attention_mask) == self._max_seq_length
        anno_seq = self._tokenizer.convert_tokens_to_ids(all_qd_toks)

        eos_mask = [0] * len(all_qd_toks)
        eos_mask[eos_position] = 1
        encoder_input_ids = np.asarray(anno_seq)
        encode_attention_mask = np.asarray(encode_attention_mask)

        return encoder_input_ids, encode_attention_mask, eos_mask

    def anno_main(self, qd_pairs):
        all_qd = []
        for i in range(len(qd_pairs)):
            qd = qd_pairs[i]
            qd = self._tokenizer.tokenize(qd)
            qd += ["[eos]"]
            all_qd.append(qd)
        all_qd = self.check_length(all_qd)

        history = all_qd[:-2]
        query_tok = all_qd[-2]
        doc_tok = all_qd[-1]
        
        history_toks = []
        for iidx, sent in enumerate(history):
            history_toks.extend(sent)
        
        query_tok += ["</s>"]
        doc_tok += ["</s>"]

        # encoder input tokens
        all_qd_toks_rank = ["<s>"] + ["[rank]"] + history_toks + query_tok + doc_tok

        # encoder input

        encoder_input_ids_rank, encode_attention_mask_rank, eos_mask = self.anno_encoder(all_qd_toks_rank)

        return encoder_input_ids_rank, encode_attention_mask_rank, eos_mask
    
    def __getitem__(self, idx):
        line = linecache.getline(self._filename, idx + 1)
        line = line.strip().split("\t")
        label = int(line[0])
        next_q = line[1]
        click_doc = line[2]
        first_q = line[3]
        qd_pairs = line[4:]

        encoder_input_ids_rank, encode_attention_mask_rank, eos_position = self.anno_main(qd_pairs)

        # only use encoder while inferencing.
        batch = {
            'encoder_input_ids_rank': np.asarray(encoder_input_ids_rank), 
            'encode_attention_mask_rank': np.asarray(encode_attention_mask_rank),
            'eos_position': np.asarray(eos_position, dtype=np.int64),
            'ranking_labels': float(label),
        }

        return batch
    
    def __len__(self):
        return self._total_data

# Test dataset for Tiangong-ST
class TgTestFileDataset(Dataset):
    def __init__(self, filename, tokenizer):
        super(TgTestFileDataset, self).__init__()
        self._filename = filename
        self._max_seq_length = 128
        self._tokenizer = tokenizer
        with open(filename, "r") as f:
            self._total_data = len(f.readlines())
    
    def check_length(self, pairlist):
        '''
        Make sure the current session sequence has even numbers of behaviors.
        '''
        assert len(pairlist) % 2 == 0
        max_seq_length = self._max_seq_length - 3
        if len(pairlist) == 2:
            while len(pairlist[0]) + len(pairlist[1]) + 2 > max_seq_length:
                if len(pairlist[0]) > len(pairlist[1]):
                    pairlist[0].pop(0)
                else:
                    pairlist[1].pop(-1)
        else:
            q_d_minimum_length = 0
            for i in range(len(pairlist)):
                q_d_minimum_length += len(pairlist[i]) + 1
            if q_d_minimum_length > max_seq_length:
                pairlist.pop(0)
                pairlist.pop(0)
                pairlist = self.check_length(pairlist)
        return pairlist
    
    def anno_encoder(self, all_qd_toks):
        '''
        Get input ids of encoder.
        '''

        # position of the [CLS] token
        eos_position = 0

        encode_attention_mask = [1] * len(all_qd_toks)
        assert len(all_qd_toks) <= self._max_seq_length
        while len(all_qd_toks) < self._max_seq_length:
            all_qd_toks.append("[PAD]")
            encode_attention_mask.append(0)
        assert len(all_qd_toks) == len(encode_attention_mask) == self._max_seq_length
        anno_seq = self._tokenizer.convert_tokens_to_ids(all_qd_toks)

        eos_mask = [0] * len(all_qd_toks)
        eos_mask[eos_position] = 1
        encoder_input_ids = np.asarray(anno_seq)
        encode_attention_mask = np.asarray(encode_attention_mask)

        return encoder_input_ids, encode_attention_mask, eos_mask

    def anno_main(self, qd_pairs):
        all_qd = []
        for i in range(len(qd_pairs)):
            qd = qd_pairs[i]
            qd = self._tokenizer.tokenize(qd)
            qd += ["[eos]"]
            all_qd.append(qd)
        all_qd = self.check_length(all_qd)

        history = all_qd[:-2]
        query_tok = all_qd[-2]
        doc_tok = all_qd[-1]
        
        history_toks = []
        for iidx, sent in enumerate(history):
            history_toks.extend(sent)
        
        query_tok += ["[SEP]"]
        doc_tok += ["[SEP]"]

        all_qd_toks_rank = ["[CLS]"] + ["[rank]"] + history_toks + query_tok + doc_tok

        # encoder input

        encoder_input_ids_rank, encode_attention_mask_rank, eos_mask = self.anno_encoder(all_qd_toks_rank)

        return encoder_input_ids_rank, encode_attention_mask_rank, eos_mask
    
    def __getitem__(self, idx):
        line = linecache.getline(self._filename, idx + 1)
        line = line.strip().split("\t")
        label = int(line[0])
        next_q = line[1]
        click_doc = line[2]
        first_q = line[3]
        qd_pairs = line[4:]

        encoder_input_ids_rank, encode_attention_mask_rank, eos_position = self.anno_main(qd_pairs)

        batch = {
            'encoder_input_ids_rank': np.asarray(encoder_input_ids_rank), 
            'encode_attention_mask_rank': np.asarray(encode_attention_mask_rank),
            'eos_position': np.asarray(eos_position, dtype=np.int64),
            'ranking_labels': float(label),
        }

        return batch
    
    def __len__(self):
        return self._total_data

# Training dataset for AOL
class TrainFileDataset(Dataset):
    def __init__(self, filename, tokenizer, dataset):
        super(TrainFileDataset, self).__init__()
        self._filename = filename
        self._max_seq_length = 128
        self._max_decoder_length = 60
        self._tokenizer = tokenizer
        self._dataset = dataset
        with open(filename, "r",encoding='utf8') as f:
            self._total_data = len(f.readlines())
    
    def check_length(self, pairlist):
        if len(pairlist) % 2 != 0:
            print(pairlist)
        assert len(pairlist) % 2 == 0
        max_seq_length = self._max_seq_length - 3
        if len(pairlist) == 2:
            while len(pairlist[0]) + len(pairlist[1]) + 2 > max_seq_length:
                if len(pairlist[0]) > len(pairlist[1]):
                    pairlist[0].pop(0)
                else:
                    pairlist[1].pop(-1)
        else:
            q_d_minimum_length = 0
            for i in range(len(pairlist)):
                q_d_minimum_length += len(pairlist[i]) + 1
            if q_d_minimum_length > max_seq_length:
                pairlist.pop(0)
                pairlist.pop(0)
                pairlist = self.check_length(pairlist)
        return pairlist
    
    def anno_encoder(self, all_qd_toks):
        '''
        Get input ids of encoder.
        '''

        # position of the [CLS] token
        eos_position = 0

        encode_attention_mask = [1] * len(all_qd_toks)
        assert len(all_qd_toks) <= self._max_seq_length
        while len(all_qd_toks) < self._max_seq_length:
            all_qd_toks.append("<pad>")
            encode_attention_mask.append(0)
        assert len(all_qd_toks) == len(encode_attention_mask) == self._max_seq_length
        anno_seq = self._tokenizer.convert_tokens_to_ids(all_qd_toks)

        eos_mask = [0] * len(all_qd_toks)
        eos_mask[eos_position] = 1
        encoder_input_ids = np.asarray(anno_seq)
        encode_attention_mask = np.asarray(encode_attention_mask)

        return encoder_input_ids, encode_attention_mask, eos_mask

    def anno_main(self, qd_pairs):
        all_qd = []
        for i in range(len(qd_pairs)):
            qd = qd_pairs[i]
            qd = self._tokenizer.tokenize(qd)
            qd += ["[eos]"]
            all_qd.append(qd)
        all_qd = self.check_length(all_qd)

        history = all_qd[:-2]
        query_tok = all_qd[-2]
        doc_tok = all_qd[-1]
        
        history_toks = []
        for iidx, sent in enumerate(history):
            history_toks.extend(sent)
        
        query_tok += ["</s>"]
        doc_tok += ["</s>"]

        all_qd_toks_rank = ["<s>"] + ["[rank]"] + history_toks + query_tok + doc_tok

        # encoder input

        encoder_input_ids_rank, encode_attention_mask_rank, eos_mask = self.anno_encoder(all_qd_toks_rank)

        return encoder_input_ids_rank, encode_attention_mask_rank, eos_mask
    
    def __getitem__(self, idx):
        line = linecache.getline(self._filename, idx + 1)
        label, num_neg, click_doc, next_q, previous_q, qd_pairs, docs, simq = split_3gen_2w(line)

        if len(docs) < num_neg+1:
            print(len(docs), docs, num_neg)
        assert len(docs) >= num_neg+1

        batches = []

        # gen labels

        next_q_labels, click_doc_labels, previous_q_labels, simq_labels = self.get_sep_gen_labels(next_q, click_doc, previous_q, simq)
        
        # batches

        for i in range(len(docs)):
            doc = docs[i]
            qds = qd_pairs + [doc]

            encoder_input_ids_rank, encode_attention_mask_rank, eos_position = self.anno_main(qds)

            # Task identifiers for different tasks.
            
            encoder_input_ids_rank[1] = self._tokenizer("[genfq]")['input_ids'][1]
            encoder_input_ids_gen_fq = copy.deepcopy(encoder_input_ids_rank)

            encoder_input_ids_rank[1] = self._tokenizer("[rank]")['input_ids'][1]
            encoder_input_ids_rank1 = copy.deepcopy(encoder_input_ids_rank)
            
            encoder_input_ids_rank[1] = self._tokenizer("[gencd]")['input_ids'][1]
            encoder_input_ids_gen_cd = copy.deepcopy(encoder_input_ids_rank)
            
            encoder_input_ids_rank[1] = self._tokenizer("[gensq]")['input_ids'][1]
            encoder_input_ids_gen_sq = copy.deepcopy(encoder_input_ids_rank)

            batch = {
                'encoder_input_ids_gen_fq': np.asarray(encoder_input_ids_gen_fq), 
                'encoder_input_ids_gen_cd': np.asarray(encoder_input_ids_gen_cd), 
                'encoder_input_ids_gen_sq': np.asarray(encoder_input_ids_gen_sq), 
                'encoder_input_ids_rank': np.asarray(encoder_input_ids_rank1), 
                'encode_attention_mask_rank': np.asarray(encode_attention_mask_rank),
                'next_q_labels': np.asarray(next_q_labels, dtype=np.int64), 
                'click_doc_labels': np.asarray(click_doc_labels, dtype=np.int64), 
                'previous_q_labels': np.asarray(previous_q_labels, dtype=np.int64),
                'simq_labels': np.asarray(simq_labels, dtype=np.int64),
                'eos_position': np.asarray(eos_position, dtype=np.int64),
            }
            batches.append(batch)
        
        if self._dataset == 'aol' or self._dataset == 'poi':
            batches, loss_mask = self.pad_aol_batches(batches, num_neg)
            assert len(batches) == 5
            assert len(loss_mask) == 4
        elif self._dataset == 'tiangong':
            batches, loss_mask = self.pad_tg_atches(batches, num_neg)
            assert len(batches) == 10
            assert len(loss_mask) == 9

        for batch in batches:
            batch["loss_mask"] = np.asarray(loss_mask, dtype=np.int64)

        return batches

    def pad_aol_batches(self, batches, num_neg):
        if len(batches) == 5:
            return batches, [1]*4
        else:
            mask = [1]*(len(batches)-1) + [0]*(4-num_neg)
            for i in range(5-len(batches)):
                batches.append(
                    batches[-1]
                )
            return batches, mask
    
    def pad_tg_atches(self, batches, num_neg):
        if len(batches) == 10:
            return batches, [1]*9
        else:
            mask = [1]*(len(batches)-1) + [0]*(9-num_neg)
            for i in range(10-len(batches)):
                batches.append(
                    batches[-1]
                )
            return batches, mask
    
    def get_sep_gen_labels(self, next_q, click_doc, previous_q, simq):
        if len(next_q) == 1:
            next_q_encode = self._tokenizer(next_q[0], padding="max_length", max_length=self._max_decoder_length, truncation=True)
            next_q_labels = np.asarray(next_q_encode.input_ids)
            next_q_labels[next_q_labels == self._tokenizer.pad_token_id] = -100
        elif len(next_q) >= 2:
            next_q_toks = []

            for q in next_q:
                next_q_toks += self._tokenizer.tokenize(q) + ["</s>"]

            next_q_toks = next_q_toks[:self._max_decoder_length-2]
            next_q_toks = ["<s>"] + next_q_toks + ["</s>"]

            assert len(next_q_toks) <= self._max_decoder_length
            while len(next_q_toks) < self._max_decoder_length:
                next_q_toks.append("<pad>")
            assert len(next_q_toks) == self._max_decoder_length
            next_q_labels = np.asarray(self._tokenizer.convert_tokens_to_ids(next_q_toks))
            next_q_labels[next_q_labels == self._tokenizer.pad_token_id] = -100

        if len(click_doc) == 1:
            click_doc_encode = self._tokenizer(click_doc[0], padding="max_length", max_length=self._max_decoder_length, truncation=True)
            click_doc_labels = np.asarray(click_doc_encode.input_ids)
            click_doc_labels[click_doc_labels == self._tokenizer.pad_token_id] = -100
        elif len(click_doc) >= 2:
            click_doc_toks = []

            for d in click_doc:
                click_doc_toks += self._tokenizer.tokenize(d) + ["</s>"]

            click_doc_toks = click_doc_toks[:self._max_decoder_length-2]
            click_doc_toks = ["<s>"] + click_doc_toks + ["</s>"]

            assert len(click_doc_toks) <= self._max_decoder_length
            while len(click_doc_toks) < self._max_decoder_length:
                click_doc_toks.append("<pad>")
            assert len(click_doc_toks) == self._max_decoder_length
            click_doc_labels = np.asarray(self._tokenizer.convert_tokens_to_ids(click_doc_toks))
            click_doc_labels[click_doc_labels == self._tokenizer.pad_token_id] = -100

        if len(previous_q) == 1:
            previous_q_encode = self._tokenizer(previous_q[0], padding="max_length", max_length=self._max_decoder_length, truncation=True)
            previous_q_labels = np.asarray(previous_q_encode.input_ids)
            previous_q_labels[previous_q_labels == self._tokenizer.pad_token_id] = -100
        elif len(previous_q) >= 2:
            previous_q_toks = []

            for q in previous_q:
                previous_q_toks += self._tokenizer.tokenize(q) + ["</s>"]

            previous_q_toks = previous_q_toks[:self._max_decoder_length-2]
            previous_q_toks = ["<s>"] + previous_q_toks + ["</s>"]

            assert len(previous_q_toks) <= self._max_decoder_length
            while len(previous_q_toks) < self._max_decoder_length:
                previous_q_toks.append("<pad>")
            assert len(previous_q_toks) == self._max_decoder_length
            previous_q_labels = np.asarray(self._tokenizer.convert_tokens_to_ids(previous_q_toks))
            previous_q_labels[previous_q_labels == self._tokenizer.pad_token_id] = -100
        
        if len(simq) == 1:
            simq_encode = self._tokenizer(simq[0], padding="max_length", max_length=self._max_decoder_length, truncation=True)
            simq_labels = np.asarray(simq_encode.input_ids)
            simq_labels[simq_labels == self._tokenizer.pad_token_id] = -100
        elif len(simq) >= 2:
            simq_toks = []

            for q in simq:
                simq_toks += self._tokenizer.tokenize(q) + ["</s>"]

            simq_toks = simq_toks[:self._max_decoder_length-2]
            simq_toks = ["<s>"] + simq_toks + ["</s>"]

            assert len(simq_toks) <= self._max_decoder_length
            while len(simq_toks) < self._max_decoder_length:
                simq_toks.append("<pad>")
            assert len(simq_toks) == self._max_decoder_length
            simq_labels = np.asarray(self._tokenizer.convert_tokens_to_ids(simq_toks))
            simq_labels[simq_labels == self._tokenizer.pad_token_id] = -100
        return next_q_labels, click_doc_labels, previous_q_labels, simq_labels

    def __len__(self):
        return self._total_data

# Training dataset for Tiangong-ST
class TgTrainFileDataset(Dataset):
    def __init__(self, filename, tokenizer, dataset):
        super(TgTrainFileDataset, self).__init__()
        self._filename = filename
        self._max_seq_length = 128
        self._max_decoder_length = 60
        self._tokenizer = tokenizer
        self._dataset = dataset
        with open(filename, "r") as f:
            self._total_data = len(f.readlines())
    
    def check_length(self, pairlist):
        if len(pairlist) % 2 != 0:
            print(pairlist)
        assert len(pairlist) % 2 == 0
        max_seq_length = self._max_seq_length - 3
        if len(pairlist) == 2:
            while len(pairlist[0]) + len(pairlist[1]) + 2 > max_seq_length:
                if len(pairlist[0]) > len(pairlist[1]):
                    pairlist[0].pop(0)
                else:
                    pairlist[1].pop(-1)
        else:
            q_d_minimum_length = 0
            for i in range(len(pairlist)):
                q_d_minimum_length += len(pairlist[i]) + 1
            if q_d_minimum_length > max_seq_length:
                pairlist.pop(0)
                pairlist.pop(0)
                pairlist = self.check_length(pairlist)
        return pairlist
    
    def anno_encoder(self, all_qd_toks):
        '''
        Get input ids of encoder.
        '''

        # position of the [CLS] token
        eos_position = 0

        encode_attention_mask = [1] * len(all_qd_toks)
        assert len(all_qd_toks) <= self._max_seq_length
        while len(all_qd_toks) < self._max_seq_length:
            all_qd_toks.append("[PAD]")
            encode_attention_mask.append(0)
        assert len(all_qd_toks) == len(encode_attention_mask) == self._max_seq_length
        anno_seq = self._tokenizer.convert_tokens_to_ids(all_qd_toks)

        eos_mask = [0] * len(all_qd_toks)
        eos_mask[eos_position] = 1
        encoder_input_ids = np.asarray(anno_seq)
        encode_attention_mask = np.asarray(encode_attention_mask)

        return encoder_input_ids, encode_attention_mask, eos_mask

    def anno_main(self, qd_pairs):
        all_qd = []
        for i in range(len(qd_pairs)):
            qd = qd_pairs[i]
            qd = self._tokenizer.tokenize(qd)
            qd += ["[eos]"]
            all_qd.append(qd)
        all_qd = self.check_length(all_qd)

        history = all_qd[:-2]
        query_tok = all_qd[-2]
        doc_tok = all_qd[-1]
        
        history_toks = []
        for iidx, sent in enumerate(history):
            history_toks.extend(sent)
        
        query_tok += ["[SEP]"]
        doc_tok += ["[SEP]"]

        all_qd_toks_rank = ["[CLS]"] + ["[rank]"] + history_toks + query_tok + doc_tok

        # encoder input

        encoder_input_ids_rank, encode_attention_mask_rank, eos_mask = self.anno_encoder(all_qd_toks_rank)

        return encoder_input_ids_rank, encode_attention_mask_rank, eos_mask
    
    def __getitem__(self, idx):
        line = linecache.getline(self._filename, idx + 1)
        label, num_neg, click_doc, next_q, previous_q, qd_pairs, docs, simq = split_3gen_2w(line)

        if len(docs) < num_neg+1:
            print(len(docs), docs, num_neg)
        assert len(docs) >= num_neg+1

        batches = []

        # gen labels

        next_q_labels, click_doc_labels, previous_q_labels, simq_labels = self.get_sep_gen_labels(next_q, click_doc, previous_q, simq)
        
        # batches

        for i in range(len(docs)):
            doc = docs[i]
            qds = qd_pairs + [doc]

            encoder_input_ids_rank, encode_attention_mask_rank, eos_position = self.anno_main(qds)
            
            encoder_input_ids_rank[1] = self._tokenizer("[genfq]")['input_ids'][1]
            encoder_input_ids_gen_fq = copy.deepcopy(encoder_input_ids_rank)

            encoder_input_ids_rank[1] = self._tokenizer("[rank]")['input_ids'][1]
            encoder_input_ids_rank1 = copy.deepcopy(encoder_input_ids_rank)
            
            encoder_input_ids_rank[1] = self._tokenizer("[gencd]")['input_ids'][1]
            encoder_input_ids_gen_cd = copy.deepcopy(encoder_input_ids_rank)
            
            encoder_input_ids_rank[1] = self._tokenizer("[gensq]")['input_ids'][1]
            encoder_input_ids_gen_sq = copy.deepcopy(encoder_input_ids_rank)

            batch = {
                'encoder_input_ids_gen_fq': np.asarray(encoder_input_ids_gen_fq), 
                'encoder_input_ids_gen_cd': np.asarray(encoder_input_ids_gen_cd), 
                'encoder_input_ids_gen_sq': np.asarray(encoder_input_ids_gen_sq), 
                'encoder_input_ids_rank': np.asarray(encoder_input_ids_rank1), 
                'encode_attention_mask_rank': np.asarray(encode_attention_mask_rank),
                'next_q_labels': np.asarray(next_q_labels, dtype=np.int64), 
                'click_doc_labels': np.asarray(click_doc_labels, dtype=np.int64), 
                'previous_q_labels': np.asarray(previous_q_labels, dtype=np.int64),
                'simq_labels': np.asarray(simq_labels, dtype=np.int64),
                'eos_position': np.asarray(eos_position, dtype=np.int64),
            }
            batches.append(batch)
        
        if self._dataset == 'aol':
            batches, loss_mask = self.pad_aol_batches(batches, num_neg)
            assert len(batches) == 5
            assert len(loss_mask) == 4
        elif self._dataset == 'tiangong':
            batches, loss_mask = self.pad_tg_atches(batches, num_neg)
            assert len(batches) == 10
            assert len(loss_mask) == 9

        for batch in batches:
            batch["loss_mask"] = np.asarray(loss_mask, dtype=np.int64)

        return batches

    def pad_aol_batches(self, batches, num_neg):
        if len(batches) == 5:
            return batches, [1]*4
        else:
            mask = [1]*(len(batches)-1) + [0]*(4-num_neg)
            for i in range(5-len(batches)):
                batches.append(
                    batches[-1]
                )
            return batches, mask
    
    def pad_tg_atches(self, batches, num_neg):
        if len(batches) == 10:
            return batches, [1]*9
        else:
            mask = [1]*(len(batches)-1) + [0]*(9-num_neg)
            for i in range(10-len(batches)):
                batches.append(
                    batches[-1]
                )
            return batches, mask
    
    def get_sep_gen_labels(self, next_q, click_doc, previous_q, simq):
        if len(next_q) == 1:
            next_q_encode = self._tokenizer(next_q[0], padding="max_length", max_length=self._max_decoder_length, truncation=True)
            next_q_labels = np.asarray(next_q_encode.input_ids)
            next_q_labels[next_q_labels == self._tokenizer.pad_token_id] = -100
        elif len(next_q) >= 2:
            next_q_toks = []

            for q in next_q:
                next_q_toks += self._tokenizer.tokenize(q) + ["[SEP]"]

            next_q_toks = next_q_toks[:self._max_decoder_length-2]
            next_q_toks = ["[CLS]"] + next_q_toks + ["[SEP]"]

            assert len(next_q_toks) <= self._max_decoder_length
            while len(next_q_toks) < self._max_decoder_length:
                next_q_toks.append("[PAD]")
            assert len(next_q_toks) == self._max_decoder_length
            next_q_labels = np.asarray(self._tokenizer.convert_tokens_to_ids(next_q_toks))
            next_q_labels[next_q_labels == self._tokenizer.pad_token_id] = -100

        if len(click_doc) == 1:
            click_doc_encode = self._tokenizer(click_doc[0], padding="max_length", max_length=self._max_decoder_length, truncation=True)
            click_doc_labels = np.asarray(click_doc_encode.input_ids)
            click_doc_labels[click_doc_labels == self._tokenizer.pad_token_id] = -100
        elif len(click_doc) >= 2:
            click_doc_toks = []

            for d in click_doc:
                click_doc_toks += self._tokenizer.tokenize(d) + ["[SEP]"]

            click_doc_toks = click_doc_toks[:self._max_decoder_length-2]
            click_doc_toks = ["[CLS]"] + click_doc_toks + ["[SEP]"]

            assert len(click_doc_toks) <= self._max_decoder_length
            while len(click_doc_toks) < self._max_decoder_length:
                click_doc_toks.append("[PAD]")
            assert len(click_doc_toks) == self._max_decoder_length
            click_doc_labels = np.asarray(self._tokenizer.convert_tokens_to_ids(click_doc_toks))
            click_doc_labels[click_doc_labels == self._tokenizer.pad_token_id] = -100

        if len(previous_q) == 1:
            previous_q_encode = self._tokenizer(previous_q[0], padding="max_length", max_length=self._max_decoder_length, truncation=True)
            previous_q_labels = np.asarray(previous_q_encode.input_ids)
            previous_q_labels[previous_q_labels == self._tokenizer.pad_token_id] = -100
        elif len(previous_q) >= 2:
            previous_q_toks = []

            for q in previous_q:
                previous_q_toks += self._tokenizer.tokenize(q) + ["[SEP]"]

            previous_q_toks = previous_q_toks[:self._max_decoder_length-2]
            previous_q_toks = ["[CLS]"] + previous_q_toks + ["[SEP]"]

            assert len(previous_q_toks) <= self._max_decoder_length
            while len(previous_q_toks) < self._max_decoder_length:
                previous_q_toks.append("[PAD]")
            assert len(previous_q_toks) == self._max_decoder_length
            previous_q_labels = np.asarray(self._tokenizer.convert_tokens_to_ids(previous_q_toks))
            previous_q_labels[previous_q_labels == self._tokenizer.pad_token_id] = -100
        
        if len(simq) == 1:
            simq_encode = self._tokenizer(simq[0], padding="max_length", max_length=self._max_decoder_length, truncation=True)
            simq_labels = np.asarray(simq_encode.input_ids)
            simq_labels[simq_labels == self._tokenizer.pad_token_id] = -100
        elif len(simq) >= 2:
            simq_toks = []

            for q in simq:
                simq_toks += self._tokenizer.tokenize(q) + ["[SEP]"]

            simq_toks = simq_toks[:self._max_decoder_length-2]
            simq_toks = ["[CLS]"] + simq_toks + ["[SEP]"]

            assert len(simq_toks) <= self._max_decoder_length
            while len(simq_toks) < self._max_decoder_length:
                simq_toks.append("[PAD]")
            assert len(simq_toks) == self._max_decoder_length
            simq_labels = np.asarray(self._tokenizer.convert_tokens_to_ids(simq_toks))
            simq_labels[simq_labels == self._tokenizer.pad_token_id] = -100
        return next_q_labels, click_doc_labels, previous_q_labels, simq_labels
    
    def __len__(self):
        return self._total_data

# Different data processing functions for different values of w.

def split_3gen_1w(line):
    items = line.strip().split("====")
    line = items[0].strip().split("\t")
    label = int(line[0])
    num_neg = int(line[1])
    next_q = line[2]
    click_doc = line[3]
    previous_q = line[4]
    simq = line[5]
    qd_pairs = line[6:]

    docs = items[1].split("\t")

    return label, num_neg, [click_doc], [next_q], [previous_q], qd_pairs, docs, [simq]


def split_3gen_2w(line):
    items = line.strip().split("====")
    line = items[0].strip().split("\t")
    label = int(line[0])
    num_neg = int(line[1])
    next_q1 = line[2]
    next_q2 = line[3]
    click_doc = line[4]
    next_doc = line[5]
    previous_q1 = line[6]
    previous_q2 = line[7]
    simq = line[8]
    qd_pairs = line[9:]

    docs = items[1].split("\t")

    return label, num_neg, [click_doc, next_doc], [next_q1, next_q2], [previous_q1, previous_q2], qd_pairs, docs, [simq]

def split_3gen_3w(line):
    items = line.strip().split("====")
    line = items[0].strip().split("\t")
    label = int(line[0])
    num_neg = int(line[1])
    next_q1 = line[2]
    next_q2 = line[3]
    next_q3 = line[4]
    click_doc = line[5]
    next_doc1 = line[6]
    next_doc2 = line[7]
    previous_q1 = line[8]
    previous_q2 = line[9]
    previous_q3 = line[10]
    simq = line[11]
    qd_pairs = line[12:]

    docs = items[1].split("\t")

    return label, num_neg, [click_doc, next_doc1, next_doc2], [next_q1, next_q2, next_q3], [previous_q1, previous_q2, previous_q3], qd_pairs, docs, [simq]

def split_3gen_4w(line):
    items = line.strip().split("====")
    line = items[0].strip().split("\t")
    label = int(line[0])
    num_neg = int(line[1])
    next_q1 = line[2]
    next_q2 = line[3]
    next_q3 = line[4]
    next_q4 = line[5]
    click_doc = line[6]
    next_doc1 = line[7]
    next_doc2 = line[8]
    next_doc3 = line[9]
    previous_q1 = line[10]
    previous_q2 = line[11]
    previous_q3 = line[12]
    simq = line[13]
    qd_pairs = line[14:]

    docs = items[1].split("\t")

    return label, num_neg, [click_doc, next_doc1, next_doc2, next_doc3], [next_q1, next_q2, next_q3, next_q4], [previous_q1, previous_q2, previous_q3], qd_pairs, docs, [simq]
