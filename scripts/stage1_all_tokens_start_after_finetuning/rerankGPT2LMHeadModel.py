from typing import Callable, Dict, Optional, Tuple, Union, Any, collections
import torch
import copy 
import random
import numpy as np

from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss

from torch.utils.data import Dataset

from transformers.file_utils import (
    WEIGHTS_NAME,
    is_apex_available,
    is_datasets_available,
    is_in_notebook,
    is_sagemaker_distributed_available,
    is_torch_tpu_available,
)

from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    SequenceClassifierOutputWithPast,
)

from transformers import GPT2LMHeadModel, GPT2Model

class wiki2_GPT2Dataset(Dataset):
    def __init__(self, txt_list, tokenizer, gpt2_type="gpt2", max_length=768):

        self.tokenizer = tokenizer
        self.input_ids = []
    
        for txt in txt_list:
            encodings_dict = tokenizer(txt, truncation=True, max_length=max_length, padding="max_length")
            self.input_ids.append(torch.tensor(encodings_dict['input_ids']))
    
    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx]
    
class wiki2021_GPT2Dataset(Dataset):
    def __init__(self, input_ids):
        self.input_ids = input_ids
    
    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx]

class rerankGPT2LMHeadModel_labelInCandidateWhenTraning(GPT2LMHeadModel):
    def __init__(self, config, MAX_LEN, CAN_NUM, num_of_rerank):
        super().__init__(config)
        self.MAX_LEN = MAX_LEN
        self.CAN_NUM = CAN_NUM
        self.num_of_rerank = num_of_rerank
        self.VOCAB_SIZE = config.vocab_size
        
        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
#         self.dropout = nn.Dropout(config.hidden_dropout_prob)
#         self.rerank_transformer = GPT2Model(config)
        self.rerank_linear_head = nn.Linear(config.n_embd, 1, bias=False)

        self.init_weights()
    def forward(
        self,
        input_ids=None,
        labels=None,
        is_training=False,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            ``labels = input_ids`` Indices are selected in ``[-100, 0, ..., config.vocab_size]`` All labels set to
            ``-100`` are ignored (masked), the loss is only computed for labels in ``[0, ..., config.vocab_size]``
        """        
        # rerank_places = random.sample(np.arange(1, MAX_LEN-2-CAN_NUM*2).tolist(), k=num_of_rerank) #no duplicate
        rerank_places = random.sample(np.arange(1, self.MAX_LEN).tolist(), k=self.num_of_rerank) #no duplicate
        rerank_places = np.concatenate(([0], np.sort(rerank_places), [self.MAX_LEN])) #add first and last tokens to make segments
        
        past_key_values = None
        hidden_states_in_rerank_place = []
        labels_in_rerank_place = []
        all_rerank_hidden_states = []
        all_rerank_labels = []
        hidden_states = []
        
        no_rerank_logits = []
        check_out_num = 0
        for i in range(self.num_of_rerank+1):
            #normal stage
            segment_input_ids = input_ids[:, rerank_places[i]:rerank_places[i+1]]

            segment_outputs = self.transformer(
                segment_input_ids,
                past_key_values = past_key_values,
                token_type_ids = torch.zeros_like(segment_input_ids)
            )

            segment_hidden = segment_outputs[0]
            past_key_values = segment_outputs[1]

            hidden_states.append(segment_hidden)
            #rerank stage (just for rerank places)
            if i == self.num_of_rerank:
                break

            #rerank stage
            #get logits in rerank place
            logits_before_rerank = self.lm_head(segment_hidden[:, -1, :])
            #get candidate token ids according to the logits
            candidate_token_logits, candidate_token_ids = torch.topk(logits_before_rerank, self.CAN_NUM)
            rerank_labels = labels[..., rerank_places[i+1]]
            labels_in_rerank_place.append(rerank_labels)
            hidden_states_in_rerank_place.append(segment_hidden[:, -1, :])
            
            #randomize the order of candidates
            shuffle_index = np.random.permutation(self.CAN_NUM)
            candidate_token_logits = candidate_token_logits[:, shuffle_index]
            candidate_token_ids = candidate_token_ids[:, shuffle_index]
            
            #check whether or not label in candidates
            check_labels = rerank_labels.tolist()
            check_candidates = candidate_token_ids.tolist()
            
            assert len(check_labels)==len(check_candidates)
            
            
            #when training, check whether or not label in candidates, if not we add label into candidates
            if is_training:
                #check every data in batch
                rerank_labels_this_place = []
                for j in range(len(check_labels)):
                    if check_labels[j] not in check_candidates[j]:
                        check_out_num+=1
                        replace_index = np.random.randint(self.CAN_NUM)
                        candidate_token_ids[j][replace_index] = check_labels[j]
                        rerank_labels_this_place.append(replace_index)          
                    else:
                        rerank_labels_this_place.append(check_candidates[j].index(check_labels[j]))
                all_rerank_labels.append(torch.tensor(rerank_labels_this_place, device=input_ids.device))
            #when eval, check whether or not label in candidates, if not we do not do rerank
            else:
                rerank_labels = []
                check_in_index = []

                for j in range(len(check_labels)): 
                    if check_labels[j] in check_candidates[j]:
                        rerank_labels.append(check_candidates[j].index(check_labels[j]))
                        check_in_index.append(j)
                    else:
                        check_out_num+=1
                rerank_labels = torch.tensor(rerank_labels, device=input_ids.device)
                    
                if rerank_labels.shape[0] == 0:
                    continue
                else:
                    all_rerank_labels.append(rerank_labels)

            #make context for rerank stage, 50256 is the token_id for </endoftext/>
            sep_token = torch.ones(size = [candidate_token_ids.shape[0], 1], dtype = torch.long, device=input_ids.device) * 50256
            candidate_context_ids = torch.cat([sep_token, candidate_token_ids], -1)
       
            token_type_ids = torch.ones_like(candidate_context_ids)
            candidate_position_id = torch.ones_like(candidate_context_ids, dtype = torch.long, device=input_ids.device) * rerank_places[i+1]
            #get output from gpt2
            rerank_outputs = self.transformer(candidate_context_ids,
                                              past_key_values=past_key_values,
                                              token_type_ids=token_type_ids,
                                              position_ids = candidate_position_id
                          )

            #get rerank logits for candidates
            rerank_hidden_states = rerank_outputs[0][:, 1:]

            if not is_training:
                all_rerank_hidden_states.append(rerank_hidden_states[check_in_index])
                no_rerank_logits.append(candidate_token_logits[check_in_index])
            else:
                all_rerank_hidden_states.append(rerank_hidden_states)
                no_rerank_logits.append(candidate_token_logits)
        
#         print("\n batch info:")
#         print("there are ", check_out_num/(self.num_of_rerank*batch_size), "labels not in candidates")
        
        #-------------------------------------------------------------------------
        # cal loss, loss = normal loss + rerank loss
        loss_fct = CrossEntropyLoss(reduction='none')
        
        # cal rerank loss
        rerank_loss = None
        
        all_rerank_hidden_states = torch.cat(all_rerank_hidden_states, 0)
        all_rerank_logits = self.rerank_linear_head(all_rerank_hidden_states)
        all_rerank_logits = torch.reshape(all_rerank_logits, [-1, self.CAN_NUM])
        all_rerank_labels = torch.cat(all_rerank_labels, 0)
        
        rerank_loss = loss_fct(all_rerank_logits, all_rerank_labels)
        
                # cal normal loss
        normal_loss = None

        hidden_states = torch.cat(hidden_states, 1)
        lm_logits = self.lm_head(hidden_states)

        # Shift so that tokens < n predict n
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        normal_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        # cal normal loss in rerank place
        normal_loss_in_rerank_place_across_all_vocab = None
        normal_loss_in_rerank_place_across_candidates = None

        hidden_states_in_rerank_place = torch.cat(hidden_states_in_rerank_place, 0)
        lm_logits_in_rerank_place = self.lm_head(hidden_states_in_rerank_place)
        lm_logits_in_rerank_place = torch.reshape(lm_logits_in_rerank_place, [-1, self.VOCAB_SIZE])
        labels_in_rerank_place = torch.cat(labels_in_rerank_place, 0)

        no_rerank_logits = torch.cat(no_rerank_logits, 0)
        no_rerank_logits = torch.reshape(no_rerank_logits, [-1, self.CAN_NUM])

        stage1_loss_in_rerank_place_across_all_vocab = loss_fct(lm_logits_in_rerank_place, labels_in_rerank_place)
        stage1_loss_in_rerank_place_across_candidates = loss_fct(no_rerank_logits, all_rerank_labels)
        
        return {"stage1_loss_in_rerank_place_across_all_vocab": stage1_loss_in_rerank_place_across_all_vocab,
                "stage1_loss_in_rerank_place_across_candidates": stage1_loss_in_rerank_place_across_candidates,
                "normal_loss": normal_loss,
                "rerank_loss": rerank_loss,}
    
        
class rerankGPT2LMHeadModel_stage1_all_tokens_stage2_sample_tokens(GPT2LMHeadModel):
    def __init__(self, config, MAX_LEN, CAN_NUM, num_of_rerank):
        super().__init__(config)
        self.MAX_LEN = MAX_LEN
        self.CAN_NUM = CAN_NUM
        self.num_of_rerank = num_of_rerank
        self.VOCAB_SIZE = config.vocab_size
        
        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        self.rerank_linear_head = nn.Linear(config.n_embd, 1, bias=False)

        self.init_weights()
    def forward(
        self,
        input_ids=None,
        labels=None,
        is_training=False,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            ``labels = input_ids`` Indices are selected in ``[-100, 0, ..., config.vocab_size]`` All labels set to
            ``-100`` are ignored (masked), the loss is only computed for labels in ``[0, ..., config.vocab_size]``
        """
        # make some model parameter not change during rerank (like dropout) ??????????????
        # model.eval()
        if is_training:
            self.transformer.eval()

        rerank_places = random.sample(np.arange(1, self.MAX_LEN).tolist(), k=self.num_of_rerank) #no duplicate
        rerank_places = np.concatenate(([0], np.sort(rerank_places), [self.MAX_LEN])) #add first and last tokens to make segments
        
        past_key_values = None
        all_rerank_hidden_states = []
        all_rerank_labels = []
        no_rerank_logits = []
        check_out_num = 0
        
        labels_in_rerank_place = []
        hidden_states_in_rerank_place = []
        
        hidden_states = []
        
        for i in range(self.num_of_rerank+1):
            #normal stage
            segment_input_ids = input_ids[:, rerank_places[i]:rerank_places[i+1]]

            segment_outputs = self.transformer(
                segment_input_ids,
                past_key_values = past_key_values,
                token_type_ids = torch.zeros_like(segment_input_ids)
            )

            segment_hidden = segment_outputs[0]
            past_key_values = segment_outputs[1]

            hidden_states.append(segment_hidden)
            #rerank stage (just for rerank places)
            if i == self.num_of_rerank:
                break

            #rerank stage
            #get logits in rerank place
            logits_before_rerank = self.lm_head(segment_hidden[:, -1, :])
            #get candidate token ids according to the logits
            candidate_token_logits, candidate_token_ids = torch.topk(logits_before_rerank, self.CAN_NUM)
            rerank_labels = labels[..., rerank_places[i+1]]
            
            labels_in_rerank_place.append(rerank_labels)
            hidden_states_in_rerank_place.append(segment_hidden[:, -1, :])
            
            #randomize the order of candidates
            shuffle_index = np.random.permutation(self.CAN_NUM)
            candidate_token_logits = candidate_token_logits[:, shuffle_index]
            candidate_token_ids = candidate_token_ids[:, shuffle_index]
            
            #check whether or not label in candidates
            check_labels = rerank_labels.tolist()
            check_candidates = candidate_token_ids.tolist()
            
            assert len(check_labels)==len(check_candidates)
            
            #check whether or not label in candidates, if not we do not do rerank
            rerank_labels = []
            check_in_index = []

            for j in range(len(check_labels)): 
                if check_labels[j] in check_candidates[j]:
                    rerank_labels.append(check_candidates[j].index(check_labels[j]))
                    check_in_index.append(j)
                else:
                    check_out_num+=1
            rerank_labels = torch.tensor(rerank_labels, device=input_ids.device)

            if rerank_labels.shape[0] == 0:
                continue
            else:
                all_rerank_labels.append(rerank_labels)

            #make context for rerank stage, 50256 is the token_id for </endoftext/>
            sep_token = torch.ones(size = [candidate_token_ids.shape[0], 1], dtype = torch.long, device=input_ids.device) * 50256
            candidate_context_ids = torch.cat([sep_token, candidate_token_ids], -1)
       
            token_type_ids = torch.ones_like(candidate_context_ids)
            candidate_position_id = torch.ones_like(candidate_context_ids, dtype = torch.long, device=input_ids.device) * rerank_places[i+1]
            #get output from gpt2
            rerank_outputs = self.transformer(candidate_context_ids,
                                              past_key_values=past_key_values,
                                              token_type_ids=token_type_ids,
                                              position_ids = candidate_position_id
                          )

            #get rerank logits for candidates
            rerank_hidden_states = rerank_outputs[0][:, 1:]

            all_rerank_hidden_states.append(rerank_hidden_states[check_in_index])
            no_rerank_logits.append(candidate_token_logits[check_in_index])
            
        if is_training:
            #model.train()
            self.transformer.train()
        
        #-------------------------------------------------------------------------
        # cal loss, loss = normal loss + rerank loss
        loss_fct = CrossEntropyLoss(reduction='none')

        # cal rerank loss
        rerank_loss = None

        all_rerank_hidden_states = torch.cat(all_rerank_hidden_states, 0)
        all_rerank_logits = self.rerank_linear_head(all_rerank_hidden_states)
        all_rerank_logits = torch.reshape(all_rerank_logits, [-1, self.CAN_NUM])
        all_rerank_labels = torch.cat(all_rerank_labels, 0)

        rerank_loss = loss_fct(all_rerank_logits, all_rerank_labels)
        
        stage1_loss_in_rerank_place_across_candidates = None


        no_rerank_logits = torch.cat(no_rerank_logits, 0)
        no_rerank_logits = torch.reshape(no_rerank_logits, [-1, self.CAN_NUM])

        stage1_loss_in_rerank_place_across_candidates = loss_fct(no_rerank_logits, all_rerank_labels)
        
        stage1_loss_in_rerank_place_across_all_vocab = None
        
        hidden_states_in_rerank_place = torch.cat(hidden_states_in_rerank_place, 0)
        lm_logits_in_rerank_place = self.lm_head(hidden_states_in_rerank_place)
        lm_logits_in_rerank_place = torch.reshape(lm_logits_in_rerank_place, [-1, self.VOCAB_SIZE])
        labels_in_rerank_place = torch.cat(labels_in_rerank_place, 0)

        stage1_loss_in_rerank_place_across_all_vocab = loss_fct(lm_logits_in_rerank_place, labels_in_rerank_place)

        # cal normal loss
        normal_loss = None

        hidden_states = torch.cat(hidden_states, 1)
        lm_logits = self.lm_head(hidden_states)

        # Shift so that tokens < n predict n
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        normal_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return {"stage1_loss_in_rerank_place_across_all_vocab": stage1_loss_in_rerank_place_across_all_vocab,
                "stage1_loss_in_rerank_place_across_candidates": stage1_loss_in_rerank_place_across_candidates,
                "normal_loss": normal_loss,
                "rerank_loss": rerank_loss,}
        
        
class rerankGPT2LMHeadModel_stage1_all_tokens_no_stage2(GPT2LMHeadModel):
    def __init__(self, config, MAX_LEN, CAN_NUM, num_of_rerank):
        super().__init__(config)
        self.MAX_LEN = MAX_LEN
        self.CAN_NUM = CAN_NUM
        self.num_of_rerank = num_of_rerank
        self.VOCAB_SIZE = config.vocab_size
        
        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.init_weights()
    def forward(
        self,
        input_ids=None,
        labels=None,
        is_training=False,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            ``labels = input_ids`` Indices are selected in ``[-100, 0, ..., config.vocab_size]`` All labels set to
            ``-100`` are ignored (masked), the loss is only computed for labels in ``[0, ..., config.vocab_size]``
        """
        # make some model parameter not change during rerank (like dropout) ??????????????
        # model.eval()
        if is_training:
            self.transformer.eval()

        rerank_places = random.sample(np.arange(1, self.MAX_LEN).tolist(), k=self.num_of_rerank) #no duplicate
        rerank_places = np.concatenate(([0], np.sort(rerank_places), [self.MAX_LEN])) #add first and last tokens to make segments
        
        past_key_values = None
        all_rerank_hidden_states = []
        all_rerank_labels = []
        no_rerank_logits = []
        check_out_num = 0
        
        hidden_states = []
        
        for i in range(self.num_of_rerank+1):
            #normal stage
            segment_input_ids = input_ids[:, rerank_places[i]:rerank_places[i+1]]

            segment_outputs = self.transformer(
                segment_input_ids,
                past_key_values = past_key_values
            )

            segment_hidden = segment_outputs[0]
            past_key_values = segment_outputs[1]

            hidden_states.append(segment_hidden)
        
#         print("\n batch info:")
#         print("there are ", check_out_num/(self.num_of_rerank*batch_size), "labels not in candidates")
    
        #model.train()
        self.transformer.train()

        #-------------------------------------------------------------------------
        # cal loss, loss = normal loss + rerank loss
        loss_fct = CrossEntropyLoss(reduction='none')

        # cal normal loss
        normal_loss = None

        hidden_states = torch.cat(hidden_states, 1)
        lm_logits = self.lm_head(hidden_states)

        # Shift so that tokens < n predict n
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        normal_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return {"normal_loss": normal_loss}
        
class rerankGPT2LMHeadModel_randomlySamplingNwordsIntoTheCandidate(GPT2LMHeadModel):
    def __init__(self, config, MAX_LEN, CAN_NUM, num_of_rerank):
        super().__init__(config)
        self.MAX_LEN = MAX_LEN
        self.CAN_NUM = CAN_NUM
        self.num_of_rerank = num_of_rerank
        self.VOCAB_SIZE = config.vocab_size
        
        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
#         self.dropout = nn.Dropout(config.hidden_dropout_prob)
#         self.rerank_transformer = GPT2Model(config)
        self.rerank_linear_head = nn.Linear(config.n_embd, 1, bias=False)

        self.init_weights()
    def forward(
        self,
        input_ids=None,
        labels=None,
        is_training=False,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            ``labels = input_ids`` Indices are selected in ``[-100, 0, ..., config.vocab_size]`` All labels set to
            ``-100`` are ignored (masked), the loss is only computed for labels in ``[0, ..., config.vocab_size]``
        """        
        # rerank_places = random.sample(np.arange(1, MAX_LEN-2-CAN_NUM*2).tolist(), k=num_of_rerank) #no duplicate
        rerank_places = random.sample(np.arange(1, self.MAX_LEN).tolist(), k=self.num_of_rerank) #no duplicate
        rerank_places = np.concatenate(([0], np.sort(rerank_places), [self.MAX_LEN])) #add first and last tokens to make segments
        
        past_key_values = None
        hidden_states_in_rerank_place = []
        labels_in_rerank_place = []
        all_rerank_hidden_states = []
        all_rerank_labels = []
        hidden_states = []
        
        no_rerank_logits = []
        check_out_num = 0
        for i in range(self.num_of_rerank+1):
            #normal stage
            segment_input_ids = input_ids[:, rerank_places[i]:rerank_places[i+1]]

            segment_outputs = self.transformer(
                segment_input_ids,
                past_key_values = past_key_values,
                token_type_ids = torch.zeros_like(segment_input_ids)
            )

            segment_hidden = segment_outputs[0]
            past_key_values = segment_outputs[1]

            hidden_states.append(segment_hidden)
            #rerank stage (just for rerank places)
            if i == self.num_of_rerank:
                break

            #rerank stage
            #get logits in rerank place
            logits_before_rerank = self.lm_head(segment_hidden[:, -1, :])
            #get candidate token ids according to the logits
            #candidate_token_logits, candidate_token_ids = torch.topk(logits_before_rerank, self.CAN_NUM)
            sorted_logits, indices = torch.sort(logits_before_rerank, descending=True)

            top_logits = sorted_logits[:, :50]
            top_indices = indices[:, :50]

            sorted_logits = sorted_logits[:, 50:1050]
            indices = indices[:, 50:1050]

            p = torch.ones(1000)/1000
            n = 50
            replace = False

            idx = torch.multinomial(p, num_samples=n, replacement=replace)
            selected_logits = sorted_logits[:, idx.tolist()]
            selected_indices = indices[:, idx.tolist()]

            candidate_token_logits = torch.cat((top_logits, selected_logits), dim=-1)
            candidate_token_ids = torch.cat((top_indices, selected_indices), dim=-1)
            
            rerank_labels = labels[..., rerank_places[i+1]]
            labels_in_rerank_place.append(rerank_labels)
            hidden_states_in_rerank_place.append(segment_hidden[:, -1, :])
            
            #randomize the order of candidates
            shuffle_index = np.random.permutation(self.CAN_NUM)
            candidate_token_logits = candidate_token_logits[:, shuffle_index]
            candidate_token_ids = candidate_token_ids[:, shuffle_index]
            
            #check whether or not label in candidates
            check_labels = rerank_labels.tolist()
            check_candidates = candidate_token_ids.tolist()
            
            assert len(check_labels)==len(check_candidates)
            
            
            #when training, check whether or not label in candidates, if not we add label into candidates
            if is_training:
                #check every data in batch
                rerank_labels_this_place = []
                for j in range(len(check_labels)):
                    if check_labels[j] not in check_candidates[j]:
                        check_out_num+=1
                        replace_index = np.random.randint(self.CAN_NUM)
                        candidate_token_ids[j][replace_index] = check_labels[j]
                        rerank_labels_this_place.append(replace_index)          
                    else:
                        rerank_labels_this_place.append(check_candidates[j].index(check_labels[j]))
                all_rerank_labels.append(torch.tensor(rerank_labels_this_place, device=input_ids.device))
            #when eval, check whether or not label in candidates, if not we do not do rerank
            else:
                rerank_labels = []
                check_in_index = []

                for j in range(len(check_labels)): 
                    if check_labels[j] in check_candidates[j]:
                        rerank_labels.append(check_candidates[j].index(check_labels[j]))
                        check_in_index.append(j)
                    else:
                        check_out_num+=1
                rerank_labels = torch.tensor(rerank_labels, device=input_ids.device)
                    
                if rerank_labels.shape[0] == 0:
                    continue
                else:
                    all_rerank_labels.append(rerank_labels)

            #make context for rerank stage, 50256 is the token_id for </endoftext/>
            sep_token = torch.ones(size = [candidate_token_ids.shape[0], 1], dtype = torch.long, device=input_ids.device) * 50256
            candidate_context_ids = torch.cat([sep_token, candidate_token_ids], -1)
       
            token_type_ids = torch.ones_like(candidate_context_ids)
            candidate_position_id = torch.ones_like(candidate_context_ids, dtype = torch.long, device=input_ids.device) * rerank_places[i+1]
            #get output from gpt2
            rerank_outputs = self.transformer(candidate_context_ids,
                                              past_key_values=past_key_values,
                                              token_type_ids=token_type_ids,
                                              position_ids = candidate_position_id
                          )

            #get rerank logits for candidates
            rerank_hidden_states = rerank_outputs[0][:, 1:]

            if not is_training:
                all_rerank_hidden_states.append(rerank_hidden_states[check_in_index])
                no_rerank_logits.append(candidate_token_logits[check_in_index])
            else:
                all_rerank_hidden_states.append(rerank_hidden_states)
                no_rerank_logits.append(candidate_token_logits)
        
#         print("\n batch info:")
#         print("there are ", check_out_num/(self.num_of_rerank*batch_size), "labels not in candidates")
        
        #-------------------------------------------------------------------------
        # cal loss, loss = normal loss + rerank loss
        loss_fct = CrossEntropyLoss(reduction='none')
        
        # cal rerank loss
        rerank_loss = None
        
        all_rerank_hidden_states = torch.cat(all_rerank_hidden_states, 0)
        all_rerank_logits = self.rerank_linear_head(all_rerank_hidden_states)
        all_rerank_logits = torch.reshape(all_rerank_logits, [-1, self.CAN_NUM])
        all_rerank_labels = torch.cat(all_rerank_labels, 0)
        
        rerank_loss = loss_fct(all_rerank_logits, all_rerank_labels)
        
                # cal normal loss
        normal_loss = None

        hidden_states = torch.cat(hidden_states, 1)
        lm_logits = self.lm_head(hidden_states)

        # Shift so that tokens < n predict n
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        normal_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        # cal normal loss in rerank place
        normal_loss_in_rerank_place_across_all_vocab = None
        normal_loss_in_rerank_place_across_candidates = None

        hidden_states_in_rerank_place = torch.cat(hidden_states_in_rerank_place, 0)
        lm_logits_in_rerank_place = self.lm_head(hidden_states_in_rerank_place)
        lm_logits_in_rerank_place = torch.reshape(lm_logits_in_rerank_place, [-1, self.VOCAB_SIZE])
        labels_in_rerank_place = torch.cat(labels_in_rerank_place, 0)

        no_rerank_logits = torch.cat(no_rerank_logits, 0)
        no_rerank_logits = torch.reshape(no_rerank_logits, [-1, self.CAN_NUM])

        stage1_loss_in_rerank_place_across_all_vocab = loss_fct(lm_logits_in_rerank_place, labels_in_rerank_place)
        stage1_loss_in_rerank_place_across_candidates = loss_fct(no_rerank_logits, all_rerank_labels)
        
        return {"stage1_loss_in_rerank_place_across_all_vocab": stage1_loss_in_rerank_place_across_all_vocab,
                "stage1_loss_in_rerank_place_across_candidates": stage1_loss_in_rerank_place_across_candidates,
                "normal_loss": normal_loss,
                "rerank_loss": rerank_loss,}
    
class rerankGPT2LMHeadModel_labelInCandidateWhenTraning_prior_probability_sharing_head(GPT2LMHeadModel):
    def __init__(self, config, MAX_LEN, CAN_NUM, num_of_rerank):
        super().__init__(config)
        self.MAX_LEN = MAX_LEN
        self.CAN_NUM = CAN_NUM
        self.num_of_rerank = num_of_rerank
        self.VOCAB_SIZE = config.vocab_size
        
        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.init_weights()
    def forward(
        self,
        input_ids=None,
        labels=None,
        is_training=False,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            ``labels = input_ids`` Indices are selected in ``[-100, 0, ..., config.vocab_size]`` All labels set to
            ``-100`` are ignored (masked), the loss is only computed for labels in ``[0, ..., config.vocab_size]``
        """        
        # rerank_places = random.sample(np.arange(1, MAX_LEN-2-CAN_NUM*2).tolist(), k=num_of_rerank) #no duplicate
        rerank_places = random.sample(np.arange(1, self.MAX_LEN).tolist(), k=self.num_of_rerank) #no duplicate
        rerank_places = np.concatenate(([0], np.sort(rerank_places), [self.MAX_LEN])) #add first and last tokens to make segments
        
        past_key_values = None
        hidden_states_in_rerank_place = []
        labels_in_rerank_place = []
        all_rerank_hidden_states = []
        all_rerank_labels = []
        hidden_states = []
        
        no_rerank_logits = []
        
        all_candidate_token_ids = []
        check_out_num = 0
        for i in range(self.num_of_rerank+1):
            #normal stage
            segment_input_ids = input_ids[:, rerank_places[i]:rerank_places[i+1]]

            segment_outputs = self.transformer(
                segment_input_ids,
                past_key_values = past_key_values,
                token_type_ids = torch.zeros_like(segment_input_ids)
            )

            segment_hidden = segment_outputs[0]
            past_key_values = segment_outputs[1]

            hidden_states.append(segment_hidden)
            #rerank stage (just for rerank places)
            if i == self.num_of_rerank:
                break

            #rerank stage
            #get logits in rerank place
            logits_before_rerank = self.lm_head(segment_hidden[:, -1, :])
            #get candidate token ids according to the logits
            candidate_token_logits, candidate_token_ids = torch.topk(logits_before_rerank, self.CAN_NUM)
            rerank_labels = labels[..., rerank_places[i+1]]
            labels_in_rerank_place.append(rerank_labels)
            hidden_states_in_rerank_place.append(segment_hidden[:, -1, :])
            
            #randomize the order of candidates
            shuffle_index = np.random.permutation(self.CAN_NUM)
            candidate_token_logits = candidate_token_logits[:, shuffle_index]
            candidate_token_ids = candidate_token_ids[:, shuffle_index]
            
            #check whether or not label in candidates
            check_labels = rerank_labels.tolist()
            check_candidates = candidate_token_ids.tolist()
            
            assert len(check_labels)==len(check_candidates)
            
            
            #when training, check whether or not label in candidates, if not we add label into candidates
            if is_training:
                #check every data in batch
                rerank_labels_this_place = []
                for j in range(len(check_labels)):
                    if check_labels[j] not in check_candidates[j]:
                        check_out_num+=1
                        replace_index = np.random.randint(self.CAN_NUM)
                        candidate_token_ids[j][replace_index] = check_labels[j]
                        rerank_labels_this_place.append(replace_index)          
                    else:
                        rerank_labels_this_place.append(check_candidates[j].index(check_labels[j]))
                all_rerank_labels.append(torch.tensor(rerank_labels_this_place, device=input_ids.device))
            #when eval, check whether or not label in candidates, if not we do not do rerank
            else:
                rerank_labels = []
                check_in_index = []

                for j in range(len(check_labels)): 
                    if check_labels[j] in check_candidates[j]:
                        rerank_labels.append(check_candidates[j].index(check_labels[j]))
                        check_in_index.append(j)
                    else:
                        check_out_num+=1
                rerank_labels = torch.tensor(rerank_labels, device=input_ids.device)
                    
                if rerank_labels.shape[0] == 0:
                    continue
                else:
                    all_rerank_labels.append(rerank_labels)

            #make context for rerank stage, 50256 is the token_id for </endoftext/>
            sep_token = torch.ones(size = [candidate_token_ids.shape[0], 1], dtype = torch.long, device=input_ids.device) * 50256
            candidate_context_ids = torch.cat([sep_token, candidate_token_ids], -1)
       
            token_type_ids = torch.ones_like(candidate_context_ids)
            candidate_position_id = torch.ones_like(candidate_context_ids, dtype = torch.long, device=input_ids.device) * rerank_places[i+1]
            #get output from gpt2
            rerank_outputs = self.transformer(candidate_context_ids,
                                              past_key_values=past_key_values,
                                              token_type_ids=token_type_ids,
                                              position_ids = candidate_position_id
                          )

            #get rerank logits for candidates
            rerank_hidden_states = rerank_outputs[0][:, 1:]

            if not is_training:
                all_rerank_hidden_states.append(rerank_hidden_states[check_in_index])
                no_rerank_logits.append(candidate_token_logits[check_in_index])
                all_candidate_token_ids.append(candidate_token_ids[check_in_index])
            else:
                all_rerank_hidden_states.append(rerank_hidden_states)
                no_rerank_logits.append(candidate_token_logits)
                all_candidate_token_ids.append(candidate_token_ids)
        
#         print("\n batch info:")
#         print("there are ", check_out_num/(self.num_of_rerank*batch_size), "labels not in candidates")
        
        #-------------------------------------------------------------------------
        # cal loss, loss = normal loss + rerank loss
        loss_fct = CrossEntropyLoss(reduction='none')
        
        # cal rerank loss
        rerank_loss = None

        all_rerank_hidden_states = torch.cat(all_rerank_hidden_states, 0)
        all_candidate_token_ids = torch.flatten(torch.cat(all_candidate_token_ids, 0))
        rerank_linear_head = self.lm_head.weight[all_candidate_token_ids]
        all_rerank_logits = []

        all_rerank_hidden_states = torch.reshape(all_rerank_hidden_states, [-1, all_rerank_hidden_states.shape[-1]])
        for i in range(rerank_linear_head.shape[0]):
            all_rerank_logits.append(torch.matmul(all_rerank_hidden_states[i], rerank_linear_head[i].T))
        all_rerank_logits = torch.reshape(torch.stack(all_rerank_logits), [-1, self.CAN_NUM]) 

        all_rerank_labels = torch.cat(all_rerank_labels, 0)

        rerank_loss = loss_fct(all_rerank_logits, all_rerank_labels)
        
        # cal normal loss
        normal_loss = None

        hidden_states = torch.cat(hidden_states, 1)
        lm_logits = self.lm_head(hidden_states)

        # Shift so that tokens < n predict n
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        normal_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        # cal normal loss in rerank place
        normal_loss_in_rerank_place_across_all_vocab = None
        normal_loss_in_rerank_place_across_candidates = None

        hidden_states_in_rerank_place = torch.cat(hidden_states_in_rerank_place, 0)
        lm_logits_in_rerank_place = self.lm_head(hidden_states_in_rerank_place)
        lm_logits_in_rerank_place = torch.reshape(lm_logits_in_rerank_place, [-1, self.VOCAB_SIZE])
        labels_in_rerank_place = torch.cat(labels_in_rerank_place, 0)

        no_rerank_logits = torch.cat(no_rerank_logits, 0)
        no_rerank_logits = torch.reshape(no_rerank_logits, [-1, self.CAN_NUM])

        stage1_loss_in_rerank_place_across_all_vocab = loss_fct(lm_logits_in_rerank_place, labels_in_rerank_place)
        stage1_loss_in_rerank_place_across_candidates = loss_fct(no_rerank_logits, all_rerank_labels)
        
        return {"stage1_loss_in_rerank_place_across_all_vocab": stage1_loss_in_rerank_place_across_all_vocab,
                "stage1_loss_in_rerank_place_across_candidates": stage1_loss_in_rerank_place_across_candidates,
                "normal_loss": normal_loss,
                "rerank_loss": rerank_loss,}
    
class rerankGPT2LMHeadModel_stage1_all_tokens_stage2_sample_tokens_prior_probability_sharing_head(GPT2LMHeadModel):
    def __init__(self, config, MAX_LEN, CAN_NUM, num_of_rerank):
        super().__init__(config)
        self.MAX_LEN = MAX_LEN
        self.CAN_NUM = CAN_NUM
        self.num_of_rerank = num_of_rerank
        self.VOCAB_SIZE = config.vocab_size
        
        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.init_weights()
    def forward(
        self,
        input_ids=None,
        labels=None,
        is_training=False,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            ``labels = input_ids`` Indices are selected in ``[-100, 0, ..., config.vocab_size]`` All labels set to
            ``-100`` are ignored (masked), the loss is only computed for labels in ``[0, ..., config.vocab_size]``
        """
        # make some model parameter not change during rerank (like dropout) ??????????????
        # model.eval()
        if is_training:
            self.transformer.eval()

        rerank_places = random.sample(np.arange(1, self.MAX_LEN).tolist(), k=self.num_of_rerank) #no duplicate
        rerank_places = np.concatenate(([0], np.sort(rerank_places), [self.MAX_LEN])) #add first and last tokens to make segments
        
        past_key_values = None
        all_rerank_hidden_states = []
        all_rerank_labels = []
        no_rerank_logits = []
        check_out_num = 0
        
        labels_in_rerank_place = []
        hidden_states_in_rerank_place = []
        
        hidden_states = []
        
        all_candidate_token_ids = []
        
        for i in range(self.num_of_rerank+1):
            #normal stage
            segment_input_ids = input_ids[:, rerank_places[i]:rerank_places[i+1]]

            segment_outputs = self.transformer(
                segment_input_ids,
                past_key_values = past_key_values,
                token_type_ids = torch.zeros_like(segment_input_ids)
            )

            segment_hidden = segment_outputs[0]
            past_key_values = segment_outputs[1]

            hidden_states.append(segment_hidden)
            #rerank stage (just for rerank places)
            if i == self.num_of_rerank:
                break

            #rerank stage
            #get logits in rerank place
            logits_before_rerank = self.lm_head(segment_hidden[:, -1, :])
            #get candidate token ids according to the logits
            candidate_token_logits, candidate_token_ids = torch.topk(logits_before_rerank, self.CAN_NUM)
            rerank_labels = labels[..., rerank_places[i+1]]
            
            labels_in_rerank_place.append(rerank_labels)
            hidden_states_in_rerank_place.append(segment_hidden[:, -1, :])
            
            #randomize the order of candidates
            shuffle_index = np.random.permutation(self.CAN_NUM)
            candidate_token_logits = candidate_token_logits[:, shuffle_index]
            candidate_token_ids = candidate_token_ids[:, shuffle_index]
            
            #check whether or not label in candidates
            check_labels = rerank_labels.tolist()
            check_candidates = candidate_token_ids.tolist()
            
            assert len(check_labels)==len(check_candidates)
            
            #check whether or not label in candidates, if not we do not do rerank
            rerank_labels = []
            check_in_index = []

            for j in range(len(check_labels)): 
                if check_labels[j] in check_candidates[j]:
                    rerank_labels.append(check_candidates[j].index(check_labels[j]))
                    check_in_index.append(j)
                else:
                    check_out_num+=1
            rerank_labels = torch.tensor(rerank_labels, device=input_ids.device)

            if rerank_labels.shape[0] == 0:
                continue
            else:
                all_rerank_labels.append(rerank_labels)

            #make context for rerank stage, 50256 is the token_id for </endoftext/>
            sep_token = torch.ones(size = [candidate_token_ids.shape[0], 1], dtype = torch.long, device=input_ids.device) * 50256
            candidate_context_ids = torch.cat([sep_token, candidate_token_ids], -1)
       
            token_type_ids = torch.ones_like(candidate_context_ids)
            candidate_position_id = torch.ones_like(candidate_context_ids, dtype = torch.long, device=input_ids.device) * rerank_places[i+1]
            #get output from gpt2
            rerank_outputs = self.transformer(candidate_context_ids,
                                              past_key_values=past_key_values,
                                              token_type_ids=token_type_ids,
                                              position_ids = candidate_position_id
                          )

            #get rerank logits for candidates
            rerank_hidden_states = rerank_outputs[0][:, 1:]

            all_rerank_hidden_states.append(rerank_hidden_states[check_in_index])
            no_rerank_logits.append(candidate_token_logits[check_in_index])
            all_candidate_token_ids.append(candidate_token_ids[check_in_index])
            
        if is_training:
            #model.train()
            self.transformer.train()
        
        #-------------------------------------------------------------------------
        # cal loss, loss = normal loss + rerank loss
        loss_fct = CrossEntropyLoss(reduction='none')
        
        # cal rerank loss
        rerank_loss = None

        all_rerank_hidden_states = torch.cat(all_rerank_hidden_states, 0)
        all_candidate_token_ids = torch.flatten(torch.cat(all_candidate_token_ids, 0))
        rerank_linear_head = self.lm_head.weight[all_candidate_token_ids]
        all_rerank_logits = []

        all_rerank_hidden_states = torch.reshape(all_rerank_hidden_states, [-1, all_rerank_hidden_states.shape[-1]])
        for i in range(rerank_linear_head.shape[0]):
            all_rerank_logits.append(torch.matmul(all_rerank_hidden_states[i], rerank_linear_head[i].T))
        all_rerank_logits = torch.reshape(torch.stack(all_rerank_logits), [-1, self.CAN_NUM]) 

        all_rerank_labels = torch.cat(all_rerank_labels, 0)

        rerank_loss = loss_fct(all_rerank_logits, all_rerank_labels)
        
        stage1_loss_in_rerank_place_across_candidates = None


        no_rerank_logits = torch.cat(no_rerank_logits, 0)
        no_rerank_logits = torch.reshape(no_rerank_logits, [-1, self.CAN_NUM])

        stage1_loss_in_rerank_place_across_candidates = loss_fct(no_rerank_logits, all_rerank_labels)
        
        stage1_loss_in_rerank_place_across_all_vocab = None
        
        hidden_states_in_rerank_place = torch.cat(hidden_states_in_rerank_place, 0)
        lm_logits_in_rerank_place = self.lm_head(hidden_states_in_rerank_place)
        lm_logits_in_rerank_place = torch.reshape(lm_logits_in_rerank_place, [-1, self.VOCAB_SIZE])
        labels_in_rerank_place = torch.cat(labels_in_rerank_place, 0)

        stage1_loss_in_rerank_place_across_all_vocab = loss_fct(lm_logits_in_rerank_place, labels_in_rerank_place)

        # cal normal loss
        normal_loss = None

        hidden_states = torch.cat(hidden_states, 1)
        lm_logits = self.lm_head(hidden_states)

        # Shift so that tokens < n predict n
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        normal_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return {"stage1_loss_in_rerank_place_across_all_vocab": stage1_loss_in_rerank_place_across_all_vocab,
                "stage1_loss_in_rerank_place_across_candidates": stage1_loss_in_rerank_place_across_candidates,
                "normal_loss": normal_loss,
                "rerank_loss": rerank_loss,}
    
class rerankGPT2LMHeadModel_stage1_all_tokens_stage2_all_tokens(GPT2LMHeadModel):
    def __init__(self, config, MAX_LEN, CAN_NUM, num_of_rerank):
        super().__init__(config)
        self.MAX_LEN = MAX_LEN
        self.CAN_NUM = CAN_NUM
        self.num_of_rerank = num_of_rerank
        self.VOCAB_SIZE = config.vocab_size
        
        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        self.rerank_linear_head = nn.Linear(config.n_embd, config.n_embd, bias=False)

        self.init_weights()
        self.rerank_linear_head.weight.data.normal_(mean=0.0, std=0.0002)
        
    @property
    def wte(self):
        """
        Get the weights for the token embeddings from the transformer
        """
        return self.transformer.wte
    
    def forward(
        self,
        input_ids=None,
        labels=None,
        is_training=False,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            ``labels = input_ids`` Indices are selected in ``[-100, 0, ..., config.vocab_size]`` All labels set to
            ``-100`` are ignored (masked), the loss is only computed for labels in ``[0, ..., config.vocab_size]``
        """
        # make some model parameter not change during rerank (like dropout) ??????????????
        # model.eval()
        if is_training:
            self.transformer.eval()

        rerank_places = random.sample(np.arange(1, self.MAX_LEN).tolist(), k=self.num_of_rerank) #no duplicate
        rerank_places = np.concatenate(([0], np.sort(rerank_places), [self.MAX_LEN])) #add first and last tokens to make segments
        
        past_key_values = None
        all_stage1_logits = []
        all_stage2_logits = []
        all_rerank_labels = []
        
        rerank_hidden_states_meganitude = []
        
        hidden_states = []
        
        for i in range(self.num_of_rerank+1):
            #normal stage
            segment_input_ids = input_ids[:, rerank_places[i]:rerank_places[i+1]]

            segment_outputs = self.transformer(
                segment_input_ids,
                past_key_values = past_key_values,
                token_type_ids = torch.zeros_like(segment_input_ids)
            )

            segment_hidden = segment_outputs[0]
            past_key_values = segment_outputs[1]

            hidden_states.append(segment_hidden)
            #rerank stage (just for rerank places)
            if i == self.num_of_rerank:
                break

            #rerank stage
            #get logits in rerank place
            stage1_hidden_states = segment_hidden[:, -1, :]
            stage1_logits = self.lm_head(stage1_hidden_states)
            #get candidate token ids according to the logits
            candidate_token_logits, candidate_token_ids = torch.topk(stage1_logits, self.CAN_NUM)
            rerank_labels = labels[..., rerank_places[i+1]]

            all_stage1_logits.append(stage1_logits)
            all_rerank_labels.append(rerank_labels)

            #make context for rerank stage, 50256 is the token_id for </endoftext/>
            sep_token = torch.ones(size = [candidate_token_ids.shape[0], 1], dtype = torch.long, device=input_ids.device) * 50256
            candidate_context_ids = torch.cat([sep_token, sep_token, candidate_token_ids], -1)
            candidate_context_embeds = self.wte(candidate_context_ids)
            candidate_context_embeds[:, 1, :] = stage1_hidden_states
       
            token_type_ids = torch.ones_like(candidate_context_ids)
            candidate_position_id = torch.ones_like(candidate_context_ids, dtype = torch.long, device=input_ids.device) * rerank_places[i+1]
            #get output from gpt2
            
            rerank_outputs = self.transformer(inputs_embeds=candidate_context_embeds,
                                              past_key_values=past_key_values,
                                              token_type_ids=token_type_ids,
                                              position_ids = candidate_position_id
                          )

            #get rerank logits for candidates
            rerank_hidden_states = self.rerank_linear_head(rerank_outputs[0][:, 2:])
            rerank_hidden_states_meganitude.append(rerank_hidden_states.norm(dim=-1).mean())
            
            stage2_logits = stage1_logits.clone()
            
            stage2_candidates_logits = torch.matmul(rerank_hidden_states, stage1_hidden_states.unsqueeze(-1)).squeeze(-1)
#             stage2_logits = stage2_logits.scatter(1, candidate_token_ids, stage2_candidates_logits)
            stage2_logits = stage2_logits.scatter_add(1, candidate_token_ids, stage2_candidates_logits)
            
            all_stage2_logits.append(stage2_logits)
            
        if is_training:
            #model.train()
            self.transformer.train()
        
        #-------------------------------------------------------------------------
        rerank_hidden_states_meganitude = torch.stack(rerank_hidden_states_meganitude, 0)
        
        # cal loss, loss = normal loss + rerank loss
        loss_fct = CrossEntropyLoss(reduction='none')

        # cal rerank loss
        rerank_loss = None
        
        all_stage2_logits = torch.cat(all_stage2_logits, 0)
        all_rerank_labels = torch.cat(all_rerank_labels, 0)

        rerank_loss = loss_fct(all_stage2_logits, all_rerank_labels)
        
        # cal normal loss in rerank place (for comparision with rerank results), only evaluate
        normal_loss_in_rerank_place = None

        all_stage1_logits = torch.cat(all_stage1_logits, 0)

        normal_loss_in_rerank_place = loss_fct(all_stage1_logits, all_rerank_labels)

        # cal normal loss
        normal_loss = None

        hidden_states = torch.cat(hidden_states, 1)
        lm_logits = self.lm_head(hidden_states)

        # Shift so that tokens < n predict n
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        normal_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return {"normal_loss": normal_loss,
                "rerank_loss": rerank_loss,
                "stage1_loss_in_rerank_place_across_all_vocab": normal_loss_in_rerank_place,
                "rerank_hidden_states_meganitude": rerank_hidden_states_meganitude}
    
class rerankGPT2LMHeadModel_stage1_all_tokens_stage2_all_tokens_use_different_layer(GPT2LMHeadModel):
    def __init__(self, config, MAX_LEN, CAN_NUM, num_of_rerank):
        super().__init__(config)
        self.MAX_LEN = MAX_LEN
        self.CAN_NUM = CAN_NUM
        self.num_of_rerank = num_of_rerank
        self.VOCAB_SIZE = config.vocab_size
        
        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        self.rerank_linear_head = nn.Linear(config.n_embd, config.n_embd, bias=False)

        self.init_weights()
        self.rerank_linear_head.weight.data.normal_(mean=0.0, std=0.0002)
        
    @property
    def wte(self):
        """
        Get the weights for the token embeddings from the transformer
        """
        return self.transformer.wte
    
    def forward(
        self,
        input_ids=None,
        labels=None,
        is_training=False,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            ``labels = input_ids`` Indices are selected in ``[-100, 0, ..., config.vocab_size]`` All labels set to
            ``-100`` are ignored (masked), the loss is only computed for labels in ``[0, ..., config.vocab_size]``
        """
        # make some model parameter not change during rerank (like dropout) ??????????????
        # model.eval()
        if is_training:
            self.transformer.eval()

        rerank_places = random.sample(np.arange(1, self.MAX_LEN).tolist(), k=self.num_of_rerank) #no duplicate
        rerank_places = np.concatenate(([0], np.sort(rerank_places), [self.MAX_LEN])) #add first and last tokens to make segments
        
        past_key_values = None
        all_stage1_logits = []
        all_stage2_logits = []
        all_rerank_labels = []
        
        rerank_hidden_states_meganitude = []
        
        hidden_states = []
        
        for i in range(self.num_of_rerank+1):
            #normal stage
            segment_input_ids = input_ids[:, rerank_places[i]:rerank_places[i+1]]

            segment_outputs = self.transformer(
                segment_input_ids,
                past_key_values = past_key_values,
                token_type_ids = torch.zeros_like(segment_input_ids)
            )

            segment_hidden = segment_outputs[0]
            past_key_values = segment_outputs[1]

            hidden_states.append(segment_hidden)
            #rerank stage (just for rerank places)
            if i == self.num_of_rerank:
                break

            #rerank stage
            #get logits in rerank place
            stage1_hidden_states = segment_hidden[:, -1, :]
            stage1_logits = self.lm_head(stage1_hidden_states)
            #get candidate token ids according to the logits
            candidate_token_logits, candidate_token_ids = torch.topk(stage1_logits, self.CAN_NUM)
            rerank_labels = labels[..., rerank_places[i+1]]

            all_stage1_logits.append(stage1_logits)
            all_rerank_labels.append(rerank_labels)

            #make context for rerank stage, 50256 is the token_id for </endoftext/>
            sep_token = torch.ones(size = [candidate_token_ids.shape[0], 1], dtype = torch.long, device=input_ids.device) * 50256
            candidate_context_ids = torch.cat([sep_token, sep_token, candidate_token_ids], -1)
            candidate_context_embeds = self.wte(candidate_context_ids)
            candidate_context_embeds[:, 1, :] = stage1_hidden_states
       
            token_type_ids = torch.ones_like(candidate_context_ids)
            candidate_position_id = torch.ones_like(candidate_context_ids, dtype = torch.long, device=input_ids.device) * rerank_places[i+1]
            #get output from gpt2
            
            rerank_outputs = self.transformer(inputs_embeds=candidate_context_embeds,
                                              past_key_values=past_key_values,
                                              token_type_ids=token_type_ids,
                                              position_ids = candidate_position_id,
                                              output_hidden_states = True
                          )

            #get rerank logits for candidates
#             rerank_hidden_states = rerank_outputs[0][:, 2:] #use last layer hidden states
#             rerank_hidden_states = rerank_outputs.hidden_states[-5][:, 2:] #use 8th layer hidden states
            rerank_hidden_states = rerank_outputs.hidden_states[-7][:, 2:] #use 6th layer hidden states

            #get rerank logits for candidates
            rerank_hidden_states = self.rerank_linear_head(rerank_hidden_states)
            rerank_hidden_states_meganitude.append(rerank_hidden_states.norm(dim=-1).mean())
            
            stage2_logits = stage1_logits.clone()
            
            stage2_candidates_logits = torch.matmul(rerank_hidden_states, stage1_hidden_states.unsqueeze(-1)).squeeze(-1)
            stage2_logits = stage2_logits.scatter_add(1, candidate_token_ids, stage2_candidates_logits)
            
            all_stage2_logits.append(stage2_logits)
            
        if is_training:
            #model.train()
            self.transformer.train()
        
        #-------------------------------------------------------------------------
        rerank_hidden_states_meganitude = torch.stack(rerank_hidden_states_meganitude, 0)
        
        # cal loss, loss = normal loss + rerank loss
        loss_fct = CrossEntropyLoss(reduction='none')

        # cal rerank loss
        rerank_loss = None
        
        all_stage2_logits = torch.cat(all_stage2_logits, 0)
        all_rerank_labels = torch.cat(all_rerank_labels, 0)

        rerank_loss = loss_fct(all_stage2_logits, all_rerank_labels)
        
        # cal normal loss in rerank place (for comparision with rerank results), only evaluate
        normal_loss_in_rerank_place = None

        all_stage1_logits = torch.cat(all_stage1_logits, 0)

        normal_loss_in_rerank_place = loss_fct(all_stage1_logits, all_rerank_labels)

        # cal normal loss
        normal_loss = None

        hidden_states = torch.cat(hidden_states, 1)
        lm_logits = self.lm_head(hidden_states)

        # Shift so that tokens < n predict n
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        normal_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return {"normal_loss": normal_loss,
                "rerank_loss": rerank_loss,
                "stage1_loss_in_rerank_place_across_all_vocab": normal_loss_in_rerank_place,
                "rerank_hidden_states_meganitude": rerank_hidden_states_meganitude}