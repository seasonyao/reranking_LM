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

    
class rerankGPT2LMHeadModel_not_repeat_candidate(GPT2LMHeadModel):
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
        
        if not is_training:
            all_candidate_token_ids = []
            all_input_ids = []
            all_prediction_ids = []
        
        for i in range(self.num_of_rerank+1):
            #normal stage
            segment_input_ids = input_ids[:, rerank_places[i]:rerank_places[i+1]]

            segment_outputs = self.transformer(
                segment_input_ids,
                past_key_values = past_key_values
            )

            segment_hidden = segment_outputs[0]
            past_key_values = segment_outputs[1]

            #rerank stage (just for rerank places)
            if i == self.num_of_rerank:
                break

            #rerank stage
            #get logits in rerank place
            logits_before_rerank = self.lm_head(segment_hidden[:, -1, :])
            #get candidate token ids according to the logits
            candidate_token_logits, candidate_token_ids = torch.topk(logits_before_rerank, self.CAN_NUM)
            rerank_labels = labels[..., rerank_places[i+1]]
            
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
            #candidate_context_ids = torch.cat([sep_token, candidate_token_ids, sep_token, candidate_token_ids], -1)
            candidate_context_ids = torch.cat([sep_token, candidate_token_ids], -1)

            #get output from gpt2
            rerank_outputs = self.transformer(candidate_context_ids,
                            past_key_values=past_key_values,
                          )

            #get rerank logits for candidates
            rerank_hidden_states = rerank_outputs[0][:, 1:]

            all_rerank_hidden_states.append(rerank_hidden_states[check_in_index])
            no_rerank_logits.append(candidate_token_logits[check_in_index])
            
            if not is_training:
                all_candidate_token_ids.append(candidate_token_ids[check_in_index])
                all_prediction_ids.append(input_ids[:, rerank_places[i+1]][check_in_index])
                all_input_ids.append(input_ids[:, :rerank_places[i+1]][check_in_index])

        
#         print("\n batch info:")
#         print("there are ", check_out_num/(self.num_of_rerank*batch_size), "labels not in candidates")
    
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

            # cal normal loss in rerank place (for comparision with rerank results), only evaluate
            normal_loss_in_rerank_place = None


            no_rerank_logits = torch.cat(no_rerank_logits, 0)
            no_rerank_logits = torch.reshape(no_rerank_logits, [-1, self.CAN_NUM])
            #no_rerank_labels = torch.cat(all_rerank_labels, 0) #no_rerank_labels == all_rerank_labels

            normal_loss_in_rerank_place = loss_fct(no_rerank_logits, all_rerank_labels)


            return {"normal_loss_in_rerank_place": normal_loss_in_rerank_place,
                    "rerank_loss": rerank_loss,}
        # for evaluation, we will evaluate the model's performance on different difficult level
        else:
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

            # cal normal loss in rerank place (for comparision with rerank results), only evaluate
            normal_loss_in_rerank_place = None
            no_rerank_logits = torch.cat(no_rerank_logits, 0)
            no_rerank_logits = torch.reshape(no_rerank_logits, [-1, self.CAN_NUM])
            #no_rerank_labels = torch.cat(all_rerank_labels, 0) #no_rerank_labels == all_rerank_labels

            normal_loss_in_rerank_place = loss_fct(no_rerank_logits, all_rerank_labels)
            
            for i in range(len(all_input_ids)):
                target = torch.ones(size = [all_input_ids[i].shape[0], self.MAX_LEN], dtype = torch.long, device=input_ids.device) * 50256
                target[:, :all_input_ids[i].shape[1]] = all_input_ids[i]
                all_input_ids[i] = target
                
            all_input_ids = torch.cat(all_input_ids, 0)
            all_prediction_ids = torch.cat(all_prediction_ids, 0)
            all_candidate_token_ids = torch.cat(all_candidate_token_ids, 0)

            return {"all_rerank_logits": all_rerank_logits,
                    "no_rerank_logits": no_rerank_logits,
                    "difficult_level": all_rerank_labels,
                    "rerank_loss": rerank_loss,
                    "normal_loss_in_rerank_place": normal_loss_in_rerank_place,
                    "all_candidate_token_ids": all_candidate_token_ids,
                    "all_prediction_ids": all_prediction_ids,
                    "all_input_ids": all_input_ids}
        
        
class rerankGPT2LMHeadModel_randomize_candidates_order(GPT2LMHeadModel):
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
        
        if not is_training:
            all_candidate_token_ids = []
            all_input_ids = []
            all_prediction_ids = []
        
        for i in range(self.num_of_rerank+1):
            #normal stage
            segment_input_ids = input_ids[:, rerank_places[i]:rerank_places[i+1]]

            segment_outputs = self.transformer(
                segment_input_ids,
                past_key_values = past_key_values
            )

            segment_hidden = segment_outputs[0]
            past_key_values = segment_outputs[1]

            #rerank stage (just for rerank places)
            if i == self.num_of_rerank:
                break

            #rerank stage
            #get logits in rerank place
            logits_before_rerank = self.lm_head(segment_hidden[:, -1, :])
            #get candidate token ids according to the logits
            candidate_token_logits, candidate_token_ids = torch.topk(logits_before_rerank, self.CAN_NUM)
            rerank_labels = labels[..., rerank_places[i+1]]
            
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

            #get output from gpt2
            rerank_outputs = self.transformer(candidate_context_ids,
                            past_key_values=past_key_values,
                          )

            #get rerank logits for candidates
            rerank_hidden_states = rerank_outputs[0][:, 1:]

            all_rerank_hidden_states.append(rerank_hidden_states[check_in_index])
            no_rerank_logits.append(candidate_token_logits[check_in_index])
            
            if not is_training:
                all_candidate_token_ids.append(candidate_token_ids[check_in_index])
                all_prediction_ids.append(input_ids[:, rerank_places[i+1]][check_in_index])
                all_input_ids.append(input_ids[:, :rerank_places[i+1]][check_in_index])

        
#         print("\n batch info:")
#         print("there are ", check_out_num/(self.num_of_rerank*batch_size), "labels not in candidates")
    
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

            # cal normal loss in rerank place (for comparision with rerank results), only evaluate
            normal_loss_in_rerank_place = None


            no_rerank_logits = torch.cat(no_rerank_logits, 0)
            no_rerank_logits = torch.reshape(no_rerank_logits, [-1, self.CAN_NUM])
            #no_rerank_labels = torch.cat(all_rerank_labels, 0) #no_rerank_labels == all_rerank_labels

            normal_loss_in_rerank_place = loss_fct(no_rerank_logits, all_rerank_labels)


            return {"normal_loss_in_rerank_place": normal_loss_in_rerank_place,
                    "rerank_loss": rerank_loss,}
        # for evaluation, we will evaluate the model's performance on different difficult level
        else:
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

            # cal normal loss in rerank place (for comparision with rerank results), only evaluate
            normal_loss_in_rerank_place = None
            no_rerank_logits = torch.cat(no_rerank_logits, 0)
            no_rerank_logits = torch.reshape(no_rerank_logits, [-1, self.CAN_NUM])
            #no_rerank_labels = torch.cat(all_rerank_labels, 0) #no_rerank_labels == all_rerank_labels

            normal_loss_in_rerank_place = loss_fct(no_rerank_logits, all_rerank_labels)
            
            for i in range(len(all_input_ids)):
                target = torch.ones(size = [all_input_ids[i].shape[0], self.MAX_LEN], dtype = torch.long, device=input_ids.device) * 50256
                target[:, :all_input_ids[i].shape[1]] = all_input_ids[i]
                all_input_ids[i] = target
                
            all_input_ids = torch.cat(all_input_ids, 0)
            all_prediction_ids = torch.cat(all_prediction_ids, 0)
            all_candidate_token_ids = torch.cat(all_candidate_token_ids, 0)

            return {"all_rerank_logits": all_rerank_logits,
                    "no_rerank_logits": no_rerank_logits,
                    "difficult_level": all_rerank_labels,
                    "rerank_loss": rerank_loss,
                    "normal_loss_in_rerank_place": normal_loss_in_rerank_place,
                    "all_candidate_token_ids": all_candidate_token_ids,
                    "all_prediction_ids": all_prediction_ids,
                    "all_input_ids": all_input_ids}
        
        
class rerankGPT2LMHeadModel_token_type_embeddings(GPT2LMHeadModel):
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
        
        if not is_training:
            all_candidate_token_ids = []
            all_input_ids = []
            all_prediction_ids = []
        
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

            #rerank stage (just for rerank places)
            if i == self.num_of_rerank:
                break

            #rerank stage
            #get logits in rerank place
            logits_before_rerank = self.lm_head(segment_hidden[:, -1, :])
            #get candidate token ids according to the logits
            candidate_token_logits, candidate_token_ids = torch.topk(logits_before_rerank, self.CAN_NUM)
            rerank_labels = labels[..., rerank_places[i+1]]
            
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
            #candidate_context_ids = torch.cat([sep_token, candidate_token_ids, sep_token, candidate_token_ids], -1)
            candidate_context_ids = torch.cat([sep_token, candidate_token_ids], -1)

            token_type_ids = torch.ones_like(candidate_context_ids)
            #get output from gpt2
            rerank_outputs = self.transformer(candidate_context_ids,
                                              past_key_values=past_key_values,
                                              token_type_ids=token_type_ids)

            #get rerank logits for candidates
            rerank_hidden_states = rerank_outputs[0][:, 1:]

            all_rerank_hidden_states.append(rerank_hidden_states[check_in_index])
            no_rerank_logits.append(candidate_token_logits[check_in_index])
            
            if not is_training:
                all_candidate_token_ids.append(candidate_token_ids[check_in_index])
                all_prediction_ids.append(input_ids[:, rerank_places[i+1]][check_in_index])
                all_input_ids.append(input_ids[:, :rerank_places[i+1]][check_in_index])

        
#         print("\n batch info:")
#         print("there are ", check_out_num/(self.num_of_rerank*batch_size), "labels not in candidates")
    
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

            # cal normal loss in rerank place (for comparision with rerank results), only evaluate
            normal_loss_in_rerank_place = None


            no_rerank_logits = torch.cat(no_rerank_logits, 0)
            no_rerank_logits = torch.reshape(no_rerank_logits, [-1, self.CAN_NUM])
            #no_rerank_labels = torch.cat(all_rerank_labels, 0) #no_rerank_labels == all_rerank_labels

            normal_loss_in_rerank_place = loss_fct(no_rerank_logits, all_rerank_labels)


            return {"normal_loss_in_rerank_place": normal_loss_in_rerank_place,
                    "rerank_loss": rerank_loss,}
        # for evaluation, we will evaluate the model's performance on different difficult level
        else:
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

            # cal normal loss in rerank place (for comparision with rerank results), only evaluate
            normal_loss_in_rerank_place = None
            no_rerank_logits = torch.cat(no_rerank_logits, 0)
            no_rerank_logits = torch.reshape(no_rerank_logits, [-1, self.CAN_NUM])
            #no_rerank_labels = torch.cat(all_rerank_labels, 0) #no_rerank_labels == all_rerank_labels

            normal_loss_in_rerank_place = loss_fct(no_rerank_logits, all_rerank_labels)
            
            for i in range(len(all_input_ids)):
                target = torch.ones(size = [all_input_ids[i].shape[0], self.MAX_LEN], dtype = torch.long, device=input_ids.device) * 50256
                target[:, :all_input_ids[i].shape[1]] = all_input_ids[i]
                all_input_ids[i] = target
                
            all_input_ids = torch.cat(all_input_ids, 0)
            all_prediction_ids = torch.cat(all_prediction_ids, 0)
            all_candidate_token_ids = torch.cat(all_candidate_token_ids, 0)

            return {"all_rerank_logits": all_rerank_logits,
                    "no_rerank_logits": no_rerank_logits,
                    "difficult_level": all_rerank_labels,
                    "rerank_loss": rerank_loss,
                    "normal_loss_in_rerank_place": normal_loss_in_rerank_place,
                    "all_candidate_token_ids": all_candidate_token_ids,
                    "all_prediction_ids": all_prediction_ids,
                    "all_input_ids": all_input_ids}