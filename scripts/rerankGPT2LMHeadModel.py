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
        if not is_training:
            no_rerank_logits = [] #no_rerank_labels is the same with all_rerank_labels, only need to be cal when eval
        check_out_num = 0
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
            labels_in_rerank_place.append(rerank_labels)
            hidden_states_in_rerank_place.append(segment_hidden[:, -1, :])
            
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
            candidate_context_ids = torch.cat([sep_token, candidate_token_ids, sep_token, candidate_token_ids], -1)
       
            #get output from gpt2
            rerank_outputs = self.transformer(candidate_context_ids,
                            past_key_values=past_key_values,
                          )

            #get rerank logits for candidates
            rerank_hidden_states = rerank_outputs[0][:, 2+self.CAN_NUM:2+self.CAN_NUM*2]

            if not is_training:
                all_rerank_hidden_states.append(rerank_hidden_states[check_in_index])
                no_rerank_logits.append(candidate_token_logits[check_in_index])
            else:
                all_rerank_hidden_states.append(rerank_hidden_states)
        
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
        
        # cal normal loss in rerank place (for comparision with rerank results), only evaluate
        normal_loss_in_rerank_place = None
        if is_training:
            # cal normal loss in rerank place (for comparision with rerank results)        
            normal_loss_in_rerank_place = None

            hidden_states_in_rerank_place = torch.cat(hidden_states_in_rerank_place, 0)
            lm_logits_in_rerank_place = self.lm_head(hidden_states_in_rerank_place)
            lm_logits_in_rerank_place = torch.reshape(lm_logits_in_rerank_place, [-1, self.VOCAB_SIZE])
            labels_in_rerank_place = torch.cat(labels_in_rerank_place, 0)

            normal_loss_in_rerank_place = loss_fct(lm_logits_in_rerank_place, labels_in_rerank_place)
        else:
            no_rerank_logits = torch.cat(no_rerank_logits, 0)
            no_rerank_logits = torch.reshape(no_rerank_logits, [-1, self.CAN_NUM])
            #no_rerank_labels = torch.cat(all_rerank_labels, 0) #no_rerank_labels == all_rerank_labels

            normal_loss_in_rerank_place = loss_fct(no_rerank_logits, all_rerank_labels)
  
        
        return {"normal_loss_in_rerank_place": normal_loss_in_rerank_place,
                "rerank_loss": rerank_loss,}
    
    
class rerankGPT2LMHeadModel_exclude_cases_label_not_in_candidates(GPT2LMHeadModel):
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
            candidate_context_ids = torch.cat([sep_token, candidate_token_ids, sep_token, candidate_token_ids], -1)

            
            #get output from gpt2
            rerank_outputs = self.transformer(candidate_context_ids,
                            past_key_values=past_key_values,
                          )

            #get rerank logits for candidates
            rerank_hidden_states = rerank_outputs[0][:, 2+self.CAN_NUM:2+self.CAN_NUM*2]

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

class rerankGPT2LMHeadModel_not_sharing_stage1_stage2(GPT2LMHeadModel):
    def __init__(self, config, MAX_LEN, CAN_NUM, num_of_rerank):
        super().__init__(config)
        self.MAX_LEN = MAX_LEN
        self.CAN_NUM = CAN_NUM
        self.num_of_rerank = num_of_rerank
        self.VOCAB_SIZE = config.vocab_size
        
        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        self.rerank_transformer = GPT2Model(config)
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
        
        stage1_past_key_values = None
        stage2_past_key_values = None
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
                past_key_values = stage1_past_key_values
            )

            segment_hidden = segment_outputs[0]
            stage1_past_key_values = segment_outputs[1]

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
            candidate_context_ids = torch.cat([sep_token, candidate_token_ids, sep_token, candidate_token_ids], -1)

            
            #get output from gpt2
            # stage1 and 2 not share, so stage2 past_key_values could not use stage1's
            stage2_past_key_values = self.rerank_transformer(
                segment_input_ids,
                past_key_values = stage2_past_key_values
            )[1]
            
            rerank_outputs = self.rerank_transformer(candidate_context_ids,
                            past_key_values=stage2_past_key_values,
                          )

            #get rerank logits for candidates
            rerank_hidden_states = rerank_outputs[0][:, 2+self.CAN_NUM:2+self.CAN_NUM*2]

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
        
        
class rerankGPT2LMHeadModel_no_stage2(GPT2LMHeadModel):
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
        all_rerank_labels = []
        no_rerank_logits = []
        check_out_num = 0
        
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

            no_rerank_logits.append(candidate_token_logits[check_in_index])         
    

        loss_fct = CrossEntropyLoss(reduction='none')

        # cal normal loss in rerank place (for comparision with rerank results), only evaluate
        normal_loss_in_rerank_place = None


        no_rerank_logits = torch.cat(no_rerank_logits, 0)
        no_rerank_logits = torch.reshape(no_rerank_logits, [-1, self.CAN_NUM])
        all_rerank_labels = torch.cat(all_rerank_labels, 0)

        normal_loss_in_rerank_place = loss_fct(no_rerank_logits, all_rerank_labels)


        return {"normal_loss_in_rerank_place": normal_loss_in_rerank_place}
        
        
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
            candidate_context_ids = torch.cat([sep_token, candidate_token_ids, sep_token, candidate_token_ids], -1)

            
            #get output from gpt2
            rerank_outputs = self.transformer(candidate_context_ids,
                            past_key_values=past_key_values,
                          )

            #get rerank logits for candidates
            rerank_hidden_states = rerank_outputs[0][:, 2+self.CAN_NUM:2+self.CAN_NUM*2]

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
        
class rerankGPT2LMHeadModel_add_candidates_predict_next_word(GPT2LMHeadModel):
    def __init__(self, config, MAX_LEN, CAN_NUM, num_of_rerank):
        super().__init__(config)
        self.MAX_LEN = MAX_LEN
        self.CAN_NUM = CAN_NUM
        self.num_of_rerank = num_of_rerank
        self.VOCAB_SIZE = config.vocab_size
        
        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        self.rerank_linear_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

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
        hidden_states_in_rerank_place = []
        labels_in_rerank_place = []
        all_rerank_hidden_states = []
        
        check_out_num = 0
        
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
                    rerank_labels.append(check_labels[j])
                    check_in_index.append(j)
                else:
                    check_out_num+=1
            rerank_labels = torch.tensor(rerank_labels, device=input_ids.device)

            if rerank_labels.shape[0] == 0:
                continue

            #make context for rerank stage, 50256 is the token_id for </endoftext/>
            sep_token = torch.ones(size = [candidate_token_ids.shape[0], 1], dtype = torch.long, device=input_ids.device) * 50256
            candidate_context_ids = torch.cat([sep_token, candidate_token_ids, sep_token, candidate_token_ids, sep_token], -1)

            
            #get output from gpt2
            rerank_outputs = self.transformer(candidate_context_ids,
                            past_key_values=past_key_values,
                          )

            #get rerank logits for next word
            rerank_hidden_states = rerank_outputs[0]
            
            all_rerank_hidden_states.append(rerank_hidden_states[:, -1, :][check_in_index])
            hidden_states_in_rerank_place.append(segment_hidden[:, -1, :][check_in_index])
            labels_in_rerank_place.append(rerank_labels)

        
#         print("\n batch info:")
#         print("there are ", check_out_num/(self.num_of_rerank*batch_size), "labels not in candidates")
    
        #model.train()
        self.transformer.train()

        #-------------------------------------------------------------------------
        # cal loss, loss = normal loss + rerank loss
        loss_fct = CrossEntropyLoss(reduction='none')

        labels_in_rerank_place = torch.cat(labels_in_rerank_place, 0)
        # cal rerank loss
        rerank_loss = None

        all_rerank_hidden_states = torch.cat(all_rerank_hidden_states, 0)
        all_rerank_logits = self.rerank_linear_head(all_rerank_hidden_states)
        all_rerank_logits = torch.reshape(all_rerank_logits, [-1, self.VOCAB_SIZE])

        rerank_loss = loss_fct(all_rerank_logits, labels_in_rerank_place)

        # cal normal loss in rerank place (for comparision with rerank results)        
        normal_loss_in_rerank_place = None

        hidden_states_in_rerank_place = torch.cat(hidden_states_in_rerank_place, 0)
        lm_logits_in_rerank_place = self.lm_head(hidden_states_in_rerank_place)
        lm_logits_in_rerank_place = torch.reshape(lm_logits_in_rerank_place, [-1, self.VOCAB_SIZE])

        normal_loss_in_rerank_place = loss_fct(lm_logits_in_rerank_place, labels_in_rerank_place)

        return {"normal_loss_in_rerank_place": normal_loss_in_rerank_place,
                "rerank_loss": rerank_loss,}
    
class rerankGPT2LMHeadModel_prior_probability(GPT2LMHeadModel):
    def __init__(self, config, MAX_LEN, CAN_NUM, num_of_rerank):
        super().__init__(config)
        self.MAX_LEN = MAX_LEN
        self.CAN_NUM = CAN_NUM
        self.num_of_rerank = num_of_rerank
        self.VOCAB_SIZE = config.vocab_size
        
        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        self.rerank_linear_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

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
        all_candidate_token_ids = []
        check_out_num = 0
        
        if not is_training:
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
            candidate_context_ids = torch.cat([sep_token, candidate_token_ids, sep_token, candidate_token_ids], -1)

            
            #get output from gpt2
            rerank_outputs = self.transformer(candidate_context_ids,
                            past_key_values=past_key_values,
                          )

            #get rerank logits for candidates
            rerank_hidden_states = rerank_outputs[0][:, 2+self.CAN_NUM:2+self.CAN_NUM*2]

            all_rerank_hidden_states.append(rerank_hidden_states[check_in_index])
            no_rerank_logits.append(candidate_token_logits[check_in_index])
            all_candidate_token_ids.append(candidate_token_ids[check_in_index])
            
            if not is_training:
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
            all_candidate_token_ids = torch.flatten(torch.cat(all_candidate_token_ids, 0))
            rerank_linear_head = self.rerank_linear_head.weight[all_candidate_token_ids]
            all_rerank_logits = []
            
            all_rerank_hidden_states = torch.reshape(all_rerank_hidden_states, [-1, all_rerank_hidden_states.shape[-1]])
            for i in range(rerank_linear_head.shape[0]):
                all_rerank_logits.append(torch.matmul(all_rerank_hidden_states[i], rerank_linear_head[i].T))
            all_rerank_logits = torch.reshape(torch.stack(all_rerank_logits), [-1, self.CAN_NUM]) 
            
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
            # cal rerank loss
            loss_fct = CrossEntropyLoss(reduction='none')

            # cal rerank loss
            rerank_loss = None

            all_rerank_hidden_states = torch.cat(all_rerank_hidden_states, 0)
            all_candidate_token_ids = torch.cat(all_candidate_token_ids, 0)
            all_candidate_token_ids_shape = all_candidate_token_ids.shape #use for return
            all_candidate_token_ids = torch.flatten(all_candidate_token_ids)
            rerank_linear_head = self.rerank_linear_head.weight[all_candidate_token_ids]
            all_rerank_logits = []
            
            all_rerank_hidden_states = torch.reshape(all_rerank_hidden_states, [-1, all_rerank_hidden_states.shape[-1]])
            for i in range(rerank_linear_head.shape[0]):
                all_rerank_logits.append(torch.matmul(all_rerank_hidden_states[i], rerank_linear_head[i].T))
            all_rerank_logits = torch.reshape(torch.stack(all_rerank_logits), [-1, self.CAN_NUM]) 
            
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
            all_candidate_token_ids = torch.reshape(all_candidate_token_ids, all_candidate_token_ids_shape)

            return {"all_rerank_logits": all_rerank_logits,
                    "no_rerank_logits": no_rerank_logits,
                    "difficult_level": all_rerank_labels,
                    "rerank_loss": rerank_loss,
                    "normal_loss_in_rerank_place": normal_loss_in_rerank_place,
                    "all_candidate_token_ids": all_candidate_token_ids,
                    "all_prediction_ids": all_prediction_ids,
                    "all_input_ids": all_input_ids}
        
class rerankGPT2LMHeadModel_prior_probability_sharing_head(GPT2LMHeadModel):
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
        all_candidate_token_ids = []
        check_out_num = 0
        
        if not is_training:
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
            candidate_context_ids = torch.cat([sep_token, candidate_token_ids, sep_token, candidate_token_ids], -1)

            
            #get output from gpt2
            rerank_outputs = self.transformer(candidate_context_ids,
                            past_key_values=past_key_values,
                          )

            #get rerank logits for candidates
            rerank_hidden_states = rerank_outputs[0][:, 2+self.CAN_NUM:2+self.CAN_NUM*2]

            all_rerank_hidden_states.append(rerank_hidden_states[check_in_index])
            no_rerank_logits.append(candidate_token_logits[check_in_index])
            all_candidate_token_ids.append(candidate_token_ids[check_in_index])
            
            if not is_training:
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
            all_candidate_token_ids = torch.flatten(torch.cat(all_candidate_token_ids, 0))
            rerank_linear_head = self.lm_head.weight[all_candidate_token_ids]
            all_rerank_logits = []
            
            all_rerank_hidden_states = torch.reshape(all_rerank_hidden_states, [-1, all_rerank_hidden_states.shape[-1]])
            for i in range(rerank_linear_head.shape[0]):
                all_rerank_logits.append(torch.matmul(all_rerank_hidden_states[i], rerank_linear_head[i].T))
            all_rerank_logits = torch.reshape(torch.stack(all_rerank_logits), [-1, self.CAN_NUM]) 
            
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
            # cal rerank loss
            loss_fct = CrossEntropyLoss(reduction='none')

            # cal rerank loss
            rerank_loss = None

            all_rerank_hidden_states = torch.cat(all_rerank_hidden_states, 0)
            all_candidate_token_ids = torch.cat(all_candidate_token_ids, 0)
            all_candidate_token_ids_shape = all_candidate_token_ids.shape #use for return
            all_candidate_token_ids = torch.flatten(all_candidate_token_ids)
            rerank_linear_head = self.lm_head.weight[all_candidate_token_ids]
            all_rerank_logits = []
            
            all_rerank_hidden_states = torch.reshape(all_rerank_hidden_states, [-1, all_rerank_hidden_states.shape[-1]])
            for i in range(rerank_linear_head.shape[0]):
                all_rerank_logits.append(torch.matmul(all_rerank_hidden_states[i], rerank_linear_head[i].T))
            all_rerank_logits = torch.reshape(torch.stack(all_rerank_logits), [-1, self.CAN_NUM]) 
            
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
            all_candidate_token_ids = torch.reshape(all_candidate_token_ids, all_candidate_token_ids_shape)

            return {"all_rerank_logits": all_rerank_logits,
                    "no_rerank_logits": no_rerank_logits,
                    "difficult_level": all_rerank_labels,
                    "rerank_loss": rerank_loss,
                    "normal_loss_in_rerank_place": normal_loss_in_rerank_place,
                    "all_candidate_token_ids": all_candidate_token_ids,
                    "all_prediction_ids": all_prediction_ids,
                    "all_input_ids": all_input_ids}
        
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
        
        hidden_states = []
        
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
            candidate_context_ids = torch.cat([sep_token, candidate_token_ids, sep_token, candidate_token_ids], -1)

            
            #get output from gpt2
            rerank_outputs = self.transformer(candidate_context_ids,
                            past_key_values=past_key_values,
                          )

            #get rerank logits for candidates
            rerank_hidden_states = rerank_outputs[0][:, 2+self.CAN_NUM:2+self.CAN_NUM*2]

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

            # cal normal loss
            normal_loss = None

            hidden_states = torch.cat(hidden_states, 1)
            lm_logits = self.lm_head(hidden_states)

            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            normal_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
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
                    "normal_loss": normal_loss,
                    "normal_loss_in_rerank_place": normal_loss_in_rerank_place,
                    "all_candidate_token_ids": all_candidate_token_ids,
                    "all_prediction_ids": all_prediction_ids,
                    "all_input_ids": all_input_ids}
        
        
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
        
class rerankGPT2LMHeadModel_token_type_embeddings01(GPT2LMHeadModel):
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
            candidate_context_ids = torch.cat([sep_token, candidate_token_ids, sep_token, candidate_token_ids], -1)

            token_type_ids = torch.ones_like(candidate_context_ids)
            #get output from gpt2
            rerank_outputs = self.transformer(candidate_context_ids,
                                              past_key_values=past_key_values,
                                              token_type_ids=token_type_ids
                          )

            #get rerank logits for candidates
            rerank_hidden_states = rerank_outputs[0][:, 2+self.CAN_NUM:2+self.CAN_NUM*2]

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
        
class rerankGPT2LMHeadModel_token_type_embeddings012(GPT2LMHeadModel):
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
            candidate_context_ids = torch.cat([sep_token, candidate_token_ids, sep_token, candidate_token_ids], -1)

            token_type_ids = torch.ones_like(candidate_context_ids)
            token_type_ids[:, -self.CAN_NUM:] = 2
            #get output from gpt2
            rerank_outputs = self.transformer(candidate_context_ids,
                                              past_key_values=past_key_values,
                                              token_type_ids=token_type_ids
                          )

            #get rerank logits for candidates
            rerank_hidden_states = rerank_outputs[0][:, 2+self.CAN_NUM:2+self.CAN_NUM*2]

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