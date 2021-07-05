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
    
class wiki2021_GPT2Dataset(Dataset):
    def __init__(self, input_ids):
        self.input_ids = input_ids
    
    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx]
    
class rerankGPT2LMHeadModel_finetuning_gpt2(GPT2LMHeadModel):
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
        rerank_places = random.sample(np.arange(self.MAX_LEN-1).tolist(), k=self.num_of_rerank) #no duplicate
        rerank_places = np.sort(rerank_places) #add first and last tokens to make segments
        
        outputs = self.transformer(
            input_ids,
        )
        
        hidden_states = outputs[0]

        loss_fct = CrossEntropyLoss(reduction='none')

        # cal normal loss
        normal_loss = None

        lm_logits = self.lm_head(hidden_states)

        # Shift so that tokens < n predict n
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        normal_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        # cal normal loss in rerank place
        normal_loss_in_rerank_place = normal_loss.view((input_ids.shape[0], -1))[:, rerank_places].view(-1)

        return {"normal_loss": normal_loss,
                "normal_loss_in_rerank_place": normal_loss_in_rerank_place}

        
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

        rerank_places = random.sample(np.arange(1, self.MAX_LEN).tolist(), k=self.num_of_rerank) #no duplicate
        rerank_places = np.concatenate(([0], np.sort(rerank_places), [self.MAX_LEN])) #add first and last tokens to make segments
        
        past_key_values = None
        all_rerank_hidden_states = []
        all_rerank_labels = []
        no_rerank_logits = []
        check_out_num = 0
        
        hidden_states = []
        all_stage1_logits_rerank_place= []
        all_rerank_labels = []
        
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
            
            if i == self.num_of_rerank:
                break
            
            stage1_hidden_states_rerank_place = segment_hidden[:, -1, :]
            stage1_logits_rerank_place = self.lm_head(stage1_hidden_states_rerank_place)
            rerank_labels = labels[..., rerank_places[i+1]]

            all_stage1_logits_rerank_place.append(stage1_logits_rerank_place)
            all_rerank_labels.append(rerank_labels)
        
#         print("\n batch info:")
#         print("there are ", check_out_num/(self.num_of_rerank*batch_size), "labels not in candidates")
    

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
        
        # cal normal loss in rerank place
        normal_loss_in_rerank_place = None
        all_stage1_logits_rerank_place = torch.cat(all_stage1_logits_rerank_place, 0)
        all_rerank_labels = torch.cat(all_rerank_labels, 0)
        
        normal_loss_in_rerank_place = loss_fct(all_stage1_logits_rerank_place, all_rerank_labels)

        return {"normal_loss": normal_loss,
                "normal_loss_in_rerank_place": normal_loss_in_rerank_place}
        
    
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
            stage2_logits = stage2_logits.scatter(1, candidate_token_ids, stage2_candidates_logits)
#             stage2_logits = stage2_logits.scatter_add(1, candidate_token_ids, stage2_candidates_logits)
            
            all_stage2_logits.append(stage2_logits)
        
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