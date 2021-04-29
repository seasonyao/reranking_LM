import os
import time
import datetime
import torch
import math
import copy 
import random
from packaging import version
import pandas as pd
import numpy as np
import pickle

from torch import nn
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Dataset, DataLoader, random_split, RandomSampler, SequentialSampler

from transformers import GPT2LMHeadModel,  GPT2Tokenizer, GPT2Config, GPT2LMHeadModel, GPT2Model
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import Trainer, TrainerState, TrainingArguments

from transformers.utils import logging
logger = logging.get_logger(__name__)

from rerankGPT2LMHeadModel import rerankGPT2LMHeadModel_stage1_all_tokens_no_stage2, wiki2021_GPT2Dataset

batch_size = 8
MAX_LEN = 128
CAN_NUM = 100
num_of_rerank = 20

# some parameters I cooked up that work reasonably well
epochs = 1
learning_rate = 1e-5
warmup_steps = 1e2
epsilon = 1e-8

# Set the seed value all over the place to make this reproducible.
seed_val = 42
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

SAVE_PATH = "/mnt/nfs/work1/llcao/zonghaiyao/LM/"


# I'm not really doing anything with the config buheret
configuration = GPT2Config.from_pretrained('gpt2', output_hidden_states=False)

# Load the GPT tokenizer.
tokenizer = GPT2Tokenizer.from_pretrained('gpt2', pad_token='<|endoftext|>') #gpt2-medium


# instantiate the model
model = rerankGPT2LMHeadModel_stage1_all_tokens_no_stage2.from_pretrained("/mnt/nfs/work1/llcao/zhiyilai/reranking_LM/results/baseline_wiki2021/stage1_all_tokens_no_stage2_canNUM20/last_model", 
                                                                  config=configuration,
                                                                  MAX_LEN = MAX_LEN,
                                                                  CAN_NUM = CAN_NUM, 
                                                                  num_of_rerank = num_of_rerank)

# this step is necessary because I've added some tokens (bos_token, etc) to the embeddings
# otherwise the tokenizer and model tensors won't match up
model.resize_token_embeddings(len(tokenizer))

if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
model = torch.nn.DataParallel(model) # Encapsulate the model

# Tell pytorch to run this model on the GPU.
device = torch.device("cuda")
model.cuda()

#----------------------------------------------------------------------------------------
with open(SAVE_PATH + 'data/wiki2021/wiki2021_5to8_train_dataset.pkl', 'rb') as f:
    train_input_ids = pickle.load(f)
with open(SAVE_PATH + 'data/wiki2021/wiki2021_0to4_validation_dataset.pkl', 'rb') as f:
    validation_input_ids = pickle.load(f)
with open(SAVE_PATH + 'data/wiki2021/wiki2021_0to4_inside_validation_dataset.pkl', 'rb') as f:
    inside_validation_input_ids = pickle.load(f)
    
train_dataset = wiki2021_GPT2Dataset(train_input_ids)
validation_dataset = wiki2021_GPT2Dataset(validation_input_ids)
inside_validation_dataset = wiki2021_GPT2Dataset(inside_validation_input_ids)

# Create the DataLoaders for our training and validation datasets.
# We'll take training samples in random order. 
train_dataloader = DataLoader(
            train_dataset,  # The training samples.
            sampler = RandomSampler(train_dataset), # Select batches randomly
            batch_size = batch_size # Trains with this batch size.
        )

# # For validation the order doesn't matter, so we'll just read them sequentially.
# validation_dataloader = DataLoader(
#             validation_dataset, # The validation samples.
#             sampler = SequentialSampler(validation_dataset), # Pull out batches sequentially.
#             batch_size = batch_size # Evaluate with this batch size.
#         )

# For inside_validation the order doesn't matter, so we'll just read them sequentially.
inside_validation_dataloader = DataLoader(
            inside_validation_dataset, # The validation samples.
            sampler = SequentialSampler(inside_validation_dataset), # Pull out batches sequentially.
            batch_size = batch_size # Evaluate with this batch size.
        )
#----------------------------------------------------------------------------------------

# Note: AdamW is a class from the huggingface library (as opposed to pytorch) 
optimizer = AdamW(model.parameters(), lr = learning_rate, eps = epsilon)

# Total number of training steps is [number of batches] x [number of epochs]. 
# (Note that this is not the same as the number of training samples).
total_steps = len(train_dataloader) * epochs

# Create the learning rate scheduler.
# This changes the learning rate as the training loop progresses
scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = warmup_steps, 
                                            num_training_steps = total_steps)

def format_time(elapsed):
    return str(datetime.timedelta(seconds=int(round((elapsed)))))

total_t0 = time.time()

model = model.to(device)

for epoch_i in range(0, epochs):

    # ========================================
    #               Training
    # ========================================

    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')

    t0 = time.time()

    total_train_normal_loss = 0

    model.train()

    for step, batch in enumerate(train_dataloader):      
        model.zero_grad()

        outputs = model(  input_ids=batch,            #batch_input_ids
                          labels=batch,               #batch_labels
                          is_training=True,
                       )

        normal_loss = outputs["normal_loss"].mean()
        
        batch_normal_loss = normal_loss.item()
        total_train_normal_loss += batch_normal_loss
        
        normal_loss.backward()
        optimizer.step()
        scheduler.step()
        
        if step % 1000 == 0 and not step == 0:
            elapsed = format_time(time.time() - t0)
            print('  Batch {:>5,}  of  {:>5,}. Loss: {:>5,}.   Elapsed: {:}.'.format(step, len(train_dataloader), batch_normal_loss, elapsed))

        # Get inside eval every x batches.
        if step % 10000 == 0 and not step == 0:           
            t1 = time.time()

            elapsed = format_time(time.time() - t0)
            print('  Batch {:>5,}  of  {:>5,}. Loss: {:>5,}.   Elapsed: {:}.'.format(step, len(train_dataloader), batch_normal_loss, elapsed))

            model.eval()

            total_eval_normal_loss = 0
            

            # Evaluate data for one epoch
            for batch in inside_validation_dataloader:        
                with torch.no_grad():        
                    outputs = model(input_ids=batch,         #batch_input_ids
                                    labels=batch,            #batch_labels
                                    is_training=False,
                                    )

                    normal_loss = outputs["normal_loss"].mean()

                batch_normal_loss = normal_loss.item()
                total_eval_normal_loss += batch_normal_loss

            avg_val_normal_loss = total_eval_normal_loss / len(inside_validation_dataloader)         

            validation_time = format_time(time.time() - t1)    

            print("  Average inside Validation normal_loss:", avg_val_normal_loss)
            print("  inside Validation took:", validation_time)
            
    
            model.train()
        
        if step % 10000 == 0 and not step == 0:
            # save model
            model.module.save_pretrained(
                "results/stage1_all_tokens_start_after_finetuning/stage1_all_tokens_no_stage2/"+str(step)
            )

    

print("")
print("Training complete!")
print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))
# print(f"Perplexity: {math.exp(eval_loss):.2f}")
model.module.save_pretrained("results/stage1_all_tokens_start_after_finetuning/stage1_all_tokens_no_stage2/last_model")