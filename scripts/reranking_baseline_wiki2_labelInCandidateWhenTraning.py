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

from torch import nn
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Dataset, DataLoader, random_split, RandomSampler, SequentialSampler

from transformers import GPT2LMHeadModel,  GPT2Tokenizer, GPT2Config, GPT2LMHeadModel, GPT2Model
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import Trainer, TrainerState, TrainingArguments

from transformers.utils import logging
logger = logging.get_logger(__name__)

from rerankGPT2LMHeadModel import rerankGPT2LMHeadModel_labelInCandidateWhenTraning, wiki2_GPT2Dataset

batch_size = 32
MAX_LEN = 128
CAN_NUM = 5
num_of_rerank = 30

# some parameters I cooked up that work reasonably well
epochs = 5
learning_rate = 5e-4
warmup_steps = 1e2
epsilon = 1e-8

# Set the seed value all over the place to make this reproducible.
seed_val = 42
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)




# I'm not really doing anything with the config buheret
configuration = GPT2Config.from_pretrained('gpt2', output_hidden_states=False)

# Load the GPT tokenizer.
tokenizer = GPT2Tokenizer.from_pretrained('gpt2', pad_token='<|endoftext|>') #gpt2-medium


# instantiate the model
# model = rerankGPT2LMHeadModel_labelInCandidateWhenTraning.from_pretrained("gpt2", config=configuration,
#                                                                           MAX_LEN = MAX_LEN,
#                                                                           CAN_NUM = CAN_NUM, 
#                                                                           num_of_rerank = num_of_rerank)
model = rerankGPT2LMHeadModel_labelInCandidateWhenTraning.from_pretrained(
                                                                  "/mnt/nfs/work1/llcao/zonghaiyao/LM/results/baseline_wiki2021",
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



# the data is in "/mnt/nfs/work1/llcao/zonghaiyao/LM/data/wikitext-2/my_train.csv"
train_df = pd.read_csv("/mnt/nfs/work1/llcao/zonghaiyao/LM/data/wikitext-2/my_train.csv")
validation_df = pd.read_csv("/mnt/nfs/work1/llcao/zonghaiyao/LM/data/wikitext-2/my_validation.csv")
test_df = pd.read_csv("/mnt/nfs/work1/llcao/zonghaiyao/LM/data/wikitext-2/my_test.csv")

train_dataset = wiki2_GPT2Dataset(train_df['text'], tokenizer, max_length=MAX_LEN)
validation_dataset = wiki2_GPT2Dataset(validation_df['text'], tokenizer, max_length=MAX_LEN)
test_dataset = wiki2_GPT2Dataset(test_df['text'], tokenizer, max_length=MAX_LEN)

# Create the DataLoaders for our training and validation datasets.
# We'll take training samples in random order. 
train_dataloader = DataLoader(
            train_dataset,  # The training samples.
            sampler = RandomSampler(train_dataset), # Select batches randomly
            batch_size = batch_size # Trains with this batch size.
        )

# For validation the order doesn't matter, so we'll just read them sequentially.
validation_dataloader = DataLoader(
            validation_dataset, # The validation samples.
            sampler = SequentialSampler(validation_dataset), # Pull out batches sequentially.
            batch_size = batch_size # Evaluate with this batch size.
        )

# For test the order doesn't matter, so we'll just read them sequentially.
test_dataloader = DataLoader(
            test_dataset, # The validation samples.
            sampler = SequentialSampler(test_dataset), # Pull out batches sequentially.
            batch_size = batch_size # Evaluate with this batch size.
        )



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

training_stats = []

model = model.to(device)

for epoch_i in range(0, epochs):

    # ========================================
    #               Training
    # ========================================

    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')

    t0 = time.time()

    total_train_loss = 0
    total_train_normal_loss = 0
    total_train_normal_loss_in_rerank_place = 0
    total_train_rerank_loss = 0

    model.train()

    for step, batch in enumerate(train_dataloader):      
        model.zero_grad()

        outputs = model(  input_ids=batch,            #batch_input_ids
                          labels=batch,               #batch_labels
                          is_training=True,
                       )

        normal_loss = outputs["normal_loss_in_rerank_place"].mean()
        rerank_loss = outputs["rerank_loss"].mean()

        loss = normal_loss + rerank_loss
        
        batch_loss = loss.item()
        total_train_loss += batch_loss
        
        batch_normal_loss = normal_loss.item()
        total_train_normal_loss += batch_normal_loss
        
        batch_rerank_loss = rerank_loss.item()
        total_train_rerank_loss += batch_rerank_loss
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        if step % 1000 == 0 and not step == 0:
            elapsed = format_time(time.time() - t0)
            print('  Batch {:>5,}  of  {:>5,}. Loss: {:>5,}.   Elapsed: {:}.'.format(step, len(train_dataloader), batch_loss, elapsed))

    # Calculate the average loss over all of the batches.
    avg_train_loss = total_train_loss / len(train_dataloader)       
    avg_train_normal_loss = total_train_normal_loss / len(train_dataloader)      
    avg_train_rerank_loss = total_train_rerank_loss / len(train_dataloader)       
    
    # Measure how long this epoch took.
    training_time = format_time(time.time() - t0)

    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    print("  Average training normal_loss: {0:.2f}".format(avg_train_normal_loss))
    print("  Average training rerank_loss: {0:.2f}".format(avg_train_rerank_loss))
    print("  Training epoch took: {:}".format(training_time))
        
    # ========================================
    #               Validation
    # ========================================

    print("")
    print("Running Validation...")

    t0 = time.time()

    model.eval()

    total_eval_loss = 0
    total_eval_normal_loss = 0
    total_eval_normal_loss_in_rerank_place = 0
    total_eval_rerank_loss = 0


    # Evaluate data for one epoch
    for batch in validation_dataloader:
        with torch.no_grad():        

            outputs = model(  input_ids=batch,         #batch_input_ids
                              labels=batch,            #batch_labels
                              is_training=False,
                           )
          
            normal_loss = outputs["normal_loss_in_rerank_place"].mean()
            rerank_loss = outputs["rerank_loss"].mean()
            
            loss = normal_loss + rerank_loss
        
        batch_loss = loss.item()
        total_eval_loss += batch_loss        
        
        batch_normal_loss = normal_loss.item()
        total_eval_normal_loss += batch_normal_loss
        
        batch_rerank_loss = rerank_loss.item()
        total_eval_rerank_loss += batch_rerank_loss

    avg_val_loss = total_eval_loss / len(validation_dataloader)
    avg_val_normal_loss = total_eval_normal_loss / len(validation_dataloader)          
    avg_val_rerank_loss = total_eval_rerank_loss / len(validation_dataloader)    
    
    validation_time = format_time(time.time() - t0)    

    print("  Validation Loss: {0:.2f}".format(avg_val_loss))
    print("  Average Validation normal_loss: {0:.2f}".format(avg_val_normal_loss))
    print("  Average Validation rerank_loss: {0:.2f}".format(avg_val_rerank_loss))
    print("  Validation took: {:}".format(validation_time))

    # Record all statistics from this epoch.
    training_stats.append(
        {
            'epoch': epoch_i + 1,
            'Training Loss': avg_train_loss,
            'Valid. Loss': avg_val_loss,
            'Training Time': training_time,
            'Validation Time': validation_time
        }
    )

print("")
print("Training complete!")
print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))
# print(f"Perplexity: {math.exp(eval_loss):.2f}")

# model.module.save_pretrained("/mnt/nfs/work1/llcao/zonghaiyao/LM/results/results/baseline_wiki2")