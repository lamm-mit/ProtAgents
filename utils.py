#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from tqdm.notebook import tqdm
from collections.abc import Iterable
import pandas as pd

import seaborn as sns
import time
import datetime
from matplotlib import pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader, random_split, RandomSampler, SequentialSampler

def extract_task (string_input, end_task_token=')', shift=0):
    #i=string_input.find(start)
    j=string_input.find(end_task_token)
    return string_input [:j+1+shift]
    
def extract_start_and_end (string_input, start_token='[', end_token=']', ):
    #i=string_input.find(start)
    i=string_input.find(start_token)
    j=string_input.find(end_token)
    return string_input [i+1:j]
    
def extract_prediction_values (result_untokenized, start_token='/', end_token='|'):

    prediction=extract_start_and_end ( result_untokenized, start_token=start_token, end_token=end_token )
    pred_task=''
    values=None
    values = [float(i) for i in prediction.split(',')]
    return np.array (values)    
    
def plot_log (training_stats):

    df_stats = pd.DataFrame(data=training_stats)
    df_stats = df_stats.set_index('epoch')

    # Increase the plot size and font size.
    sns.set(font_scale=1.5)
    plt.rcParams["figure.figsize"] = (12,6)
    
    # Plot the learning curve.
    plt.plot(df_stats['Training Loss'], 'b-o', label="Training")
    #plt.plot(df_stats['Valid. Loss'], 'g-o', label="Validation")
    #plt.plot(df_stats['Accuracy'], 'r-o', label="Validation Accuracy")
    
    # Label the plot.
    #plt.title("Training & Validation Loss, Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    #plt.xticks([1, 2, 3, 4])
    plt.savefig(f"loss.svg")
    plt.show()
    plt.close()
    # Plot the learning curve.
    #plt.plot(df_stats['Training Loss'], 'b-o', label="Training")
    #plt.plot(df_stats['Valid. Loss'], 'g-o', label="Validation")
    plt.plot(df_stats['Accuracy'], 'r-o', label="Validation Accuracy")
    
    # Label the plot.
   # plt.title("Training & Validation Loss, Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    #plt.xticks([1, 2, 3, 4])
    plt.savefig(f"accuracy.svg")
    plt.show()
    plt.close()
    return


def flatten(L):
    for item in L:
        try:
            yield from flatten(item)
        except TypeError:
            yield item


# In[ ]:


def cumsum_sma(array, period):
    
    ret = np.cumsum(array, dtype=float)
    ret[period:] = ret[period:] - ret[:-period]
    return ret[period - 1:] / period

def modify_df(df):

    max_force = max(df['Max_Smo_Force'])
    max_energy = max(df['Int_Ene'])
    df['Max_Smo_Force'] = df['Max_Smo_Force']/max_force
    df['Int_Ene'] = df['Int_Ene']/max_energy
    df['forc_data'] = df['forc_data']/max_force
    
    return df

def df_train_test_split(df, test_size=0.2, random_state=1):
    # get random sample 
    test = df.sample(frac=test_size, axis=0, random_state=random_state)
    
    index_test = test.index

    # get everything but the test sample
    train = df.drop(index=test.index)
    
    index_train = train.index
    
    train=train.reset_index(drop=True)
    test=test.reset_index(drop=True)
    
    return train, test, index_train, index_test

def return_str(vals=np.array ([.1, .5, .6, 2.])):
    ch=''
    for i in range (len (vals)):
        ch=ch+f'{vals[i]:1.3f},'
        
    return ch[:-1]

# create datasets

def create_dataset_for_ForceGPT(df, test_size=0.2, thin_factor=100, random_state=1):
    
    df_train, df_test, _ , _ = df_train_test_split(df, test_size, random_state=random_state)
    X_data_sol_train = []
    X_data_sol_test=[]
    
    for i in range(len(df_train)):
     
        avgFResult = cumsum_sma(df_train['forc_data'][i],10)[::thin_factor]

        #plt.plot (avgResult[::10])
        str_= f"CalculateForceHistory<{df_train['AA'][i]}> [{return_str(avgFResult)}]"
        X_data_sol_train.append (str_)

        str_= f"GenerateForceHistory<{return_str(avgFResult)}> [{df_train['AA'][i]}]"
        X_data_sol_train.append (str_)

        str_= f"CalculateForce<{df_train['AA'][i]}> [{df_train['Max_Smo_Force'][i]:1.3f}]"
        X_data_sol_train.append (str_)

        str_= f"CalculateEnergy<{df_train['AA'][i]}> [{df_train['Int_Ene'][i]:1.3f}]"
        X_data_sol_train.append (str_)

        str_= f"CalculateForceEnergy<{df_train['AA'][i]}> [{df_train['Max_Smo_Force'][i]:1.3f},{df_train['Int_Ene'][i]:1.3f}]"
        X_data_sol_train.append (str_)

        str_= f"GenerateForce<{df_train['Max_Smo_Force'][i]:1.3f}> [{df_train['AA'][i]}]"
        X_data_sol_train.append (str_)

        str_= f"GenerateEnergy<{df_train['Int_Ene'][i]:1.3f}> [{df_train['AA'][i]}]"
        X_data_sol_train.append (str_)

        str_= f"GenerateForceEnergy<{df_train['Max_Smo_Force'][i]:1.3f},{df_train['Int_Ene'][i]:1.3f}> [{df_train['AA'][i]}]"
        #str_= f"GenerateForceEnergy<{df_train['AA'][i]}> [{df_train['Int_Ene'][i]:1.3f}]"
        X_data_sol_train.append (str_)

    for i in range (len (df_test)):

        str_= f"CalculateForce<{df_test['AA'][i]}> [{df_test['Max_Smo_Force'][i]:1.3f}]"
        X_data_sol_test.append (str_)

        str_= f"CalculateEnergy<{df_test['AA'][i]}> [{df_test['Int_Ene'][i]:1.3f}]"
        X_data_sol_test.append (str_)

        str_= f"CalculateForceEnergy<{df_test['AA'][i]}> [{df_test['Max_Smo_Force'][i]:1.3f},{df_test['Int_Ene'][i]:1.3f}]"
        X_data_sol_test.append (str_)

        avgFResult = cumsum_sma(df_test['forc_data'][i],10)[::thin_factor]

        str_= f"CalculateForceHistory<{df_test['AA'][i]}> [{return_str(avgFResult)}]"
        X_data_sol_test.append (str_)
               
        str_= f"GenerateForceHistory<{return_str(avgFResult)}> [{df_test['AA'][i]}]"
        X_data_sol_test.append (str_)
        
        str_= f"GenerateForce<{df_test['Max_Smo_Force'][i]:1.3f}> [{df_test['AA'][i]}]"
        X_data_sol_test.append (str_)

        str_= f"GenerateEnergy<{df_test['Int_Ene'][i]:1.3f}> [{df_test['AA'][i]}]"
        X_data_sol_test.append (str_)

        str_= f"GenerateForceEnergy<{df_test['Max_Smo_Force'][i]:1.3f},{df_train['Int_Ene'][i]:1.3f}> [{df_train['AA'][i]}]"
        X_data_sol_test.append (str_)
        
    return X_data_sol_train, X_data_sol_test


# In[ ]:


# create dataloaders

class GPTDataset(Dataset):

    def __init__(self, txt_list, tokenizer, max_length=768):

        self.tokenizer = tokenizer
        self.input_ids = []
        self.attn_masks = []

        for txt in txt_list:

      #encodings_dict = tokenizer('<|startoftext|>'+ txt + '<|endoftext|>', truncation=True, max_length=max_length, padding="max_length")
            encodings_dict = tokenizer(txt, truncation=True, max_length=max_length, padding="max_length")

            self.input_ids.append(torch.tensor(encodings_dict['input_ids']))
            self.attn_masks.append(torch.tensor(encodings_dict['attention_mask']))
        
      #print (txt)
    
    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attn_masks[idx] 

def create_dataloader(X_data_train, X_data_test, tokenizer, max_length, batch_size):
    train_data = GPTDataset(X_data_train, tokenizer, max_length=max_length)
    test_data = GPTDataset(X_data_test, tokenizer, max_length=max_length)

    train_dataloader = DataLoader(
            train_data,  # The training samples.
            sampler = RandomSampler(train_data), 
            #shuffle=True, # Select batches randomly
            batch_size = batch_size # Trains with this batch size.
        )
    
    train_dataloader_text = DataLoader(
            X_data_train,  # The training samples.
            sampler = SequentialSampler(train_data), 
            batch_size = batch_size # Trains with this batch size.
        )

    # For validation the order doesn't matter, so we'll just read them sequentially.
    validation_dataloader = DataLoader(
            test_data, # The validation samples.
            sampler = SequentialSampler(test_data), # Pull out batches sequentially.
            batch_size = batch_size # Evaluate with this batch size.
        )
    # For validation the order doesn't matter, so we'll just read them sequentially.
    validation_dataloader_text = DataLoader(
            X_data_test, # The validation samples.
            sampler = SequentialSampler(X_data_test), # Pull out batches sequentially.
            batch_size = batch_size # Evaluate with this batch size.
        )
    return train_dataloader, train_dataloader_text, validation_dataloader, validation_dataloader_text


# In[1]:


# Function to generate output from prompts
def generate_output_from_prompt(model, device, tokenizer, prompt, print_output=False, max_new_tokens=378, 
                            do_sample=True, top_k=500, top_p=0.9, 
                            num_return_sequences=1, 
                            temperature=0.01, num_beams=1):

    model.eval()
    input_ids = torch.tensor(tokenizer.encode(prompt, add_special_tokens = False)).unsqueeze(0)
    input_ids = input_ids.to(device)
    
    sample_outputs = model.generate(
                                input_ids, 
                                #bos_token_id=random.randint(1,30000),
                                #pad_token_id=tokenizer.eos_token_id,
                                eos_token_id =tokenizer.eos_token_id,
                                pad_token_id=tokenizer.eos_token_id,
                                do_sample=do_sample,   
                                top_k=top_k, 
                                #max_length = 700,
                                max_new_tokens=max_new_tokens,
                                top_p=top_p, 
                                num_return_sequences=num_return_sequences,
                                temperature=temperature,
                                num_beams=num_beams,
                                )
    if print_output:
    	for i, sample_output in enumerate(sample_outputs):
        	decoded_txt = tokenizer.decode(sample_output, skip_special_tokens=True)
        	print("{}: {}\n\n".format(i, decoded_txt))
        
    return sample_outputs

