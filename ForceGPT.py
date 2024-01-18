#!/usr/bin/env python
# coding: utf-8

#import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))
import pandas as pd
from tqdm.notebook import tqdm
import torch
import random
import numpy as np
import seaborn as sns
from transformers import get_linear_schedule_with_warmup
import time
import datetime
from matplotlib import pyplot as plt
from transformers import Trainer, TrainingArguments,DataCollatorForLanguageModeling

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoTokenizer#, BitsAndBytesConfig
import transformers

def ForceGPTmodel(model_name, device):
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        #quantization_config=bnb_config, 
        trust_remote_code=True
    )
    model.config.use_cache = False
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    return model.to(device), tokenizer
    
    

    

