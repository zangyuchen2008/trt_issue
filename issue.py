import onnx
import os,sys
from torch2trt import torch2trt
from torch2trt import TRTModule
import torch2trt_dynamic
from torch2trt_dynamic import TRTModule as TRTModule_dy
from transformers.models.bart.tokenization_bart import BartTokenizer
from transformers.models.bart.modeling_bart import BartForConditionalGeneration,BartConfig
import torch
import random
import numpy as np 
import time
from tqdm import tqdm

def load_input(tokenizer,device,data):
    inputs = tokenizer(data, max_length=128, return_tensors='pt', padding='max_length', truncation=True)#'max_length'
    input_id_tensor = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    return inputs, input_id_tensor, attention_mask


def load_bart(MODEL,device):
    model = BartForConditionalGeneration.from_pretrained(MODEL).cuda()
    tokenizer = BartTokenizer.from_pretrained(MODEL)
    encoder = model.get_encoder()
    decoder = model.get_decoder()
    return model,tokenizer, encoder, decoder

def load_trt_model_dynamic(model_path):
    model_trt = TRTModule_dy()
    model_trt.load_state_dict(torch.load(model_path))
    return model_trt

def test(gpu2cpu_op=False,):
    trt_times=[]
    raw_times=[]
    for _ in tqdm(range(10)):
        s = time.time()
        hidden = trt_dec_layer0(decoder_layer_hidden_states,de_encoder_hidden_states,
        de_encoder_layer_attention_mask,decoder_layer_attention_mask)
        if gpu2cpu_op:
            if hidden[0,0,0]: pass
            # hidden[:,0,2].tolist() 
        e = time.time()
        s1 = time.time()
        hidden = dec_layer0(decoder_layer_hidden_states,encoder_hidden_states=de_encoder_hidden_states,
        encoder_attention_mask=de_encoder_layer_attention_mask,attention_mask = decoder_layer_attention_mask)
        if gpu2cpu_op:
            if hidden[0][0,0,0]: pass
            # hidden[0][:,0,2].tolist()
        e1 = time.time()
        trt_times.append(e-s)
        raw_times.append(e1-s1)
    if gpu2cpu_op:
        print('with gpu 2 cpu operation test result: >>>>>>>>>>>>>>>>> ')
    else:
        print('with no gpu 2 cpu operation test result: >>>>>>>>>>>>>>>>> ')
    print('average time of raw model is {}'.format(sum(raw_times)/len(raw_times)))
    print('average time of trt model is {}'.format(sum(trt_times)/len(trt_times)))
    print('acceleration of trt model is {}'.format(sum(raw_times)/sum(trt_times)))  


trt_model_path = './model/tokens_layer_0.pth'
# load trt model
trt_dec_layer0 = load_trt_model_dynamic(trt_model_path)
# load coresponding bart model
MODEL= 'facebook/bart-large-cnn' #'facebook/bart-base'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data = "hello, my name is zang, it's nice day and I'd like to meet you in the garden"
model, tokenizer, encoder, decoder = load_bart(MODEL,device)
inputs, input_id_tensor, attention_mask  = load_input(tokenizer,device,data)
dec_layer0 = decoder.layers[0]
# test data 
decoder_layer_hidden_states = torch.tensor(np.random.normal(size =(10,1,1024))).type(torch.float32).to(device)
de_encoder_hidden_states = torch.tensor(np.random.normal(size =(10,128,1024))).type(torch.float32).to(device)
de_encoder_layer_attention_mask =  torch.zeros((10,1,1,128),dtype=torch.float32).to(device)
decoder_layer_attention_mask = torch.tensor(np.random.normal(size =(10,1,1,1))).type(torch.float32).to(device)

test(False)
test(True)