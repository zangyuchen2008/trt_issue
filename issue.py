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

def decoderlayer_convertor_dynamic(decoder_layer,device,first_token=True): 
    decoder_layer_hidden_states = torch.tensor(np.random.normal(size =(10,1,1024))).type(torch.float32).to(device)
    de_encoder_hidden_states = torch.tensor(np.random.normal(size =(10,128,1024))).type(torch.float32).to(device)
    de_encoder_layer_attention_mask =  torch.zeros((10,1,1,128),dtype=torch.float32).to(device)
    decoder_layer_attention_mask = torch.tensor(np.random.normal(size =(10,1,1,1))).type(torch.float32).to(device)
    # decoder_layer = decoder.layers[1]    
    # output_attentions = False
    # dmin,dopt,dmax = dyna_dim
    opt_shape_param = [
        [
            [10,1,1024],   # min
            [10,128,1024],   # opt
            [10,256,1024]    # max
        ],
        [
            [10,128,1024],   # min
            [10,128,1024],   # opt
            [10,128,1024]    # max
        ],
        [
            [10,1,1,128],    # min
            [10,1,128,128],   # opt
            [10,1,256,128]    # max
        ],
    ]

    opt_shape_param_mask = [
        [
            [10,1,1024],   # min
            [10,128,1024],   # opt
            [10,256,1024]    # max
        ],
        [
            [10,128,1024],   # min
            [10,128,1024],   # opt
            [10,128,1024]    # max
        ],
        [
            [10,1,1,128],    # min
            [10,1,128,128],   # opt
            [10,1,256,128]    # max
        ],
        [
            [10,1,1,1],    # min
            [10,1,128,128],   # opt
            [10,1,256,256]    # max
        ],
    ]
    # decoder_layer_output = decoder_layer(decoder_layer_hidden_states)
    # decoder_layer_tensorrt = torch2trt_dynamic.torch2trt_dynamic(decoder_layer, [decoder_layer_hidden_states], fp16_mode=False, opt_shape_param=opt_shape_param)
    if first_token:
        decoder_layer_output = decoder_layer(decoder_layer_hidden_states,encoder_hidden_states=de_encoder_hidden_states,
        encoder_attention_mask=de_encoder_layer_attention_mask)
        decoder_layer_tensorrt = torch2trt_dynamic.torch2trt_dynamic(decoder_layer, [decoder_layer_hidden_states,de_encoder_hidden_states
        ,de_encoder_layer_attention_mask], fp16_mode=False, opt_shape_param=opt_shape_param)
    else:
        decoder_layer_output = decoder_layer(decoder_layer_hidden_states,encoder_hidden_states=de_encoder_hidden_states,
        encoder_attention_mask=de_encoder_layer_attention_mask,attention_mask = decoder_layer_attention_mask )
        decoder_layer_tensorrt = torch2trt_dynamic.torch2trt_dynamic(decoder_layer, [decoder_layer_hidden_states,de_encoder_hidden_states
        ,de_encoder_layer_attention_mask,decoder_layer_attention_mask], fp16_mode=False, opt_shape_param=opt_shape_param_mask)        

    times = []
    raw_times=[]
    tensorrt_times=[]
    decoder_layer_hidden_states = torch.tensor(np.random.normal(size =(10,64,1024))).type(torch.float32).to(device)
    de_encoder_hidden_states = torch.tensor(np.random.normal(size =(10,128,1024))).type(torch.float32).to(device)
    de_encoder_layer_attention_mask =  torch.zeros((10,1,64,128),dtype=torch.float32).to(device)
    decoder_layer_attention_mask = torch.tensor(np.random.normal(size =(10,1,64,64))).type(torch.float32).to(device)
    for _ in range(100):
        start = time.time()
        if first_token:
            y = decoder_layer(decoder_layer_hidden_states,encoder_hidden_states=de_encoder_hidden_states,
            encoder_attention_mask=de_encoder_layer_attention_mask)
        else:
            y = decoder_layer(decoder_layer_hidden_states,encoder_hidden_states=de_encoder_hidden_states,
            encoder_attention_mask=de_encoder_layer_attention_mask,attention_mask = decoder_layer_attention_mask)
        # y = decoder_layer(decoder_layer_hidden_states)
        torch.cuda.synchronize(device)
        end = time.time()

        raw_time = end - start
        start = time.time()
        if first_token:
            y_trt = decoder_layer_tensorrt(decoder_layer_hidden_states,de_encoder_hidden_states,
            de_encoder_layer_attention_mask)
        else:
            y_trt = decoder_layer_tensorrt(decoder_layer_hidden_states,de_encoder_hidden_states,
            de_encoder_layer_attention_mask,decoder_layer_attention_mask)            
        # y_trt = decoder_layer_tensorrt(decoder_layer_hidden_states)
        torch.cuda.synchronize(device)
        end = time.time()

        tensorrt_time = end - start
        raw_times.append(raw_time)
        tensorrt_times.append(tensorrt_time)
        times.append(raw_time / tensorrt_time)
    print("raw model time is {}".format(sum(raw_times)/len(raw_times)))
    print("tesnorrt model time is {}".format(sum(tensorrt_times)/len(tensorrt_times)))
    print("coresponding tensorrt model is {} times faster".format(sum(times)/len(times)))
    # check the output against PyTorch
    print(torch.max(torch.abs(y - y_trt)))
    return decoder_layer_tensorrt


def test(gpu2cpu_op=False,synchronize=False,device=None):
    trt_times=[]
    raw_times=[]
    for _ in tqdm(range(10)):
        s = time.time()
        hidden = trt_dec_layer0(decoder_layer_hidden_states,de_encoder_hidden_states,
        de_encoder_layer_attention_mask,decoder_layer_attention_mask)
        if gpu2cpu_op:
            if hidden[0,0,0]: pass
            # hidden[:,0,2].tolist() 
        if synchronize:
            torch.cuda.synchronize(device)
        e = time.time()
        s1 = time.time()
        hidden = dec_layer0(decoder_layer_hidden_states,encoder_hidden_states=de_encoder_hidden_states,
        encoder_attention_mask=de_encoder_layer_attention_mask,attention_mask = decoder_layer_attention_mask)
        if gpu2cpu_op:
            if hidden[0][0,0,0]: pass
            # hidden[0][:,0,2].tolist()
        if synchronize:
            torch.cuda.synchronize(device)
        e1 = time.time()
        trt_times.append(e-s)
        raw_times.append(e1-s1)
    if gpu2cpu_op:
        print('test with gpu 2 cpu operation: >>>>>>>>>>>>>>>>> ')
    else:
        print('test with no gpu 2 cpu operation: >>>>>>>>>>>>>>>>> ')
    if synchronize:
        print('test with cuda synchronize: >>>>>>>>>>>>>>>>> ')
    else:
        print('test with no cuda synchronize: >>>>>>>>>>>>>>>>> ')
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

# start to test
test(False,False,device)
test(True,False,device)

# test(False,True,device)
# test(True,True,device)

# converting to tensorrt model
decoder_layer_trt = decoderlayer_convertor_dynamic(decoder.layers[0],device,first_token=True)