# *******************************bart from source**************************************************************
import onnx
import os,sys
import time
# sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)),'transformers'))
# sys.path.insert(0,os.path.join(os.path.dirname(os.path.dirname('/data/yuchen/projects/gns/torch2trt/main.py')),'transformers/scr'))
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

from transformers.generation_logits_process import MinLengthLogitsProcessor, NoRepeatNGramLogitsProcessor

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
    # tokenizer = BartTokenizer.from_pretrained(MODEL)
    # data = "hello , my name is zang, it's nice day and I'd like to meet you in the garden"
    # inputs = tokenizer(data, max_length=128, return_tensors='pt', padding='max_length', truncation=True)#'max_length'
    # input_id_tensor = inputs['input_ids'].to(device)
    # attention_mask = inputs['attention_mask'].to(device)
    # summary_ids = model.generate(input_id_tensor, num_beams=10, min_length=60, max_length=120, early_stopping=True)
    # r = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids]

## BartEncoderLayer
### encoderlayer dummy input
def encoderlayer_convertor(encoder,device):
    encoder_layer_hidden_states = torch.tensor(np.random.normal(size =(1,1024,1024))).type(torch.float32).to(device)
    encoder_layer_attention_mask =  torch.zeros((1,1,1024,1024),dtype=torch.float32).to(device)
    encoder_layer = encoder.layers[0]
    encoder_layer_tensorrt = torch2trt(encoder_layer, [encoder_layer_hidden_states,encoder_layer_attention_mask])#,torch.tensor([[output_attentions]])
    # check time
    times = []
    raw_times=[]
    tensorrt_times=[]
    for _ in range(1000):
        start = time.time()
        y = encoder_layer(encoder_layer_hidden_states, encoder_layer_attention_mask)
        end = time.time()
        raw_time = end - start
        start = time.time()
        y_trt = encoder_layer_tensorrt(encoder_layer_hidden_states,encoder_layer_attention_mask)
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
    return encoder_layer_tensorrt

# # BartDecoder
# decoder = model.get_decoder()
# input_ids = torch.tensor([2]*10).view((-1,1)).to(device)
# encoder_hidden_states = torch.tensor(np.random.normal(size =(10,24,1024))).type(torch.float32).to(device)
# encoder_attention_mask =  torch.ones((10,24),dtype=torch.int32).to(device)
# use_cache = True
# output_attentions = False
# output_hidden_states = False
# return_dict = True
# decoder_output= decoder(input_ids=input_ids,encoder_hidden_states= encoder_hidden_states,\
# encoder_attention_mask=encoder_attention_mask,use_cache = True,output_attentions = False,\
# output_hidden_states = False,return_dict=True)

# ## BartDecoderLayer
def decoderlayer_convertor(decoder,device):
    decoder_layer_hidden_states = torch.tensor(np.random.normal(size =(10,1,1024))).type(torch.float32).to(device)
    de_encoder_hidden_states = torch.tensor(np.random.normal(size =(10,128,1024))).type(torch.float32).to(device)
    de_encoder_layer_attention_mask =  torch.zeros((10,1,1,128),dtype=torch.float32).to(device)
    decoder_layer = decoder.layers[1]    
    # output_attentions = False
    decoder_layer_output = decoder_layer(decoder_layer_hidden_states,encoder_hidden_states=de_encoder_hidden_states,
    encoder_attention_mask=de_encoder_layer_attention_mask)
    decoder_layer_tensorrt = torch2trt(decoder_layer, [decoder_layer_hidden_states,de_encoder_hidden_states
    ,de_encoder_layer_attention_mask],max_batch_size=10)
    # decoder_layer_output = decoder_layer(decoder_layer_hidden_states)
    # decoder_layer_tensorrt = torch2trt(decoder_layer, [decoder_layer_hidden_states],max_batch_size=10)

    times = []
    raw_times=[]
    tensorrt_times=[]
    for _ in range(1000):
        start = time.time()
        y = decoder_layer(decoder_layer_hidden_states,encoder_hidden_states=de_encoder_hidden_states,
        encoder_attention_mask=de_encoder_layer_attention_mask)
        # y = decoder_layer(decoder_layer_hidden_states)
        end = time.time()

        raw_time = end - start
        start = time.time()
        y_trt = decoder_layer_tensorrt(decoder_layer_hidden_states,de_encoder_hidden_states,
        de_encoder_layer_attention_mask)
        # y_trt = decoder_layer_tensorrt(decoder_layer_hidden_states)
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

# ## BartDecoderLayer#,dyna_dim
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
# self_att_test
def decoderAtt_convertor(decoder,device):
    decoder_layer_hidden_states = torch.tensor(np.random.normal(size =(10,1,1024))).type(torch.float32).to(device)
    de_encoder_hidden_states = torch.tensor(np.random.normal(size =(1,128,1024))).type(torch.float32).to(device)
    de_encoder_layer_attention_mask =  torch.zeros((1,1,1,128),dtype=torch.float32).to(device)
    decoder_layer = decoder.layers[0]
    self_attn = decoder_layer.self_attn    
    # output_attentions = False
    # decoder_layer_output = decoder_layer(decoder_layer_hidden_states,encoder_hidden_states=de_encoder_hidden_states,
    # encoder_attention_mask=de_encoder_layer_attention_mask)
    # decoder_layer_tensorrt = torch2trt(decoder_layer, [decoder_layer_hidden_states,de_encoder_hidden_states
    # ,de_encoder_layer_attention_mask])
    self_attn_output = self_attn(decoder_layer_hidden_states)
    self_attn_tensorrt = torch2trt(self_attn, [decoder_layer_hidden_states],max_batch_size=10)

    times = []
    raw_times=[]
    tensorrt_times=[]
    for _ in range(1000):
        start = time.time()
        # y = decoder_layer(decoder_layer_hidden_states,encoder_hidden_states=de_encoder_hidden_states,
        # encoder_attention_mask=de_encoder_layer_attention_mask)
        y = decoder_layer(decoder_layer_hidden_states)
        end = time.time()

        raw_time = end - start
        start = time.time()
        # y_trt = decoder_layer_tensorrt(decoder_layer_hidden_states,de_encoder_hidden_states,
        # de_encoder_layer_attention_mask)
        y_trt = decoder_layer_tensorrt(decoder_layer_hidden_states)
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


def load_trt_model(model_path):
    model_trt = TRTModule()
    model_trt.load_state_dict(torch.load(model_path))
    return model_trt

def load_trt_model_dynamic(model_path):
    model_trt = TRTModule_dy()
    model_trt.load_state_dict(torch.load(model_path))
    return model_trt


# performance test
def performance_test(trt_model_path,model,device):
    decoder_layer_hidden_states = torch.tensor(np.random.normal(size =(10,1,1024))).type(torch.float32).to(device)
    de_encoder_hidden_states = torch.tensor(np.random.normal(size =(10,128,1024))).type(torch.float32).to(device)
    de_encoder_layer_attention_mask =  torch.zeros((10,1,1,128),dtype=torch.float32).to(device)
    decoder_layer_attention_mask = torch.tensor(np.random.normal(size =(10,1,1,1))).type(torch.float32).to(device)
    
    trt_1st_token_models = {}
    trt_later_token_models = {}
    # for index in tqdm(range(12)):
    #     trt_1st_token_models.append(load_trt_model_dynamic(os.path.join(trt_model_path,'token1_layer_' + str(index) + '.pth')))
    # for index in tqdm(range(12)):
    #     trt_later_token_models[str(index)] = load_trt_model_dynamic(os.path.join(trt_model_path,'tokens_layer_' + str(index) + '.pth'))
    # trt_models = [trt_1st_token_models,trt_later_token_models]
    # trt_later_token_models['0'](decoder_layer_hidden_states,de_encoder_hidden_states,de_encoder_layer_attention_mask,decoder_layer_attention_mask)
    trt_times = []
    raw_times = []
    print('starting test>>>>>>>>>>>>>')
    summary_ids_trt = model.generate(input_id_tensor, num_beams=10, min_length=20, 
    max_length=30, early_stopping=True,use_cache=False,trt_models= True)
    summary_ids_trt = model.generate(input_id_tensor, num_beams=10, min_length=20, 
    max_length=30, early_stopping=True,use_cache=False,trt_models= False)
    for _ in tqdm(range(1)):
        s = time.time()
        summary_ids_trt = model.generate(input_id_tensor, num_beams=10, min_length=60, 
        max_length=120, early_stopping=True,use_cache=False,trt_models= True)
        r = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids_trt]
        e = time.time()
        s1 = time.time()
        summary_ids_trt = model.generate(input_id_tensor, num_beams=10, min_length=60, 
        max_length=120, early_stopping=True,use_cache=False,trt_models= False)
        r = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids_trt]
        e1 = time.time()
        trt_times.append(e-s)
        raw_times.append(e1-s1)
    print('average time of raw model is {}'.format(sum(raw_times)/len(raw_times)))
    print('average time of trt model is {}'.format(sum(trt_times)/len(trt_times)))
    print('acceleration of trt model is {}'.format(sum(raw_times)/sum(trt_times)))
    print('test end>>>>>>>>>>>>>')

# load model and convert
MODEL= 'facebook/bart-large-cnn' #'facebook/bart-base'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data = "hello, my name is zang, it's nice day and I'd like to meet you in the garden"
model, tokenizer, encoder, decoder = load_bart(MODEL,device)
inputs, input_id_tensor, attention_mask  = load_input(tokenizer,device,data)
# summary_ids = model.generate(input_id_tensor, num_beams=10, min_length=60, 
# max_length=120, early_stopping=True,use_cache=False,trt_models= None)
# r = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids]

# test




trt_model_path = '/data/yuchen/projects/gns/model/trt_decoderlayer_dynamic_f32'
performance_test(trt_model_path,model,device)

decoder_layer_hidden_states = torch.tensor(np.random.normal(size =(10,10,1024))).type(torch.float32).to(device)
de_encoder_hidden_states = torch.tensor(np.random.normal(size =(10,128,1024))).type(torch.float32).to(device)
de_encoder_layer_attention_mask =  torch.zeros((10,1,10,128),dtype=torch.float32).to(device)
decoder_layer_attention_mask = torch.tensor(np.random.normal(size =(10,1,10,10))).type(torch.float32).to(device)
    
trt_later_token_models = []
trt_times = {}
for index in tqdm(range(12)):
    trt_later_token_models.append(load_trt_model_dynamic(os.path.join(trt_model_path,'tokens_layer_' + str(index) + '.pth')))
for index , trt_modle in tqdm(enumerate(trt_later_token_models)):
    s = time.time()
    trt_modle(decoder_layer_hidden_states,de_encoder_hidden_states,de_encoder_layer_attention_mask,decoder_layer_attention_mask)
    e = time.time()
    trt_times['layer'+ str(index)] = {'TimeCost': e-s}
print(trt_times)
print('end')

# test

# '''
# convert 12 decoder layers
# '''
trt_model_path = 'model/trt_decoderlayer_dynamic_f32/'
for index,decoder_layer in tqdm(enumerate(decoder.layers)):
    decoder_layer_trt = decoderlayer_convertor_dynamic(decoder_layer,device,first_token=True)
    torch.save(decoder_layer_trt.state_dict(), os.path.join(trt_model_path,'token1_layer_' + str(index) + '.pth'))
    # decoder_layer_trt = decoderlayer_convertor_dynamic(decoder_layer,device,first_token=False)
    # torch.save(decoder_layer_trt.state_dict(), os.path.join(trt_model_path,'tokens_layer_' + str(index) + '.pth'))
print('done')





# *******************************alexnet**************************************************************
# import torch
# # from torch2trt import torch2trt
# from torchvision.models.alexnet import alexnet
# from torch2trt.torch2trt import torch2trt

# # create some regular pytorch model...
# model = alexnet(pretrained=True).eval().cuda()
# # create example data
# x = torch.ones((1, 3, 224, 224)).cuda()

# # convert to TensorRT feeding sample data as input
# model_trt = torch2trt(model, [x])


# *******************************single function test**************************************************************
# import numpy as np
# import torch
# from torch2trt.torch2trt import torch2trt
# def BartLayerNorm(normalized_shape: torch.Size, eps: float = 1e-5, elementwise_affine: bool = True):
#     if torch.cuda.is_available():
#         try:
#             from apex.normalization import FusedLayerNorm

#             return FusedLayerNorm(normalized_shape, eps, elementwise_affine)
#         except ImportError:
#             pass
#     return torch.nn.LayerNorm(normalized_shape, eps, elementwise_affine)

# class test(torch.nn.Module):
#     """Multi-headed attention from 'Attention Is All You Need' paper"""

#     def __init__(self,device):
#         super().__init__()
#         self.linear = torch.nn.Linear(8,8).to(device)
#         self.layer_norm = BartLayerNorm(8).to(device)
#     def forward(self, input1: torch.Tensor):
#         input1 = self.linear(input1)
#         return self.layer_norm(input1)

# from torch import nn
# class MyNetBN(nn.Module):
#     def __init__(self,device): 
#         super(MyNetBN, self).__init__()
#         self.classifier = nn.Sequential(
#             # nn.Linear(4, 8),
#             # nn.BatchNorm2d(2), #applying batch norm
#             # nn.ReLU(),
#             # nn.Linear(48, 24),
#             # nn.BatchNorm1d(24),
#             # nn.ReLU(),
#             # nn.Linear(24, 10)
#             nn.Conv2d(1, 3, 5),         # (N, 1, 8, 8) -> (N,  3, 24, 24)
#             nn.BatchNorm2d(3)  
#         )
#         self.classifier = self.classifier.to(device)
             
#     def forward(self, x):
#         # x = x.view(x.size(0), -1)
#         x = self.classifier(x)
#         return x
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = MyNetBN(device)
# t1= torch.tensor(np.random.normal(size=(1,1,8,8))).type(torch.float32).to(device)
# # t2= torch.tensor(np.random.normal(size=(1, 16, 24, 64))).type(torch.float32).to(device)
# model.eval()
# output= model(t1)
# model_trt = torch2trt(model,[t1])

# y = model(t1)
# y_trt = model_trt(t1)
# # check the output against PyTorch
# print(torch.max(torch.abs(y - y_trt)))

# ************************************layer norm 测试**********************
# from torch import nn
# import torch
# import numpy as np
# from torch2trt import torch2trt
# class MyNetBN(nn.Module):
#     def __init__(self,device): 
#         super(MyNetBN, self).__init__()
#         self.classifier = nn.Sequential(
#             # nn.Linear(8, 8),
#             nn.LayerNorm(8), #applying batch norm
#         )
#         self.classifier = self.classifier.to(device)
             
#     def forward(self, x):
#         # x = x.view(x.size(0), -1)
#         # inputs_embeds = torch.transpose(inputs_embeds,0,1)
#         x = self.classifier(x)
#         return x
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = MyNetBN(device)
# t1= torch.tensor(np.random.normal(size=(1, 2,8))).type(torch.float32).to(device)
# model.eval()
# output= model(t1)
# model_trt = torch2trt(model,[t1])

# y = model(t1)
# t1= torch.tensor(np.random.random_integers(0,10,size=(1,24))).type(torch.int32).to(device)
# y_trt = model_trt(t1)
# # check the output against PyTorch
# print(torch.max(torch.abs(y - y_trt)))


# ************************************embedding layer 测试**********************
# from torch import nn
# import torch
# import numpy as np
# from torch2trt import torch2trt
# class MyNetBN(nn.Module):
#     def __init__(self,device): 
#         super(MyNetBN, self).__init__()
#         self.embed_tokens = nn.Embedding(10,128, 1)
#         self.embed_tokens = self.embed_tokens.to(device)
#         # self.classifier = nn.Sequential(
#         #     # nn.Linear(8, 8),
#         #     nn.LayerNorm(8), #applying batch norm
#         # )
#         # self.classifier = self.classifier.to(device)
             
#     def forward(self, x):
#         # x = x.view(x.size(0), -1)
#         inputs_embeds = self.embed_tokens(x)
#         # inputs_embeds = torch.transpose(inputs_embeds,0,1)
#         # x = self.classifier(x)
#         return inputs_embeds
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = MyNetBN(device)
# t1= torch.tensor(np.random.random_integers(1,5,size=(1,24))).type(torch.int64).to(device)
# # t2= torch.tensor(np.random.normal(size=(1, 16, 24, 64))).type(torch.float32).to(device)
# model.eval()
# output= model(t1)
# model_trt = torch2trt(model,[t1])

# y = model(t1)
# t1= torch.tensor(np.random.random_integers(0,10,size=(1,24))).type(torch.int32).to(device)
# y_trt = model_trt(t1)
# # check the output against PyTorch
# print(torch.max(torch.abs(y - y_trt)))

# ************************************pytorch to onnx**********************
# import onnxruntime as ort
# '''
# define dummy input
# '''
# dummy_input = (input_id_tensor, attention_mask)
# input_names = ["input_ids", "attention_mask"]
# output_names = ["outputs"]
# '''
# convert model to onnx
# '''
# model_path = './model/encoder_modified/model.onnx'
# torch.onnx.export(encoder, dummy_input, model_path,
# input_names = input_names, output_names = output_names, verbose=False)

# ort_session = ort.InferenceSession(model_path)
# def to_numpy(tensor):
#     return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
# ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(input_id_tensor),
#  ort_session.get_inputs()[1].name: to_numpy(attention_mask),
# }
# pred = ort_session.run(['outputs'], ort_inputs)

# '''
# convert to s simplified onnx
# '''
# import onnx
# from onnxsim import simplify

# # # load your predefined ONNX model
# model = onnx.load(model_path)
# # convert model
# model_simp, check = simplify(model)
# model_sim_path = model_path.split('.o')[0]+'_simplified.onnx'
# onnx.save(model_simp, model_sim_path)
# assert check, "Simplified ONNX model could not be validated"

# *************************************onnx to tensorrt**********************
# import tensorrt as trt
# import pycuda.driver as cuda
# import common
# import numpy as np 
# TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
# model_path = './model/encoder_modified/model.onnx'
# model_sim_path = model_path.split('.o')[0]+'_simplified.onnx'
# model_trt_path = model_path.split('model.onnx')[0]+'enigne.trt'
# def build_engine(model_file,trt_logger, max_ws=512*1024*1024, fp16=False):
#     print("building engine")
#     # TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
#     builder = trt.Builder(trt_logger)
#     builder.fp16_mode = fp16
#     config = builder.create_builder_config()
#     config.max_workspace_size = max_ws
#     if fp16:
#         config.flags |= 1 << int(trt.BuilderFlag.FP16)
    
#     explicit_batch = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
#     network = builder.create_network(explicit_batch)
#     with trt.OnnxParser(network, TRT_LOGGER) as parser:
#         with open(model_file, 'rb') as model:
#             parsed = parser.parse(model.read())
#             print("network.num_layers", network.num_layers)
#             #last_layer = network.get_layer(network.num_layers - 1)
#             #network.mark_output(last_layer.get_output(0))
#             engine = builder.build_engine(network, config=config)
#             return engine
# engine = build_engine(model_sim_path,TRT_LOGGER)
# with open(model_trt_path, 'wb') as f:
#     f.write(bytearray(engine.serialize()))
# runtime = trt.Runtime(TRT_LOGGER)
# with open(model_trt_path, 'rb') as f:
#     engine_bytes = f.read()
#     engine = runtime.deserialize_cuda_engine(engine_bytes)

# ******************************* tensorrt performance test************************************** 
# # *****************************one output test method
# bert_output = torch.zeros((1,128,1024),device=device).cpu().detach().numpy()
# '''
# memory allocation for inputs
# '''
# input_id_numpy = input_id_tensor.type(torch.int32).detach().cpu().numpy()
# attention_numpy= attention_mask.detach().cpu().numpy()
# batch_size=1
# d_input_ids = cuda.mem_alloc(batch_size * input_id_numpy.nbytes)
# d_attention_mask = cuda.mem_alloc(batch_size * attention_numpy.nbytes)

# import pycuda.autoinit
# '''setting up a new cuda context'''
# bert_context = engine.create_execution_context()
# '''
# memory allocation for outputs
# '''
# d_output = cuda.mem_alloc(batch_size * bert_output.nbytes)

# bindings = [int(d_input_ids), int(d_attention_mask), int(d_output)]

# stream = cuda.Stream()
# # Transfer input data from python buffers to device(GPU)
# cuda.memcpy_htod_async(d_input_ids, input_id_numpy, stream)
# cuda.memcpy_htod_async(d_attention_mask, attention_numpy, stream)

# bert_context.execute_async(batch_size, bindings, stream.handle, None)

# cuda.memcpy_dtoh_async(bert_output, d_output, stream)
# stream.synchronize()
# print(bert_output)

# # *****************************another output test method
# import tensorrt as trt
# import pycuda.driver as cuda
# import common
# import numpy as np 
# TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
# model_path = './model/encoder_modified/model.onnx'
# model_sim_path = model_path.split('.o')[0]+'_simplified.onnx'
# model_trt_path = model_path.split('model.onnx')[0]+'enigne.trt'
# runtime = trt.Runtime(TRT_LOGGER)

# with open(model_trt_path, 'rb') as f:
#     engine_bytes = f.read()
#     engine = runtime.deserialize_cuda_engine(engine_bytes)

# inputs, outputs, bindings, stream = common.allocate_buffers(engine)
# np.copyto(inputs[0].host, input_id_tensor.type(torch.int32).detach().cpu().numpy())

# with engine.create_execution_context() as context:
#     [output] = common.do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)

# print(output)