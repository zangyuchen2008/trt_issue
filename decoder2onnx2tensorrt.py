import os,sys
# sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)),'transformers'))
# sys.path.insert(0,os.path.join(os.path.dirname(os.path.dirname('/data/yuchen/projects/gns/torch2trt/main.py')),'transformers/scr'))
from torch2trt import torch2trt
from transformers.models.bart.tokenization_bart import BartTokenizer
from transformers.models.bart.modeling_bart import BartForConditionalGeneration,BartConfig
import torch
import random
import time
from tqdm import tqdm
import numpy as np 
import onnxruntime as ort
import tensorrt as trt
import pycuda.driver as cuda
import onnx
from onnxsim import simplify
import common
MODEL= 'facebook/bart-large-cnn' #'facebook/bart-base'
model_path = './model/decoder_dy_batch10_length128_hascache/model.onnx'
model_sim_path = model_path.split('.o')[0]+'_simplified.onnx'
model_trt_path = model_path.split('model.onnx')[0]+'enigne.trt'

# test
# TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
# model_sim_path = model_path.split('.o')[0]+'_simplified.onnx'
# model_trt_path = model_path.split('model.onnx')[0]+'enigne.trt'
# def build_engine(model_file,trt_logger, max_ws=1024*1024*1024, fp16=False):
#     print("building engine")
#     builder = trt.Builder(trt_logger)
#     builder.fp16_mode = fp16
#     # decoder modified
#     builder.max_batch_size =10
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
#             profile = builder.create_optimization_profile()
#             profile.set_shape('input_ids', (1,1), (10,1), (15,1)) 
#             profile.set_shape('encoder_hidden_states', (1, 128, 1024), (10, 128, 1024), (15, 128, 1024)) 
#             profile.set_shape('encoder_attention_mask', (1,128), (10,128), (15,128)) 
#             config.add_optimization_profile(profile)
#             last_layer = network.get_layer(network.num_layers - 1)
#             network.mark_output(last_layer.get_output(0))
#             engine = builder.build_engine(network, config=config)
#             return engine
# engine = build_engine(model_sim_path,TRT_LOGGER)
# with open(model_trt_path, 'wb') as f:
#     f.write(bytearray(engine.serialize()))
# test

model = BartForConditionalGeneration.from_pretrained(MODEL).cuda() 
tokenizer = BartTokenizer.from_pretrained(MODEL)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data = "hello , my name is zang, it's nice day and I'd like to meet you in the garden"
inputs = tokenizer(data, max_length=128, return_tensors='pt', padding='max_length', truncation=True)#'max_length'
input_id_tensor = inputs['input_ids'].to(device)
attention_mask = inputs['attention_mask'].to(device)
# summary_ids = model.generate(input_id_tensor, num_beams=10, min_length=6, max_length=12, early_stopping=True,use_cache=True)
# summary_ids = model.generate(input_id_tensor, num_beams=10, min_length=6, max_length=12, early_stopping=True,use_cache=False)

# r = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids]

# BartEncoder
encoder = model.get_encoder()
return_dict = True
encoder_kwargs = {'attention_mask':torch.ones(input_id_tensor.shape,dtype = torch.int64).to(device)}
encoder_output= encoder(input_id_tensor, attention_mask=attention_mask)

# BartDecoder
decoder = model.get_decoder()
input_ids = input_id_tensor #torch.tensor([2]*1).view((-1,1)).to(device)
encoder_hidden_states = encoder_output[0] #torch.tensor(np.random.normal(size =(1,24,1024))).type(torch.float32).to(device)
encoder_attention_mask = attention_mask # torch.ones((1,24),dtype=torch.int32).to(device)
# use_cache = True
# output_attentions = False
# output_hidden_states = False
# return_dict = True
input_ids = torch.Tensor([[2]]).type(torch.int64).to(device).expand(10,1)
encoder_attention_mask = encoder_attention_mask[0].reshape((1,128)).expand(10,128)
encoder_hidden_states = torch.Tensor(np.random.normal(size=(10,128,1024))).type(torch.float32).to(device)
decoder_output = decoder(input_ids=input_ids,encoder_hidden_states= encoder_hidden_states,\
encoder_attention_mask=encoder_attention_mask,use_cache = True,output_attentions = False,\
output_hidden_states = False,return_dict=True)


# test
# test



# BartDecoderLayer
# decoder_layer_hidden_states = torch.tensor(np.random.normal(size =(10,1,1024))).type(torch.float32).to(device)
# de_encoder_hidden_states = torch.tensor(np.random.normal(size =(10,24,1024))).type(torch.float32).to(device)
# de_encoder_layer_attention_mask =  torch.zeros((10,1,1,24),dtype=torch.float32).to(device)
# output_attentions = False
# decoder_layer_output = decoder.layers[0](decoder_layer_hidden_states,encoder_hidden_states=de_encoder_hidden_states,
# encoder_attention_mask=de_encoder_layer_attention_mask,
# output_attentions=output_attentions)

# **************convert model to onnx
# '''
# define dummy input
# '''
# dummy_input = (input_ids,encoder_hidden_states,encoder_attention_mask) #,past_key_values
# input_names = ["input_ids","encoder_hidden_states","encoder_attention_mask"] #,"past_key_values"
# output_names = ["last_hidden_state"] #,"past_key_values"
# '''
# convert model to onnx
# '''
# # torch.onnx.export(decoder, dummy_input, model_path,
# # input_names = input_names, output_names = output_names, verbose=True,
# # dynamic_axes={'input_ids':{0:'batch'},'last_hidden_state':{0:'batch'}})

# torch.onnx.export(decoder, (input_ids, encoder_hidden_states, encoder_attention_mask), model_path,
#                 input_names=["input_ids","encoder_hidden_states","encoder_attention_mask"],
#                 output_names=["last_hidden_state"],
#                 dynamic_axes={'input_ids': {0: 'sequence'},'encoder_hidden_states':{0:'sequence'},
#                 'encoder_attention_mask':{0:'sequence'},
#                  'last_hidden_state': {0: 'sequence'}})

# ort_session = ort.InferenceSession(model_path)
# def to_numpy(tensor):
#     return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
# ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(input_ids),
#  ort_session.get_inputs()[1].name: to_numpy(encoder_hidden_states),
#  ort_session.get_inputs()[2].name: to_numpy(encoder_attention_mask),
# }
# pred = ort_session.run(["last_hidden_state"], ort_inputs) #,"past_key_values"

# **************convert model to simplified onnx
# # load your predefined ONNX model
# model = onnx.load(model_path)
# # convert model
# # model_simp, check = simplify(model)
# # onnx-simplifier不支持dynamic shapes…… https://github.com/daquexian/onnx-simplifier/issues/101
# model_simp, check = simplify(model,input_shapes={"input_ids":(1,1),"encoder_hidden_states":(1,128,1024),"encoder_attention_mask":(1,128)})
# onnx.save(model_simp, model_sim_path)

# ort_session = ort.InferenceSession(model_sim_path)
# def to_numpy(tensor):
#     return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
# ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(input_ids),
#  ort_session.get_inputs()[1].name: to_numpy(encoder_hidden_states),
#  ort_session.get_inputs()[2].name: to_numpy(encoder_attention_mask),
# }
# pred = ort_session.run(['last_hidden_state'], ort_inputs)

# **************convert simplified onnx to tensorrt
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
model_sim_path = model_path.split('.o')[0]+'_simplified.onnx'
model_trt_path = model_path.split('model.onnx')[0]+'enigne.trt'
def build_engine(model_file,trt_logger, max_ws=1024*1024*1024, fp16=False):
    print("building engine")
    builder = trt.Builder(trt_logger)
    builder.fp16_mode = fp16
    # decoder modified
    builder.max_batch_size =10
    config = builder.create_builder_config()
    config.max_workspace_size = max_ws
    if fp16:
        config.flags |= 1 << int(trt.BuilderFlag.FP16)
    
    explicit_batch = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(explicit_batch)
    with trt.OnnxParser(network, TRT_LOGGER) as parser:
        with open(model_file, 'rb') as model:
            parsed = parser.parse(model.read())
            print("network.num_layers", network.num_layers)
            # last_layer = network.get_layer(network.num_layers - 1)
            # network.mark_output(last_layer.get_output(0))
            engine = builder.build_engine(network, config=config)
            return engine
# engine = build_engine(model_sim_path,TRT_LOGGER)
engine = build_engine(model_path,TRT_LOGGER)
with open(model_trt_path, 'wb') as f:
    f.write(bytearray(engine.serialize()))

# ************************test tensorrt model increase to batch size 10
# input_ids = torch.Tensor([[2,1],[2,1],[2,1],[2,1],[2,1],[2,1],[2,1],[2,1],[2,1],[2,1]]).type(torch.int64).to(device) #
input_ids = torch.Tensor([[2],[2],[2],[2],[2],[2],[2],[2],[2],[2]]).type(torch.int64).to(device) #
encoder_hidden_states = torch.Tensor(np.random.normal(size=(10,128,1024))).type(torch.float32).to(device)
encoder_attention_mask = encoder_attention_mask[0].reshape((1,128)).expand(10,128)

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
runtime = trt.Runtime(TRT_LOGGER)
with open(model_trt_path, 'rb') as f:
    engine_bytes = f.read()
    engine = runtime.deserialize_cuda_engine(engine_bytes)
inputs, outputs, bindings, stream = common.allocate_buffers(engine)
np.copyto(inputs[0].host, input_ids.reshape(10,).type(torch.int32).detach().cpu().numpy())
np.copyto(inputs[1].host, encoder_hidden_states.reshape(1310720,).type(torch.float32).detach().cpu().numpy())
np.copyto(inputs[2].host, encoder_attention_mask.reshape(1280,).type(torch.int32).detach().cpu().numpy())

tensorrt_time_cost = []
bartencoder_time_cost = []
with engine.create_execution_context() as context:
    for _ in tqdm(range(200)):
        start = time.time()
        [output] = common.do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
        end = time.time()
        tensorrt_time_cost.append(end - start)

        start = time.time()
        decoder(input_ids=input_ids,encoder_hidden_states= encoder_hidden_states,encoder_attention_mask=encoder_attention_mask,use_cache=True)
        end = time.time()
        bartencoder_time_cost.append(end - start)
avg_tesnorrt = sum(tensorrt_time_cost)/len(tensorrt_time_cost)
avg_bartencoder = sum(bartencoder_time_cost)/len(bartencoder_time_cost)
print('tensorrt model cost avg time {}, \n bartdecoder model cost avg time {}, \n{} \
     times accelerated'.format(avg_tesnorrt,avg_bartencoder,avg_bartencoder/avg_tesnorrt))



