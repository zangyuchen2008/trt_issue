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
model_path = './model/bart/model.onnx'
model_sim_path = model_path.split('.o')[0]+'_simplified.onnx'
model_trt_path = model_path.split('model.onnx')[0]+'enigne.trt'

model = BartForConditionalGeneration.from_pretrained(MODEL).cuda() 
tokenizer = BartTokenizer.from_pretrained(MODEL)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data = "hello , my name is zang, it's nice day and I'd like to meet you in the garden"
inputs = tokenizer(data, max_length=128, return_tensors='pt', padding='max_length', truncation=True)#'max_length'
input_id_tensor = inputs['input_ids'].to(device)
attention_mask = inputs['attention_mask'].to(device)
summary_ids = model.generate(input_id_tensor, num_beams=10, min_length=60, max_length=120, early_stopping=True)
# r = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids]

# BartEncoder
encoder = model.get_encoder()
return_dict = True
encoder_kwargs = {'attention_mask':torch.ones(input_id_tensor.shape,dtype = torch.int64).to(device)}
encoder_output= encoder(input_id_tensor, attention_mask=attention_mask)

# **************convert model to onnx
'''
define dummy input
'''
dummy_input = (input_id_tensor)
input_names = ["input_ids"]
output_names = ["outputs"]
'''
convert model to onnx
'''
torch.onnx.export(model.generate, dummy_input, model_path,
input_names = input_names, output_names = output_names, verbose=True)

ort_session = ort.InferenceSession(model_path)
def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(input_id_tensor),
 ort_session.get_inputs()[1].name: to_numpy(attention_mask),
}
pred = ort_session.run(['outputs'], ort_inputs)

# **************convert model to simplified onnx
# load your predefined ONNX model
# model = onnx.load(model_path)
# # convert model
# model_simp, check = simplify(model)
# onnx.save(model_simp, model_sim_path)

# ort_session = ort.InferenceSession(model_sim_path)
# def to_numpy(tensor):
#     return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
# ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(input_id_tensor),
#  ort_session.get_inputs()[1].name: to_numpy(attention_mask),
# }
# pred = ort_session.run(['outputs'], ort_inputs)

# **************convert simplified onnx to tensorrt
# TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
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

# ************************test tensorrt model
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
runtime = trt.Runtime(TRT_LOGGER)
with open(model_trt_path, 'rb') as f:
    engine_bytes = f.read()
    engine = runtime.deserialize_cuda_engine(engine_bytes)
inputs, outputs, bindings, stream = common.allocate_buffers(engine)
np.copyto(inputs[0].host, input_id_tensor.type(torch.int32).detach().cpu().numpy())
np.copyto(inputs[1].host, attention_mask.type(torch.int32).detach().cpu().numpy())

tensorrt_time_cost = []
bartencoder_time_cost = []
with engine.create_execution_context() as context:
    for _ in tqdm(range(10000)):
        start = time.time()
        [output] = common.do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
        end = time.time()
        tensorrt_time_cost.append(end - start)

        start = time.time()
        encoder(input_id_tensor, attention_mask=attention_mask)
        end = time.time()
        bartencoder_time_cost.append(end - start)
avg_tesnorrt = sum(tensorrt_time_cost)/len(tensorrt_time_cost)
avg_bartencoder = sum(bartencoder_time_cost)/len(bartencoder_time_cost)
print('tensorrt model cost avg time {}, \n bartencoder model cost avg time {}, \n{} \
     times accelerated'.format(avg_tesnorrt,avg_bartencoder,avg_bartencoder/avg_tesnorrt))



