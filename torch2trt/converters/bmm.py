import tensorrt as trt
from torch2trt.torch2trt import tensorrt_converter
# import tensorrt

# print(tensorrt.tensorrt.MatrixOperation)
@tensorrt_converter('torch.bmm')
def convert_add(ctx):
    input_a = ctx.method_args[0]
    input_b = ctx.method_args[1]
    output = ctx.method_return
    input_a_trt, input_b_trt = add_missing_trt_tensors(ctx.network, [input_a, input_b])
    # input_a_trt, input_b_trt = broadcast_trt_tensors(ctx.network, [input_a_trt, input_b_trt], len(output.shape) - 1)
    layer = ctx.network.add_matrix_multiply(input0=input_a_trt, input1=input_b_trt)
    output._trt = layer.get_output(0)
