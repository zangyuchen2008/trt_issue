from torch2trt.torch2trt import *
from torch2trt.module_test import add_module_test
# converter added
@tensorrt_converter('torch.bmm')
@tensorrt_converter('torch.matmul')
def convert_bmm(ctx):
    input_a = ctx.method_args[0]
    input_b = ctx.method_args[1]
    output = ctx.method_return
    input_a_trt, input_b_trt = add_missing_trt_tensors(ctx.network, [input_a, input_b])
    # input_a_trt, input_b_trt = broadcast_trt_tensors(ctx.network, [input_a_trt, input_b_trt], len(output.shape) - 1)
    layer = ctx.network.add_matrix_multiply(input_a_trt,trt.MatrixOperation.NONE, input_b_trt,trt.MatrixOperation.NONE)
    output._trt = layer.get_output(0)

# @tensorrt_converter('torch.Tensor.__bool__')
# def convert_bool(ctx):
#     input = ctx.method_args[0]
#     output = ctx.method_return
#     input_trt = add_missing_trt_tensors(ctx.network, [input])

# @tensorrt_converter('torch.layer_norm')
# @tensorrt_converter('torch.nn.functional.layer_norm')
# @tensorrt_converter('torch.nn.LayerNorm.forward')
# def convert_bool(ctx):
#     input = ctx.method_args[0]
#     output = ctx.method_return
#     input_trt = add_missing_trt_tensors(ctx.network, [input])[0]
#     weight = ctx.method_args[2].detach().cpu().numpy()
#     bias = ctx.method_args[3].detach().cpu().numpy()
#     esp = ctx.method_args[4]

#     mean = np.average(input.detach().cpu().numpy(),axis=len(input.shape)-1).squeeze()
#     var = np.var(input.detach().cpu().numpy(),axis=len(input.shape)-1).squeeze()

#     scale = 1 / np.sqrt(var + esp)
#     shift = - mean * scale
#     power = np.ones_like(scale)

#     layer = ctx.network.add_shuffle(input_trt)
#     if len(input.shape) == 2:
#         layer.reshape_dims = (input.shape[1], 1, 1)
#     else:
#         layer.reshape_dims = (input.shape[1], input.shape[2], 1)  

#     layer = ctx.network.add_scale_nd(layer.get_output(0), trt.ScaleMode.CHANNEL, shift, scale, power, 0)
    
#     # reshape for adding weight and bias
#     layer = ctx.network.add_shuffle(layer.get_output(0))
#     shape = layer.get_output(0).shape
#     layer.reshape_dims = tuple([shape[1],shape[0],shape[2]])

#     layer = ctx.network.add_scale_nd(layer.get_output(0), trt.ScaleMode.CHANNEL, bias, weight,
#      np.ones_like(weight), 0)

#     # reshape back 
#     layer = ctx.network.add_shuffle(layer.get_output(0))
#     layer.reshape_dims = tuple([shape[0],shape[1]])


#     output._trt = layer.get_output(0)
# converter added
@tensorrt_converter('torch.nn.LayerNorm.forward')
def convert_bool(ctx):
    module = ctx.method_args[0]
    input = ctx.method_args[1]
    output = ctx.method_return
    input_trt = add_missing_trt_tensors(ctx.network, [input])[0]
    weight = module.weight.detach().cpu().numpy()
    bias = module.bias.detach().cpu().numpy()
    eps = module.eps
    
    mean = np.average(input.detach().cpu().numpy(),axis=len(input.shape)-1).squeeze()
    var = np.var(input.detach().cpu().numpy(),axis=len(input.shape)-1).squeeze()

    scale = 1 / np.sqrt(var + eps)
    shift = - mean * scale
    power = np.ones_like(scale)

    # add one dim for add_scale_nv method(add_scale_nv need at least 3 dims)
    layer = ctx.network.add_shuffle(input_trt)
    if len(input.shape) == 2:
        layer.reshape_dims = (input.shape[1], 1, 1)
    else:
        layer.reshape_dims = (input.shape[1], input.shape[2], 1)  

    # layer = ctx.network.add_scale_nd(layer.get_output(0), trt.ScaleMode.CHANNEL, shift, scale, power, 0)
    layer = ctx.network.add_scale_nd(layer.get_output(0), trt.ScaleMode.CHANNEL, 
    np.asarray(shift,dtype=np.float32), np.asarray(scale,dtype=np.float32) ,np.asarray(power,dtype=np.float32) , 0)
    # transpose for adding weight and bias
    layer = ctx.network.add_shuffle(layer.get_output(0))
    shape = layer.get_output(0).shape
    index = list(range(len(shape)))
    # layer.reshape_dims = tuple([shape[1],shape[0],shape[2]])
    layer.second_transpose = tuple([index[1],index[0],index[2]])
    layer = ctx.network.add_scale_nd(layer.get_output(0), trt.ScaleMode.CHANNEL, bias, weight,
     np.ones_like(weight), 0)

    # transpose back 
    layer = ctx.network.add_shuffle(layer.get_output(0))
    layer.second_transpose = tuple([index[1],index[0],index[2]])
    # reshape to squeeze last dimension 
    layer = ctx.network.add_shuffle(layer.get_output(0))
    layer.reshape_dims = tuple([shape[0],shape[1]])


    output._trt = layer.get_output(0)
# converter added
# @tensorrt_converter('torch.Tensor.expand')
# def convert_bool(ctx):
#     input_a = ctx.method_args[0]
#     input_b = ctx.method_args[1]
#     output = ctx.method_return
#     input_a_trt, input_b_trt = add_missing_trt_tensors(ctx.network, [input_a, input_b])
#     # input_a_trt, input_b_trt = broadcast_trt_tensors(ctx.network, [input_a_trt, input_b_trt], len(output.shape) - 1)
#     layer = ctx.network.add_matrix_multiply(input_a_trt,trt.MatrixOperation.NONE, input_b_trt,trt.MatrixOperation.NONE)
#     output._trt = layer.get_output(0)


# converter added
@tensorrt_converter('torch.embedding')
# # @tensorrt_converter('torch.nn.functional.embedding')
def convert_bool(ctx):
    embeding = ctx.method_args[0]
    input = ctx.method_args[1]
    input_trt = add_missing_trt_tensors(ctx.network, [input])[0]
    embeding_trt = add_missing_trt_tensors(ctx.network, [embeding])[0]
    output = ctx.method_return
    output._trt = ctx.network.add_gather(embeding_trt,input_trt,0).get_output(0)
# converter added
@tensorrt_converter('torch.nn.functional.embedding')
def convert_bool(ctx):
    input = ctx.method_args[0]
    embeding = ctx.method_args[1]
    input_trt = add_missing_trt_tensors(ctx.network, [input])[0]
    embeding_trt = add_missing_trt_tensors(ctx.network, [embeding])[0]
    output = ctx.method_return
    output._trt = ctx.network.add_gather(embeding_trt,input_trt,0).get_output(0)  

# @tensorrt_converter('torch.Tensor.expand')
# def convert_bool(ctx):
#     input = ctx.method_args[0]
#     embeding = ctx.method_args[1]
#     input_trt = add_missing_trt_tensors(ctx.network, [input])[0]
#     embeding_trt = add_missing_trt_tensors(ctx.network, [embeding])[0]
#     output = ctx.method_return
#     output._trt = ctx.network.add_gather(embeding_trt,input_trt,0).get_output(0) 


@tensorrt_converter('torch.mul')
@tensorrt_converter('torch.Tensor.__imul__')
@tensorrt_converter('torch.Tensor.__mul__')
@tensorrt_converter('torch.Tensor.__rmul__')
def convert_mul(ctx):
    input_a = ctx.method_args[0]
    input_b = ctx.method_args[1]
    output = ctx.method_return
    input_a_trt, input_b_trt = add_missing_trt_tensors(ctx.network, [input_a, input_b])
    input_a_trt, input_b_trt = broadcast_trt_tensors(ctx.network, [input_a_trt, input_b_trt], len(output.shape) - 1)
    layer = ctx.network.add_elementwise(input_a_trt, input_b_trt, trt.ElementWiseOperation.PROD)
    output._trt = layer.get_output(0)

class Mul(torch.nn.Module):
    def __init__(self):
        super(Mul, self).__init__()

    def forward(self, x, y):
        return x * y

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224), (1, 3, 224, 224)])
def test_mul_basic():
    return Mul()


class IMul(torch.nn.Module):
    def __init__(self):
        super(IMul, self).__init__()

    def forward(self, x, y):
        x *= y
        return x


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224), (1, 3, 224, 224)])
def test_mul_imul():
    return IMul()


class TorchMul(torch.nn.Module):
    def __init__(self):
        super(TorchMul, self).__init__()

    def forward(self, x, y):
        return torch.mul(x, y)
    

@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224), (1, 3, 224, 224)])
def test_mul_torchmul():
    return TorchMul()


class RMulInt(torch.nn.Module):
    def __init__(self):
        super(RMulInt, self).__init__()

    def forward(self, x):
        return 10 * x


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3, 3)])
def test_rmul_int():
    return RMulInt()


class RMulFloat(torch.nn.Module):
    def __init__(self):
        super(RMulFloat, self).__init__()

    def forward(self, x):
        return 10.0 * x


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3, 3)])
def test_rmul_float():
    return RMulFloat()


class MulConstantNoBatch(torch.nn.Module):
    def __init__(self):
        super(MulConstantNoBatch, self).__init__()
        self.register_buffer('y', torch.ones((3, 10, 10)))

    def forward(self, x):
        return x * self.y

    
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 10, 10)])
def test_mul_constant_nobatch():
    return MulConstantNoBatch()


class MulConstantBatch(torch.nn.Module):
    def __init__(self):
        super(MulConstantBatch, self).__init__()
        self.register_buffer('y', torch.ones((1, 3, 10, 10)))

    def forward(self, x):
        return x * self.y

    
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 10, 10)])
def test_mul_constant_batch():
    return MulConstantBatch()
