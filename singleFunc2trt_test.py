from torch import nn
import torch
import numpy as np 
from torch2trt import torch2trt

# layer test
class MyNetBN(nn.Module):
    def __init__(self,device): 
        super(MyNetBN, self).__init__()
        self.classifier = nn.Sequential(
            # nn.Linear(4, 8),
            # nn.BatchNorm2d(2), #applying batch norm
            # nn.ReLU(),
            # nn.Linear(48, 24),
            # nn.BatchNorm1d(24),
            # nn.ReLU(),
            # nn.Linear(24, 10)
            # nn.Conv2d(1, 3, 5),         # (N, 1, 8, 8) -> (N,  3, 24, 24)
            # nn.BatchNorm2d(3) 
            nn.LayerNorm(1024) 
        )
        self.classifier = self.classifier.to(device)
        self.embedding = nn.Embedding(10, 3).to(device)
             
    def forward(self, x):
        # x = x.view(x.size(0), -1)
        x = self.classifier(x)
        # y =  torch.split(x,2,0)
        return x

# function test
class TensorSplit(nn.Module):
    
    def __init__(self, *args, **kwargs):
        super(TensorSplit, self).__init__()
        self.args = args
        self.kwargs = kwargs
        self.weight = args[1]
        self.bias = args[2]

    def forward(self, x):
        # ashape= x.shape
        # extended_mask =  x.view([ashape[0],ashape[1],ashape[3],ashape[2]]).type(torch.float32)
        # ones = torch.ones(ashape[:3]+ (self.args[2],),dtype=torch.float32).to(extended_mask.device)
        # expanded_mask= torch.matmul(extended_mask,ones).type(torch.int64)
        # expanded_mask = torch.transpose(expanded_mask,2,3)
        mean = x.mean(dim=len(x.shape)-1,keepdim=True)
        var = ((x - mean)**2).mean(2,keepdim=True)
        para1 = (x - mean)/torch.sqrt(var + 1e-05)
        return torch.mul(para1,self.weight) + self.bias
        # return var

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


weight = torch.Tensor(np.random.normal(size=(1024,)).astype(np.float32)).to(device)
bias = torch.Tensor(np.random.normal(size=(1024,)).astype(np.float32)).to(device)
model = TensorSplit(device,weight,bias)

t1= torch.tensor(np.random.normal(size=(1,1,1024))).type(torch.float32).to(device)
# t1= torch.tensor(np.random.randint(0,10,size=(1,1,1,4))).type(torch.float32).to(device)
# model = TensorSplit(1,1,4,4)
# t1= torch.tensor(np.random.normal(size=(1,1,8,8))).type(torch.float32).to(device)
# t2= torch.tensor(np.random.normal(size=(1, 16, 24, 64))).type(torch.float32).to(device)
model.eval()
output= model(t1)
model_trt = torch2trt(model,[t1],max_batch_size=10)
print(torch.max(torch.abs(model(t1) - model_trt(t1))))