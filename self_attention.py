# @File   : self_attention.py
# @Author : He Zhuangzhuang
# @Version: 1.0
# @Date   :2021/8/11,下午8:47
import torch
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")
class Self_Attention(torch.nn.Module):
    def __init__(self,size):
        super(Self_Attention, self).__init__()
        self.W_k = torch.nn.Parameter(torch.normal(mean=0,std=1,size=(size,size)),requires_grad=True)
        self.W_q= torch.nn.Parameter(torch.normal(mean=0,std=1,size=(size,size)),requires_grad=True)
        self.W_v=torch.nn.Parameter(torch.normal(mean=0,std=1,size=(size,size)),requires_grad=True)
        self.linear = torch.nn.Linear(10,1)

    def forward(self,X):
        Q = torch.mm(self.W_q,X)
        K = torch.mm(self.W_k,X)
        V = torch.mm(self.W_v,X)
        A = torch.mm(K.T,Q)
        A_1 = F.softmax(A,dim=0)
        O = torch.mm(V,A_1)
        y = self.linear(O)
        return y

net= Self_Attention(10)
import torch
X = torch.randn(100,10)
y = torch.range(1,100)
loss = torch.nn.MSELoss()
optim = torch.optim.Adam(net.parameters(),lr=0.01)

from torch.utils import data
train = data.TensorDataset(X,y)
train_set = data.DataLoader(train,batch_size=10)
epoch = 6
for i in range(epoch):
    for X,y in train_set:
        l = loss(net(X),y)
        optim.zero_grad()
        l.backward()
        optim.step()
        print(l.data)





