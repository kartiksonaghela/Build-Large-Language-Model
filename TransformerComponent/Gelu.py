import torch
class gelu(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x):
        return 0.5*x*(1+torch.tanh(torch.sqrt(torch.tensor(2/torch.pi))*(x+0.044715*torch.pow(x,3))))

gl=gelu()
inputs = torch.tensor([
    [0.43, 0.15, 0.89],  
    [0.55, 0.87, 0.66],  
    [0.57, 0.85, 0.64],  
    [0.22, 0.58, 0.33],  
    [0.77, 0.25, 0.10],  
    [0.05, 0.80, 0.55]   
])
print(gl(inputs))