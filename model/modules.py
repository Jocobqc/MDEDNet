import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numbers
from model.network import *

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

class Cross_Atteneion_2(nn.Module):
    def __init__(self,dim, num_heads=4,LayerNorm_type='WithBias') :
        super().__init__()
        bias = False
        self.dim = dim
        self.norm_blur = LayerNorm(dim, LayerNorm_type)
        self.norm_event = LayerNorm(dim, LayerNorm_type)
        self.num_heads = num_heads
        self.blur_temperature = nn.Parameter(torch.ones(self.num_heads, 1, 1))
        self.event_temperature = nn.Parameter(torch.ones(self.num_heads, 1, 1))        
        # blur attention layer
        self.blur_q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.blur_k = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.blur_v = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.blur_project = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        # event attention layer
        self.event_q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.event_k = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.event_v = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.event_project = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
       
    def forward(self,blur,event):
        assert blur.shape == event.shape, 'The shape of feature maps from image and event branch are not equal!'

        b,c,h,w = blur.shape

        blur = self.norm_blur(blur)
        event = self.norm_event(event)

        blur_q = self.blur_q(blur)
        blur_k = self.blur_k(blur)
        blur_v = self.blur_v(blur)

        blur_q = rearrange(blur_q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        blur_k = rearrange(blur_k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        blur_v = rearrange(blur_v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        blur_q = torch.nn.functional.normalize(blur_q, dim=-1)
        blur_k = torch.nn.functional.normalize(blur_k, dim=-1)

        event_q = self.event_q(event)
        event_k = self.event_k(event)
        event_v = self.event_v(event)

        event_q = rearrange(event_q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        event_k = rearrange(event_k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        event_v = rearrange(event_v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        event_q = torch.nn.functional.normalize(event_q, dim=-1)
        event_k = torch.nn.functional.normalize(event_k, dim=-1)

        blur_att =(event_q @ blur_k.transpose(-2, -1)) * self.blur_temperature
        blur_att = blur_att.softmax(dim = -1)
        blur_res = (blur_att @ blur_v)
        blur_res = rearrange(blur_res, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        blur_res = self.blur_project(blur_res)
        
        event_att =(blur_q @ event_k.transpose(-2, -1)) * self.event_temperature
        event_att = event_att.softmax(dim = -1)
        event_res = (event_att @ event_v)
        event_res = rearrange(event_res, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        event_res = self.event_project(event_res)

        return blur_res,event_res

def Sobel(tensor):
    kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=tensor.dtype, device=tensor.device, requires_grad=False).unsqueeze(0).unsqueeze(0)
    kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=tensor.dtype, device=tensor.device, requires_grad=False).unsqueeze(0).unsqueeze(0)
    grad_x = F.conv2d(tensor, kernel_x.repeat(tensor.shape[1], 1, 1, 1), padding=1, groups=tensor.shape[1])
    grad_y = F.conv2d(tensor, kernel_y.repeat(tensor.shape[1], 1, 1, 1), padding=1, groups=tensor.shape[1])

    return grad_x+grad_y

def Laplacian(tensor):
    kernel = torch.tensor([[1,  1, 1],
                           [1, -8, 1],
                           [1,  1, 1]], dtype=tensor.dtype, device=tensor.device, requires_grad=False).unsqueeze(0).unsqueeze(0)                                 
    gradient = F.conv2d(tensor, kernel.repeat(tensor.shape[1], 1, 1, 1), padding=1, groups=tensor.shape[1])
    return gradient

class Cross_Modal_Multi_Order_Interaction(nn.Module):
    def __init__(self,dim,num_heads):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ca_0 = Cross_Atteneion_2(dim,num_heads)     
        self.w_b0 = nn.Parameter(torch.tensor(0.05, dtype = torch.float32, device = self.device), requires_grad = True)  
        self.w_e0 = nn.Parameter(torch.tensor(0.05, dtype = torch.float32, device = self.device), requires_grad = True)                                             
        self.ca_1 = Cross_Atteneion_2(dim,num_heads) 
        self.w_b1 = nn.Parameter(torch.tensor(0.05, dtype = torch.float32, device = self.device), requires_grad = True)  
        self.w_e1 = nn.Parameter(torch.tensor(0.05, dtype = torch.float32, device = self.device), requires_grad = True)                             
        self.ca_2 = Cross_Atteneion_2(dim,num_heads)  
        self.w_b2 = nn.Parameter(torch.tensor(0.05, dtype = torch.float32, device = self.device), requires_grad = True)
        self.w_e2 = nn.Parameter(torch.tensor(0.05, dtype = torch.float32, device = self.device), requires_grad = True)                                   
        self.blur_fuse = nn.Conv2d(dim*2, dim, kernel_size=1, stride=1)
        self.event_fuse = nn.Conv2d(dim*2, dim, kernel_size=1, stride=1)  
    def forward(self,blur,event):           
        blur_res = torch.zeros_like(blur)
        event_res = torch.zeros_like(event)       
        blur_res0, event_res0 = self.ca_0(blur,event)
        blur_res += blur_res0*self.w_b0
        event_res += event_res0*self.w_e0
        blur_g1 = Sobel(blur)
        blur_res1, event_res1 = self.ca_1(blur_g1,event)
        blur_res += blur_res1*self.w_b1
        event_res += event_res1*self.w_b1    
        blur_g2 = Laplacian(blur)
        blur_res2, event_res2 = self.ca_2(blur_g2,event)
        blur_res += blur_res2*self.w_b2
        event_res += event_res2*self.w_e2  
        blur_res = self.blur_fuse(torch.cat([blur_res,blur],dim=1))
        event_res = self.event_fuse(torch.cat([event_res,event],dim=1))
        return event_res,blur_res

class Spectral_Consistency_Enhancemnent(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv_fuse_0 = Conv(dim*2,dim,ksize=1)
        self.dwconv_3x3_1 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)
        self.dwconv_5x5_1 = nn.Conv2d(dim, dim, kernel_size=5, stride=1, padding=2, groups=dim)
        self.dwconv_7x7_1 = nn.Conv2d(dim, dim, kernel_size=7, stride=1, padding=3, groups=dim)
        self.conv_fuse_1 = Conv(dim,dim,ksize=1)
        self.dwconv_3x3_2 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)
        self.dwconv_5x5_2 = nn.Conv2d(dim, dim, kernel_size=5, stride=1, padding=2, groups=dim)
        self.dwconv_7x7_2 = nn.Conv2d(dim, dim, kernel_size=7, stride=1, padding=3, groups=dim)
        self.conv_fuse_2 = nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0)        
        self.conv_fuse_blur = Conv(dim*2,dim,ksize=1)
        self.conv_fuse_event = Conv(dim*2,dim,ksize=1)
        self.output_map = nn.Conv2d(dim, 1, kernel_size=3, stride=1, padding=1)         
    def forward(self, blur, event):
        f_fuse = self.conv_fuse_0(torch.cat((blur, event), dim=1))
        f_fuse = self.conv_fuse_1(self.dwconv_3x3_1(f_fuse)+self.dwconv_5x5_1(f_fuse)+self.dwconv_7x7_1(f_fuse))
        f_fuse = self.conv_fuse_2(self.dwconv_3x3_2(f_fuse)+self.dwconv_5x5_2(f_fuse)+self.dwconv_7x7_2(f_fuse))
        sc_map = torch.sigmoid(f_fuse)        
        blur_en = self.conv_fuse_blur(torch.cat([blur*sc_map,blur],dim=1))
        event_en = self.conv_fuse_event(torch.cat([event*sc_map,event],dim=1))
        return blur_en,event_en