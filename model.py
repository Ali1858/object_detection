from torch import nn
import torch
import torchvision
from torch.nn.init import kaiming_normal

class AdaptiveConcatPool2d(nn.Module):
    def __init__(self, sz=None):
        super().__init__()
        sz = sz or (1,1)
        self.ap = nn.AdaptiveAvgPool2d(sz)
        self.mp = nn.AdaptiveMaxPool2d(sz)
    def forward(self, x): return torch.cat([self.mp(x), self.ap(x)], 1)
    

class Lambda(nn.Module):
    def __init__(self, f): super().__init__(); self.f=f
    def forward(self, x): return self.f(x)

class Flatten(nn.Module):
    def __init__(self): super().__init__()
    def forward(self, x): return x.view(x.size(0), -1)

    
def apply_init(m, init_fn):
        m.apply(lambda x: cond_init(x, init_fn))
        
def cond_init(m, init_fn):
    if not isinstance(m, (nn.BatchNorm1d,nn.BatchNorm2d,nn.BatchNorm3d)):
        if hasattr(m, 'weight'): init_fn(m.weight)
        if hasattr(m, 'bias') and hasattr(m.bias, 'data'): m.bias.data.fill_(0.)
            
def to_gpu(x, *args, **kwargs):
    '''puts pytorch variable to gpu, if cuda is avaialble and USE_GPU is set to true. '''
    return x.cuda(*args, **kwargs) if USE_GPU else x

    
class Model(nn.Module):
    
    def __init__(self,num_classes,is_reg=False, is_multi=False,custom_head=None):
        super(Model,self).__init__()
        
        resnet34 = torchvision.models.resnet34(pretrained=True)
        
        for params in resnet34.parameters():
            params.requires_grad = False
            
        layers = list(resnet34.children())[:8]
        
        if custom_head: fc_layers = [custom_head]
        else: fc_layers = self.create_fc_layer(512,num_classes,None)              
        #nn.Linear(self.resnet34.fc.in_features,num_classes)
        
        self.custom_model = nn.Sequential(*(layers+fc_layers))
        
        self.is_reg = is_reg
        
        
        '''
        layers = list(resnet34.children())[:8]
        
        for layer in layers:
            print('no param')
            layer.requires_grad = False
        
        layers += [AdaptiveConcatPool2d(), Flatten()]
        
        self.nf = self.num_features(layers)*2
        
        self.c = num_classes
        
        self.ps = [None,None]
        
        self.xtra_fc  =[512]
        
        self.is_reg = is_reg
        
        self.is_multi = is_multi
        
        
        fc_layers = self.get_fc_layers()
        fc_model = nn.Sequential(*fc_layers)
        #apply_init(fc_model, kaiming_normal)
        self.model = nn.Sequential(*(layers+fc_layers))
        
        
        #for i,params in enumerate(self.model.children()):
        #    if i < 8:
        #        print('setting params not req')
        #        params.requires_grad = False
        #        
        
        '''
    

   
    def create_fc_layer(self, ni, nf, p, actn=None):
        res=[nn.BatchNorm1d(num_features=ni)]
        if p: res.append(nn.Dropout(p=p))
        res.append(nn.Linear(in_features=ni, out_features=nf))
        if actn: res.append(actn)
        return res

    def get_fc_layers(self):
        res=[]
        ni=self.nf
        for i,nf in enumerate(self.xtra_fc):
            res += self.create_fc_layer(ni, nf, p=self.ps[i], actn=nn.ReLU())
            ni=nf
        final_actn = nn.Sigmoid() if self.is_multi else nn.LogSoftmax()
        if self.is_reg: final_actn = None
        res += self.create_fc_layer(ni, self.c, p=self.ps[-1], actn=final_actn)
        return res

        
    def get_children(self,m): return m if isinstance(m, (list, tuple)) else list(m.children())

    def num_features(self,m):
        c=self.get_children(m)
        if len(c)==0: return None
        for l in reversed(c):
            if hasattr(l, 'num_features'): return l.num_features
            res = self.num_features(l)
            if res is not None: return res
        
    def forward(self,x):
        
        return self.custom_model(x)
             
        
        