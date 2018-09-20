from torch import nn
import torch
import torchvision

class Model(nn.Module):
    
    def __init__(self,num_classes,is_reg=False, is_multi=False,custom_head=None):
        super(Model,self).__init__()
        
        # load pretrained resnet34
        resnet34 = torchvision.models.resnet34(pretrained=True)
        

        for params in resnet34.parameters():
            params.requires_grad = False
            
        layers = list(resnet34.children())[:8]
        
        # at custom head if any 
        if custom_head:
            fc_layers = [custom_head]
            self.custom_model = nn.Sequential(*(layers+fc_layers))
        # or add linear layer with number of classes as output
        else:
            resnet34.fc = nn.Linear(resnet34.fc.in_features,num_classes)
            self.custom_model = resnet34        
                
    def forward(self,x):
        #predict
        return self.custom_model(x)
             
        
        