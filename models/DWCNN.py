import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
from .Model import Model
import utils
import math


mean = Variable(torch.FloatTensor([0.485, 0.456, 0.406]), requires_grad=False).cuda()
std = Variable(torch.FloatTensor([0.229, 0.224, 0.225]), requires_grad=False).cuda()
device = 'cuda' 


def flip(x, dim):
    xsize = x.size()
    dim = x.dim() + dim if dim < 0 else dim
    x = x.view(-1, *xsize[dim:])
    x = x.view(x.size(0), x.size(1), -1)[:, getattr(torch.arange(x.size(1)-1, 
                      -1, -1), ('cpu','cuda')[x.is_cuda])().long(), :]
    
    return x.view(xsize)


class DWCNNS(Model):

    def __init__(self, name, nclasses=40, pretraining=True, cnn_name='resnet50'):
        super(DWCNNS, self).__init__(name)

        self.classnames=['airplane','bathtub','bed','bench','bookshelf','bottle','bowl','car','chair',
                         'cone','cup','curtain','desk','door','dresser','flower_pot','glass_box',
                         'guitar','keyboard','lamp','laptop','mantel','monitor','night_stand',
                         'person','piano','plant','radio','range_hood','sink','sofa','stairs',
                         'stool','table','tent','toilet','tv_stand','vase','wardrobe','xbox']

        self.nclasses = nclasses
        self.pretraining = pretraining
        self.cnn_name = cnn_name
        self.use_resnet = cnn_name.startswith('resnet50')
        self.mean = Variable(torch.FloatTensor([0.485, 0.456, 0.406]), requires_grad=False).cuda()
        self.std = Variable(torch.FloatTensor([0.229, 0.224, 0.225]), requires_grad=False).cuda()
        
        self.aggregator = utils.Max_Viewpooling()
    
        if self.use_resnet:
            print('model: resnet50')
            self.net = models.resnet50(pretrained=self.pretraining)
            self.net.fc = nn.Linear(2048,40)

        else:
            print('No backbone!')


    def forward(self, x, get_ft=False):
        if self.use_resnet:
            y = self.net(x)

            if get_ft:
                ft = self.aggregator(x)
                return y, ft
            else:
                return y
        else:
            print('No backbone!!')

            
def fc_bn_block(input, output):
    return nn.Sequential(
        nn.Linear(input, output),
        nn.BatchNorm1d(output),
        nn.ReLU(inplace=True))            


def cal_scores(scores):
        n = len(scores)
        s = 0
        for score in scores:
            s += torch.ceil(score*n)
        s /= n
        return s

    
def group_fusion(view_group): 
    shape_des = sum(view_group)/len(view_group)
    return shape_des


def group_pooling(final_views, views_score, group_num):
    interval = 1.0 / group_num

    def onebatch_grouping(onebatch_views, onebatch_scores):
        viewgroup_onebatch = [[] for i in range(group_num)]
        scoregroup_onebatch = [[] for i in range(group_num)]

        for i in range(group_num):
            left = i*interval 
            right = (i+1)*interval            
            for j, score in enumerate(onebatch_scores):
                if left<=score<right:
                    viewgroup_onebatch[i].append(onebatch_views[j])
                    scoregroup_onebatch[i].append(score)
                else:
                    pass

        view_group = [sum(views)/len(views) for views in viewgroup_onebatch if len(views)>0] 

        onebatch_shape_des = group_fusion(view_group) 
        return onebatch_shape_des
    
    shape_descriptors = []
    for (onebatch_views,onebatch_scores) in zip(final_views,views_score):
        shape_descriptors.append(onebatch_grouping(onebatch_views,onebatch_scores))
    shape_descriptor = torch.stack(shape_descriptors, 0)

    return shape_descriptor


class DWCNNM(Model):

    def __init__(self, name, model, nclasses=40, cnn_name='resnet50', num_views=20):
        super(DWCNNM, self).__init__(name)

        self.classnames=['airplane','bathtub','bed','bench','bookshelf','bottle','bowl','car','chair',
                         'cone','cup','curtain','desk','door','dresser','flower_pot','glass_box',
                         'guitar','keyboard','lamp','laptop','mantel','monitor','night_stand',
                         'person','piano','plant','radio','range_hood','sink','sofa','stairs',
                         'stool','table','tent','toilet','tv_stand','vase','wardrobe','xbox']

        self.nclasses = nclasses
        self.num_views = num_views
        self.mean = Variable(torch.FloatTensor([0.485, 0.456, 0.406]), requires_grad=False).cuda()
        self.std = Variable(torch.FloatTensor([0.229, 0.224, 0.225]), requires_grad=False).cuda()

        self.group_num = 3
        
        self.num_layers = 1
        self.hidden_size = 1024
        self.lstm_layer=torch.nn.LSTM(input_size=1024,hidden_size=1024,num_layers=1,
                        bias=True,batch_first=True,dropout=0,bidirectional=True)
        
   
        self.use_resnet = cnn_name.startswith('resnet50')
    

        self.final = nn.Linear(1024*2, 1024)
      
        self.FC = nn.Sequential(fc_bn_block(2048, 768),
                                fc_bn_block(768,1))        

        if self.use_resnet:
            self.net_1 = nn.Sequential(*list(model.net.children())[:-1])
            self.net_2 = model.net.fc
        else:
            print('No backbone!!!')

    def forward(self, x, get_ft=False):

        x = self.net_1(x)

        batch_size = int(x.shape[0]/self.num_views)
        final_views = x.view((int(x.shape[0]/self.num_views),self.num_views,-1))
        h0 = torch.zeros(self.num_layers*2, final_views.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers*2, final_views.size(0), self.hidden_size).to(device)       
        final_views = self.final(final_views)   
        final_views, _ = self.lstm_layer(final_views, (h0, c0))

        x = x.view((int(x.shape[0]),-1))
        views_score = self.FC(x)
        views_score = views_score.view(batch_size, self.num_views, -1)
        views_score = F.normalize(views_score,p=2,dim=1)
        views_score = torch.sigmoid(views_score*4)

        shape_descriptor = group_pooling(final_views, views_score, self.group_num) 

        y = self.net_2(shape_descriptor)

        if get_ft:
            ft = shape_descriptor
            return y, ft
        else:
            return y
    

