import torch.nn as nn
import torch
from ltr.models.layers.blocks import LinearBlock
from ltr.external.PreciseRoIPooling.pytorch.prroi_pool import PrRoIPool2D


import math
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch.nn.modules.utils import _pair
from ltr.models.layers.conv2d_mtl import Conv2dMtl as conv_mtl



class BaseLearner(nn.Module):
    """The class for inner loop."""
    def __init__(self, in_dim):
        super().__init__()
        self.in_dim = in_dim
        self.vars = nn.ParameterList()
        fc1_w = nn.Parameter(torch.ones([1, self.in_dim], device='cuda'))
        #torch.nn.init.kaiming_normal_(self.fc1_w)
        self.vars.append(fc1_w)
        fc1_b = nn.Parameter(torch.zeros(1,device='cuda'))
        self.vars.append(fc1_b)

    def forward(self, input_x, the_vars=None):
        if the_vars is None:
            the_vars = self.vars
        fc1_w = the_vars[0]
        fc1_b = the_vars[1]
        net = F.linear(input_x, fc1_w, fc1_b)
        return net

    def parameters(self):
        return self.vars

def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
            conv_mtl(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=True),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True))

def conv_regular(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=True),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True))


class AtomSCRDetMTL(nn.Module):
    def __init__(self, input_dim, pred_input_dim, pred_inter_dim, pretrained_bbreg):
        super().__init__()

        self.main_network = Main_Network(input_dim=input_dim, pred_input_dim=pred_input_dim,pred_inter_dim=pred_inter_dim)
        self.iou_predictor = BaseLearner(pred_inter_dim[0] + pred_inter_dim[1])

        model_dict = self.state_dict()
        pre_dict = pretrained_bbreg.state_dict()
        pretrained_dict = {}
        # In current implementation, we can load the state dict with strict=False
        for k, v in pre_dict.items():
            if k.find('iou_predictor') != -1:
                if k.find('weight') != -1:
                    pretrained_dict.update({'iou_predictor.vars.0': v})
                if k.find('bias') != -1:
                    pretrained_dict.update({'iou_predictor.vars.1': v})
            else:
                new_k = 'main_network.' + k
                pretrained_dict.update({new_k: v})

        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)

    def forward(self, feat1, feat2, bb1, proposals2):
        fc34_rt_cat,batch_size, num_proposals_per_batch = self.main_network( feat1, feat2, bb1, proposals2)
        num_images=1
        iou_pred = self.iou_predictor(fc34_rt_cat).reshape(num_images, batch_size, num_proposals_per_batch)
        return iou_pred

    def predict_iou(self, modulation, feat, proposals):
        fc34_rt_cat,batch_size, num_proposals_per_batch = self.main_network.predict_fc34_rt(modulation, feat, proposals)
        num_images =1
        iou_pred = self.iou_predictor(fc34_rt_cat).reshape(num_images, batch_size, num_proposals_per_batch)
        return iou_pred

class Main_Network(nn.Module):
    """Network module for IoU prediction. Refer to the ATOM paper for an illustration of the architecture.
    It uses two backbone feature layers as input.
    args:
        input_dim:  Feature dimensionality of the two input backbone layers.
        pred_input_dim:  Dimensionality input the the prediction network.
        pred_inter_dim:  Intermediate dimensionality in the prediction network."""

    def __init__(self, input_dim=(128,256), pred_input_dim=(256,256), pred_inter_dim=(256,256)):
        super().__init__()
        # _r for reference, _t for test
        self.conv3_1r = conv_regular(input_dim[0], 128, kernel_size=3, stride=1)
        self.conv3_1t = conv(input_dim[0], 256, kernel_size=3, stride=1)

        self.conv3_2t = conv(256, pred_input_dim[0], kernel_size=3, stride=1)

        self.prroi_pool3r = PrRoIPool2D(3, 3, 1/8)
        self.prroi_pool3t = PrRoIPool2D(5, 5, 1/8)

        self.fc3_1r = conv_regular(128, 256, kernel_size=3, stride=1, padding=0)

        self.conv4_1r = conv_regular(input_dim[1], 256, kernel_size=3, stride=1)
        self.conv4_1t = conv(input_dim[1], 256, kernel_size=3, stride=1)

        self.conv4_2t = conv(256, pred_input_dim[1], kernel_size=3, stride=1)

        self.prroi_pool4r = PrRoIPool2D(1, 1, 1/16)
        self.prroi_pool4t = PrRoIPool2D(3, 3, 1 / 16)

        self.fc34_3r = conv_regular(256 + 256, pred_input_dim[0], kernel_size=1, stride=1, padding=0)
        self.fc34_4r = conv_regular(256 + 256, pred_input_dim[1], kernel_size=1, stride=1, padding=0)

        self.fc3_rt = LinearBlock(pred_input_dim[0], pred_inter_dim[0], 5)
        self.fc4_rt = LinearBlock(pred_input_dim[1], pred_inter_dim[1], 3)

    def forward(self, feat1, feat2, bb1, proposals2):
        """Runs the ATOM IoUNet during training operation.
        This forward pass is mainly used for training. Call the individual functions during tracking instead.
        args:
            feat1:  Features from the reference frames (4 or 5 dims).
            feat2:  Features from the test frames (4 or 5 dims).
            bb1:  Target boxes (x,y,w,h) in image coords in the reference samples. Dims (images, sequences, 4).
            proposals2:  Proposal boxes for which the IoU will be predicted (images, sequences, num_proposals, 4)."""

        assert bb1.dim() == 3
        assert proposals2.dim() == 4

        num_images = proposals2.shape[0]
        num_sequences = proposals2.shape[1]

        # Extract first train sample
        feat1 = [f[0,...] if f.dim()==5 else f.reshape(-1, num_sequences, *f.shape[-3:])[0,...] for f in feat1]
        bb1 = bb1[0,...]

        # Get modulation vector
        modulation = self.get_modulation(feat1, bb1)

        iou_feat = self.get_iou_feat(feat2)

        modulation = [f.reshape(1, num_sequences, -1).repeat(num_images, 1, 1).reshape(num_sequences*num_images, -1) for f in modulation]

        proposals2 = proposals2.reshape(num_sequences*num_images, -1, 4)
        fc34_rt_cat,batch_size, num_proposals_per_batch =self.predict_fc34_rt(modulation, iou_feat, proposals2)
        return fc34_rt_cat,batch_size, num_proposals_per_batch


    def predict_fc34_rt(self, modulation, feat, proposals):
        """Predicts IoU for the give proposals.
        args:
            modulation:  Modulation vectors for the targets. Dims (batch, feature_dim).
            feat:  IoU features (from get_iou_feat) for test images. Dims (batch, feature_dim, H, W).
            proposals:  Proposal boxes for which the IoU will be predicted (batch, num_proposals, 4)."""

        fc34_3_r, fc34_4_r = modulation
        c3_t, c4_t = feat

        batch_size = c3_t.size()[0]

        # Modulation
        c3_t_att = c3_t * fc34_3_r.reshape(batch_size, -1, 1, 1)
        c4_t_att = c4_t * fc34_4_r.reshape(batch_size, -1, 1, 1)

        # Add batch_index to rois
        batch_index = torch.arange(batch_size, dtype=torch.float32).reshape(-1, 1).to(c3_t.device)

        # Push the different rois for the same image along the batch dimension
        num_proposals_per_batch = proposals.shape[1]

        # input proposals2 is in format xywh, convert it to x0y0x1y1 format
        proposals_xyxy = torch.cat((proposals[:, :, 0:2], proposals[:, :, 0:2] + proposals[:, :, 2:4]), dim=2)#need cuda, but grads are destroyed

        # Add batch index
        roi2 = torch.cat((batch_index.reshape(batch_size, -1, 1).expand(-1, num_proposals_per_batch, -1),
                          proposals_xyxy), dim=2)
        roi2 = roi2.reshape(-1, 5).to(proposals_xyxy.device)

        roi3t = self.prroi_pool3t(c3_t_att, roi2)
        roi4t = self.prroi_pool4t(c4_t_att, roi2)

        fc3_rt = self.fc3_rt(roi3t)
        fc4_rt = self.fc4_rt(roi4t)

        fc34_rt_cat = torch.cat((fc3_rt, fc4_rt), dim=1)

        return fc34_rt_cat,batch_size, num_proposals_per_batch


    def get_modulation(self, feat, bb):
        """Get modulation vectors for the targets.
        args:
            feat: Backbone features from reference images. Dims (batch, feature_dim, H, W).
            bb:  Target boxes (x,y,w,h) in image coords in the reference samples. Dims (batch, 4)."""

        feat3_r, feat4_r = feat

        c3_r = self.conv3_1r(feat3_r)

        # Add batch_index to rois
        batch_size = bb.shape[0]
        batch_index = torch.arange(batch_size, dtype=torch.float32).reshape(-1, 1).to(bb.device)

        # input bb is in format xywh, convert it to x0y0x1y1 format
        bb = bb.clone()
        bb[:, 2:4] = bb[:, 0:2] + bb[:, 2:4]
        roi1 = torch.cat((batch_index, bb), dim=1)

        roi3r = self.prroi_pool3r(c3_r, roi1)

        c4_r = self.conv4_1r(feat4_r)
        roi4r = self.prroi_pool4r(c4_r, roi1)

        fc3_r = self.fc3_1r(roi3r)

        # Concatenate from block 3 and 4
        fc34_r = torch.cat((fc3_r, roi4r), dim=1)

        fc34_3_r = self.fc34_3r(fc34_r)
        fc34_4_r = self.fc34_4r(fc34_r)

        return fc34_3_r, fc34_4_r

    #
    def get_iou_feat(self, feat2):
        """Get IoU prediction features from a 4 or 5 dimensional backbone input."""
        feat2 = [f.reshape(-1, *f.shape[-3:]) if f.dim()==5 else f for f in feat2]
        feat3_t, feat4_t = feat2
        c3_t = self.conv3_2t(self.conv3_1t(feat3_t))
        c4_t = self.conv4_2t(self.conv4_1t(feat4_t))

        return c3_t, c4_t
