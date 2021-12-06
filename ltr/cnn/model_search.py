import torch
import torch.nn as nn
import torch.nn.functional as F
from ltr.cnn.operations import *
from torch.autograd import Variable
from ltr.cnn.genotypes import PRIMITIVES
from ltr.cnn.genotypes import Genotype
from collections import OrderedDict
from ltr import MultiGPU

class MixedOp(nn.Module):
  def __init__(self, C, stride, p):
    super(MixedOp, self).__init__()
    self.p=p
    self._ops = nn.ModuleList()
    for primitive in PRIMITIVES:
      op = OPS[primitive](C, stride, True,conv_type=nn.Conv2d)
      if 'pool' in primitive:
        op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
      if isinstance(op, Identity) and p > 0:
        op = nn.Sequential(op, nn.Dropout(self.p))
      self._ops.append(op)


  def forward(self, x, weights):
    return sum(w * op(x) for w, op in zip(weights, self._ops))


class Cell(nn.Module):

  def __init__(self, steps, multiplier, C,p):
    super(Cell, self).__init__()

    self._steps = steps
    self._multiplier = multiplier

    self._ops = nn.ModuleList()
    self._bns = nn.ModuleList()
    for i in range(self._steps):
      for j in range(2+i):
        stride = 1
        op = MixedOp(C, stride,p)
        self._ops.append(op)

  def forward(self, feat3, feat4, weights):

    states = [feat3, feat4]
    offset = 0
    for i in range(self._steps):
      s = sum(self._ops[offset+j](h, weights[offset+j])for j, h in enumerate(states))
      offset += len(states)
      states.append(s)

    return torch.cat(states[-self._multiplier:], dim=1)



class Network_Search(nn.Module):

  def __init__(self, C, steps=4, multiplier=4, in_ch=512,out_ch=256,C_curr=256, p_f=0.6): #layers, num_classes,
    super(Network_Search, self).__init__()
    self._C = C
    self._steps = steps
    self._multiplier = multiplier

    self.C_curr= C_curr
    self.conv_match3 = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, stride=2,padding=1)
    self.conv_match4 = nn.Conv2d(in_channels=1024, out_channels=out_ch, kernel_size=1, padding=0)
    self.cells = nn.ModuleList()

    cell = Cell(steps, multiplier, C_curr,p_f)
    self.cells += [cell]
    self._initialize_alphas()

    for m in self.modules():
      if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight.data, mode='fan_in')
        if m.bias is not None:
          m.bias.data.zero_()

  def new(self):
    model_new = Network_Search(self._C).cuda()
    for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
        x.data.copy_(y.data)
    return model_new


  def NAS_forward(self,input_features):
    s0,s1=self.conv_match3(input_features['layer2']),self.conv_match4(input_features['layer3'])
    weights = F.softmax(self.alphas_normal, dim=-1)
    for i, cell in enumerate(self.cells):
      s0, s1 = s1, cell(s0, s1,weights)
    return s1

  def forward(self, feat):
    out_NAS= self.NAS_forward(feat)
    return out_NAS

  def _initialize_alphas(self):
    k = sum(1 for i in range(self._steps) for n in range(2+i))
    num_ops = len(PRIMITIVES)
    self.alphas_normal = Variable(1e-3*torch.randn(k, num_ops).cuda(), requires_grad=True)
    self._arch_parameters = [self.alphas_normal]


  def arch_parameters(self):
    return self._arch_parameters

  def genotype(self):

    def _parse(weights):
      gene = []
      n = 2
      start = 0
      for i in range(self._steps):
        end = start + n
        W = weights[start:end].copy()
        edges = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x])) ))[:2]
        for j in edges:
          k_best = None
          for k in range(len(W[j])):
              if k_best is None or W[j][k] > W[j][k_best]:
                k_best = k
          gene.append((PRIMITIVES[k_best], j))
        start = end
        n += 1
      return gene

    gene_normal = _parse(F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy())

    concat = range(2+self._steps-self._multiplier, self._steps+2)
    genotype = Genotype(normal=gene_normal, normal_concat=concat)
    return genotype

