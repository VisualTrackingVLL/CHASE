from ltr.cnn.operations import *
from ltr.cnn.utils import drop_path
from ltr.models.layers.normalization import InstanceL2Norm
from torch.nn.parameter import Parameter



class Cell(nn.Module):

  def __init__(self, genotype, C, conv_type,):
    super(Cell, self).__init__()

    op_names, indices = zip(*genotype.normal)
    concat = genotype.normal_concat
    self.conv_type=conv_type
    self._compile(C, op_names, indices, concat)

  def _compile(self, C, op_names, indices, concat):
    assert len(op_names) == len(indices)
    self._steps = len(op_names) // 2
    self._concat = concat
    self.multiplier = len(concat)

    self._ops = nn.ModuleList()
    for name, index in zip(op_names, indices):
      stride = 1
      op = OPS[name](C, stride, True,conv_type=self.conv_type)
      self._ops += [op]
    self._indices = indices


  def forward(self, feat3, feat4, drop_prob):

    states = [feat3, feat4]
    for i in range(self._steps):
      h1 = states[self._indices[2 * i]]
      h2 = states[self._indices[2 * i + 1]]
      op1 = self._ops[2 * i]
      op2 = self._ops[2 * i + 1]
      h1 = op1(h1)
      h2 = op2(h2)
      if self.training and drop_prob > 0.:
        if not isinstance(op1, Identity):
          h1 = drop_path(h1, drop_prob)
        if not isinstance(op2, Identity):
          h2 = drop_path(h2, drop_prob)
      s = h1 + h2
      states += [s]
    return torch.cat([states[i] for i in self._concat], dim=1)


class NetworkFinal(nn.Module):

  def __init__(self, genotype,in_ch=512,out_ch=256,C_curr = 256,conv_type=nn.Conv2d):
    super(NetworkFinal, self).__init__()
    self.drop_path_prob=0 #Change it later
    self.C_curr= C_curr
    self.conv_match3 = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, stride=2,padding=1)
    self.conv_match4 = nn.Conv2d(in_channels=1024, out_channels=out_ch, kernel_size=1, padding=0)
    self.conv_type= conv_type
    self.cells = nn.ModuleList()
    cell = Cell(genotype, self.C_curr,conv_type=self.conv_type)
    self.cells += [cell]

    for m in self.modules():
      if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight.data, mode='fan_in')
        if m.bias is not None:
          m.bias.data.zero_()


  def NAS_forward(self,input_features):
    s0, s1 = self.conv_match3(input_features['layer2']), self.conv_match4(input_features['layer3'])
    for i, cell in enumerate(self.cells):
      s0, s1 = s1, cell(s0, s1, 0)
    return s1


  def forward(self, feat):
    out_NAS= self.NAS_forward(feat)
    return out_NAS

