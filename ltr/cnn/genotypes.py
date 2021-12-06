from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat')
PrDiMP_NAS=Genotype(normal=[('avg_pool_3x3', 0), ('sep_conv_5x5', 1), ('skip_connect', 1), ('sep_conv_3x3', 2), ('skip_connect', 1), ('dil_conv_5x5', 0), ('skip_connect', 1), ('skip_connect', 4)], normal_concat=range(2, 6))

#PrDiMP_Fair=Genotype(normal=[('dil_conv_3x3', 0), ('skip_connect', 1), ('dil_conv_3x3', 0), ('dil_conv_3x3', 2), ('dil_conv_3x3', 0), ('dil_conv_3x3', 2), ('dil_conv_3x3', 0), ('skip_connect', 1)], normal_concat=range(2, 6))
#NoWeightFreeOp_NAS = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('dil_conv_3x3', 3), ('dil_conv_3x3', 1), ('sep_conv_3x3', 4), ('dil_conv_3x3', 3)], normal_concat=range(2, 6))

DARTS = PrDiMP_NAS
PRIMITIVES = [
    'none',
    'max_pool_3x3',
    'avg_pool_3x3',
    'skip_connect',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5'
]
