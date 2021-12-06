### CHASE
[Paper](https://arxiv.org/abs/2107.03463)

Official implementation of the CHASE tracker. 

### Abstract
A strong visual object tracker nowadays relies on its well-crafted modules, which typically consist of manually-designed network architectures to deliver high-quality tracking results. Not surprisingly, the manual design process becomes a particularly challenging barrier, as it demands sufficient prior experience, enormous effort, intuition, and perhaps some good luck. Meanwhile, neural architecture search has gaining grounds in practical applications as a promising method in tackling the issue of automated search of feasible network structures. In this work, we propose a novel cell-level differentiable architecture search mechanism with early stopping to automate the network design of the tracking module, aiming to adapt backbone features to the objective of Siamese tracking networks during offline training. Besides, the proposed early stopping strategy avoids over-fitting and performance collapse problems leading to generalization improvement. The proposed approach is simple, efficient, and with no need to stack a series of modules to construct a network. Our approach is easy to be incorporated into existing trackers, which is empirically validated using different differentiable architecture search-based methods and tracking objectives. Extensive experimental evaluations demonstrate the superior performance of our approach over five commonly-used benchmarks. 


### Official Checkpoint
[ModelZoo](https://drive.google.com/drive/folders/1LgtXDCFWzHVAPfT9-k5-9i5uIkBz6Eis?usp=sharing)

### Main Contributors
* Javad Khaghani
* Seyed Mojtaba Marvasti-Zade

## Acknowledgments
* This implementation is based on [PyTracking](https://github.com/visionml/pytracking), [PrROIPooling](https://github.com/vacancy/PreciseRoIPooling), and [DARTS](https://github.com/quark0/darts).  

