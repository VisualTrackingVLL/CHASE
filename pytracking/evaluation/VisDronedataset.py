import numpy as np
from pytracking.evaluation.data import Sequence, BaseDataset, SequenceList


#def VisDataset():
#    return VisDatasetClass().get_sequence_list()


class VisDataset(BaseDataset):
    """ UAV123 dataset.
    Publication:
        A Benchmark and Simulator for UAV Tracking.
        Matthias Mueller, Neil Smith and Bernard Ghanem
        ECCV, 2016
        https://ivul.kaust.edu.sa/Documents/Publications/2016/A%20Benchmark%20and%20Simulator%20for%20UAV%20Tracking.pdf
    Download the dataset from https://ivul.kaust.edu.sa/Pages/pub-benchmark-simulator-uav.aspx
    """
    def __init__(self):
        super().__init__()
        self.base_path = self.env_settings.vis_path
        self.sequence_info_list = self._get_sequence_info_list()

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_info_list])

    def _construct_sequence(self, sequence_info):
        sequence_path = sequence_info['path']
        nz = sequence_info['nz']
        ext = sequence_info['ext']
        start_frame = sequence_info['startFrame']
        end_frame = sequence_info['endFrame']

        init_omit = 0
        if 'initOmit' in sequence_info:
            init_omit = sequence_info['initOmit']

        frames = ['{base_path}/{sequence_path}/img{frame:0{nz}}.{ext}'.format(base_path=self.base_path,
            sequence_path=sequence_path, frame=frame_num, nz=nz, ext=ext) for frame_num in range(start_frame+init_omit, end_frame+1)]

        anno_path = '{}{}'.format(self.base_path, sequence_info['anno_path'])

        try:
            ground_truth_rect = np.loadtxt(str(anno_path), dtype=np.float64)
        except:
            ground_truth_rect = np.loadtxt(str(anno_path), delimiter=',', dtype=np.float64)

        return Sequence(sequence_info['name'], frames, 'vis', ground_truth_rect[init_omit:,:])

    def __len__(self):
        return len(self.sequence_info_list)


    def _get_sequence_info_list(self):
        sequence_info_list = [
             {"name": "uav0000011_00000_s", "path": "sequences/uav0000011_00000_s", "startFrame": 1, "endFrame": 369,
             "nz": 7, "ext": "jpg", "anno_path": "annotations/uav0000011_00000_s.txt"},
            {"name": "uav0000021_00000_s", "path": "sequences/uav0000021_00000_s", "startFrame": 1, "endFrame": 581,
             "nz": 7, "ext": "jpg", "anno_path": "annotations/uav0000021_00000_s.txt"},
            {"name": "uav0000069_00576_s", "path": "sequences/uav0000069_00576_s", "startFrame": 1, "endFrame": 97,
             "nz": 7, "ext": "jpg", "anno_path": "annotations/uav0000069_00576_s.txt"},
            {"name": "uav0000074_01656_s", "path": "sequences/uav0000074_01656_s", "startFrame": 1, "endFrame": 457,
             "nz": 7, "ext": "jpg", "anno_path": "annotations/uav0000074_01656_s.txt"},
            {"name": "uav0000074_04320_s", "path": "sequences/uav0000074_04320_s", "startFrame": 1, "endFrame": 721,
             "nz": 7, "ext": "jpg", "anno_path": "annotations/uav0000074_04320_s.txt"},
            {"name": "uav0000074_04992_s", "path": "sequences/uav0000074_04992_s", "startFrame": 1, "endFrame": 625,
             "nz": 7, "ext": "jpg", "anno_path": "annotations/uav0000074_04992_s.txt"},
            {"name": "uav0000074_05712_s", "path": "sequences/uav0000074_05712_s", "startFrame": 1, "endFrame": 361,
             "nz": 7, "ext": "jpg", "anno_path": "annotations/uav0000074_05712_s.txt"},
            {"name": "uav0000074_06312_s", "path": "sequences/uav0000074_06312_s", "startFrame": 1, "endFrame": 649,
             "nz": 7, "ext": "jpg", "anno_path": "annotations/uav0000074_06312_s.txt"},
            {"name": "uav0000074_11915_s", "path": "sequences/uav0000074_11915_s", "startFrame": 1, "endFrame": 974,
             "nz": 7, "ext": "jpg", "anno_path": "annotations/uav0000074_11915_s.txt"},
            {"name": "uav0000079_02568_s", "path": "sequences/uav0000079_02568_s", "startFrame": 1, "endFrame": 1297,
             "nz": 7, "ext": "jpg", "anno_path": "annotations/uav0000079_02568_s.txt"},
            {"name": "uav0000088_00000_s", "path": "sequences/uav0000088_00000_s", "startFrame": 1, "endFrame": 1920,
             "nz": 7, "ext": "jpg", "anno_path": "annotations/uav0000088_00000_s.txt"},
            {"name": "uav0000093_00000_s", "path": "sequences/uav0000093_00000_s", "startFrame": 1, "endFrame": 1611,
             "nz": 7, "ext": "jpg", "anno_path": "annotations/uav0000093_00000_s.txt"},
            {"name": "uav0000093_01817_s", "path": "sequences/uav0000093_01817_s", "startFrame": 1, "endFrame": 553,
             "nz": 7, "ext": "jpg", "anno_path": "annotations/uav0000093_01817_s.txt"},
            {"name": "uav0000116_00503_s", "path": "sequences/uav0000116_00503_s", "startFrame": 1, "endFrame": 369,
             "nz": 7, "ext": "jpg", "anno_path": "annotations/uav0000116_00503_s.txt"},
            {"name": "uav0000151_00000_s", "path": "sequences/uav0000151_00000_s", "startFrame": 1, "endFrame": 2332,
             "nz": 7, "ext": "jpg", "anno_path": "annotations/uav0000151_00000_s.txt"},
            {"name": "uav0000155_01201_s", "path": "sequences/uav0000155_01201_s", "startFrame": 1, "endFrame": 671,
             "nz": 7, "ext": "jpg", "anno_path": "annotations/uav0000155_01201_s.txt"},
            {"name": "uav0000164_00000_s", "path": "sequences/uav0000164_00000_s", "startFrame": 1, "endFrame": 1868,
             "nz": 7, "ext": "jpg", "anno_path": "annotations/uav0000164_00000_s.txt"},
            {"name": "uav0000180_00050_s", "path": "sequences/uav0000180_00050_s", "startFrame": 1, "endFrame": 1251,
             "nz": 7, "ext": "jpg", "anno_path": "annotations/uav0000180_00050_s.txt"},
            {"name": "uav0000184_00625_s", "path": "sequences/uav0000184_00625_s", "startFrame": 1, "endFrame": 1101,
             "nz": 7, "ext": "jpg", "anno_path": "annotations/uav0000184_00625_s.txt"},
            {"name": "uav0000207_00675_s", "path": "sequences/uav0000207_00675_s", "startFrame": 1, "endFrame": 1626,
             "nz": 7, "ext": "jpg", "anno_path": "annotations/uav0000207_00675_s.txt"},
            {"name": "uav0000208_00000_s", "path": "sequences/uav0000208_00000_s", "startFrame": 1, "endFrame": 1169,
             "nz": 7, "ext": "jpg", "anno_path": "annotations/uav0000208_00000_s.txt"},
            {"name": "uav0000241_00001_s", "path": "sequences/uav0000241_00001_s", "startFrame": 1, "endFrame": 2783,
             "nz": 7, "ext": "jpg", "anno_path": "annotations/uav0000241_00001_s.txt"},
            {"name": "uav0000242_02327_s", "path": "sequences/uav0000242_02327_s", "startFrame": 1, "endFrame": 2569,
             "nz": 7, "ext": "jpg", "anno_path": "annotations/uav0000242_02327_s.txt"},
            {"name": "uav0000242_05160_s", "path": "sequences/uav0000242_05160_s", "startFrame": 1, "endFrame": 828,
             "nz": 7, "ext": "jpg", "anno_path": "annotations/uav0000242_05160_s.txt"},
            {"name": "uav0000294_00000_s", "path": "sequences/uav0000294_00000_s", "startFrame": 1, "endFrame": 253,
             "nz": 7, "ext": "jpg", "anno_path": "annotations/uav0000294_00000_s.txt"},
            {"name": "uav0000294_00069_s", "path": "sequences/uav0000294_00069_s", "startFrame": 1, "endFrame": 793,
             "nz": 7, "ext": "jpg", "anno_path": "annotations/uav0000294_00069_s.txt"},
            {"name": "uav0000294_01449_s", "path": "sequences/uav0000294_01449_s", "startFrame": 1, "endFrame": 392,
             "nz": 7, "ext": "jpg", "anno_path": "annotations/uav0000294_01449_s.txt"},
            {"name": "uav0000324_00069_s", "path": "sequences/uav0000324_00069_s", "startFrame": 1, "endFrame": 392,
             "nz": 7, "ext": "jpg", "anno_path": "annotations/uav0000324_00069_s.txt"},
            {"name": "uav0000340_01356_s", "path": "sequences/uav0000340_01356_s", "startFrame": 1, "endFrame": 529,
             "nz": 7, "ext": "jpg", "anno_path": "annotations/uav0000340_01356_s.txt"},
            {"name": "uav0000353_00001_s", "path": "sequences/uav0000353_00001_s", "startFrame": 1, "endFrame": 191,
             "nz": 7, "ext": "jpg", "anno_path": "annotations/uav0000353_00001_s.txt"},
            {"name": "uav0000353_01127_s", "path": "sequences/uav0000353_01127_s", "startFrame": 1, "endFrame": 438,
             "nz": 7, "ext": "jpg", "anno_path": "annotations/uav0000353_01127_s.txt"},
            {"name": "uav0000367_02761_s", "path": "sequences/uav0000367_02761_s", "startFrame": 1, "endFrame": 1322,
             "nz": 7, "ext": "jpg", "anno_path": "annotations/uav0000367_02761_s.txt"},
            {"name": "uav0000367_04137_s", "path": "sequences/uav0000367_04137_s", "startFrame": 1, "endFrame": 1429,
             "nz": 7, "ext": "jpg", "anno_path": "annotations/uav0000367_04137_s.txt"},
            {"name": "uav0000368_03312_s", "path": "sequences/uav0000368_03312_s", "startFrame": 1, "endFrame": 311,
             "nz": 7, "ext": "jpg", "anno_path": "annotations/uav0000368_03312_s.txt"},
            {"name": "uav0000368_03612_s", "path": "sequences/uav0000368_03612_s", "startFrame": 1, "endFrame": 90,
             "nz": 7, "ext": "jpg", "anno_path": "annotations/uav0000368_03612_s.txt"}]

        return sequence_info_list
