import numpy as np
from pytracking.evaluation.data import Sequence, BaseDataset, SequenceList


#def UAVDTDataset():
#    return UAVDTDatasetClass().get_sequence_list()


class UAVDTDataset(BaseDataset):
    """ UAVDT dataset.

    """
    def __init__(self):
        super().__init__()
        self.base_path = self.env_settings.uavdt_path
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

        return Sequence(sequence_info['name'], frames, 'UAVDT', ground_truth_rect[init_omit:,:])

    def __len__(self):
        return len(self.sequence_info_list)


    def _get_sequence_info_list(self):
        sequence_info_list = [
            {"name": "S0101", "path": "sequences/S0101", "startFrame": 1, "endFrame": 1784, "nz": 6, "ext": "jpg", "anno_path": "annotations/S0101_gt.txt"},
            {"name": "S0102", "path": "sequences/S0102", "startFrame": 1, "endFrame": 350, "nz": 6, "ext": "jpg", "anno_path": "annotations/S0102_gt.txt"},
            {"name": "S0103", "path": "sequences/S0103", "startFrame": 1, "endFrame": 1135, "nz": 6, "ext": "jpg", "anno_path": "annotations/S0103_gt.txt"},
            {"name": "S0201", "path": "sequences/S0201", "startFrame": 1, "endFrame": 948, "nz": 6, "ext": "jpg", "anno_path": "annotations/S0201_gt.txt"},
            {"name": "S0301", "path": "sequences/S0301", "startFrame": 1, "endFrame": 695, "nz": 6, "ext": "jpg", "anno_path": "annotations/S0301_gt.txt"},
            {"name": "S0302", "path": "sequences/S0302", "startFrame": 1, "endFrame": 440, "nz": 6, "ext": "jpg", "anno_path": "annotations/S0302_gt.txt"},
            {"name": "S0303", "path": "sequences/S0303", "startFrame": 1, "endFrame": 200, "nz": 6, "ext": "jpg", "anno_path": "annotations/S0303_gt.txt"},
            {"name": "S0304", "path": "sequences/S0304", "startFrame": 1, "endFrame": 359, "nz": 6, "ext": "jpg", "anno_path": "annotations/S0304_gt.txt"},
            {"name": "S0305", "path": "sequences/S0305", "startFrame": 1, "endFrame": 706,"nz": 6, "ext": "jpg", "anno_path": "annotations/S0305_gt.txt"},
            {"name": "S0306", "path": "sequences/S0306", "startFrame": 1, "endFrame": 295, "nz": 6, "ext": "jpg", "anno_path": "annotations/S0306_gt.txt"},
            {"name": "S0307", "path": "sequences/S0307", "startFrame": 1, "endFrame": 414,"nz": 6, "ext": "jpg", "anno_path": "annotations/S0307_gt.txt"},
            {"name": "S0308", "path": "sequences/S0308", "startFrame": 1, "endFrame": 319,"nz": 6, "ext": "jpg", "anno_path": "annotations/S0308_gt.txt"},
            {"name": "S0309", "path": "sequences/S0309", "startFrame": 1, "endFrame": 214,"nz": 6, "ext": "jpg", "anno_path": "annotations/S0309_gt.txt"},
            {"name": "S0310", "path": "sequences/S0310", "startFrame": 1, "endFrame": 118, "nz": 6, "ext": "jpg", "anno_path": "annotations/S0310_gt.txt"},
            {"name": "S0401", "path": "sequences/S0401", "startFrame": 1, "endFrame": 501,"nz": 6, "ext": "jpg", "anno_path": "annotations/S0401_gt.txt"},
            {"name": "S0402", "path": "sequences/S0402", "startFrame": 1, "endFrame": 561,"nz": 6, "ext": "jpg", "anno_path": "annotations/S0402_gt.txt"},
            {"name": "S0501", "path": "sequences/S0501", "startFrame": 1, "endFrame": 232,"nz": 6, "ext": "jpg", "anno_path": "annotations/S0501_gt.txt"},
            {"name": "S0601", "path": "sequences/S0601", "startFrame": 1, "endFrame": 82, "nz": 6, "ext": "jpg", "anno_path": "annotations/S0601_gt.txt"},
            {"name": "S0602", "path": "sequences/S0602", "startFrame": 1, "endFrame": 292,"nz": 6, "ext": "jpg", "anno_path": "annotations/S0602_gt.txt"},
            {"name": "S0701", "path": "sequences/S0701", "startFrame": 1, "endFrame": 596,"nz": 6, "ext": "jpg", "anno_path": "annotations/S0701_gt.txt"},
            {"name": "S0801", "path": "sequences/S0801", "startFrame": 1, "endFrame": 526,"nz": 6, "ext": "jpg", "anno_path": "annotations/S0801_gt.txt"},
            {"name": "S0901", "path": "sequences/S0901", "startFrame": 1, "endFrame": 350,"nz": 6, "ext": "jpg", "anno_path": "annotations/S0901_gt.txt"},
            {"name": "S1001", "path": "sequences/S1001", "startFrame": 1, "endFrame": 353,"nz": 6, "ext": "jpg", "anno_path": "annotations/S1001_gt.txt"},
            {"name": "S1101", "path": "sequences/S1101", "startFrame": 1, "endFrame": 298,"nz": 6, "ext": "jpg", "anno_path": "annotations/S1101_gt.txt"},
            {"name": "S1201", "path": "sequences/S1201", "startFrame": 1, "endFrame": 2534,"nz": 6, "ext": "jpg", "anno_path": "annotations/S1201_gt.txt"},
            {"name": "S1202", "path": "sequences/S1202", "startFrame": 1, "endFrame": 329,"nz": 6, "ext": "jpg", "anno_path": "annotations/S1202_gt.txt"},
            {"name": "S1301", "path": "sequences/S1301", "startFrame": 1, "endFrame": 537,"nz": 6, "ext": "jpg", "anno_path": "annotations/S1301_gt.txt"},
            {"name": "S1302", "path": "sequences/S1302", "startFrame": 1, "endFrame": 403,"nz": 6, "ext": "jpg", "anno_path": "annotations/S1302_gt.txt"},
            {"name": "S1303", "path": "sequences/S1303", "startFrame": 1, "endFrame": 1112,"nz": 6, "ext": "jpg", "anno_path": "annotations/S1303_gt.txt"},
            {"name": "S1304", "path": "sequences/S1304", "startFrame": 1, "endFrame": 519,"nz": 6, "ext": "jpg", "anno_path": "annotations/S1304_gt.txt"},
            {"name": "S1305", "path": "sequences/S1305", "startFrame": 1, "endFrame": 1378,"nz": 6, "ext": "jpg", "anno_path": "annotations/S1305_gt.txt"},
            {"name": "S1306", "path": "sequences/S1306", "startFrame": 1, "endFrame": 2435,"nz": 6, "ext": "jpg", "anno_path": "annotations/S1306_gt.txt"},
            {"name": "S1307", "path": "sequences/S1307", "startFrame": 1, "endFrame": 742,"nz": 6, "ext": "jpg", "anno_path": "annotations/S1307_gt.txt"},
            {"name": "S1308", "path": "sequences/S1308", "startFrame": 1, "endFrame": 983,"nz": 6, "ext": "jpg", "anno_path": "annotations/S1308_gt.txt"},
            {"name": "S1309", "path": "sequences/S1309", "startFrame": 1, "endFrame": 1302,"nz": 6, "ext": "jpg", "anno_path": "annotations/S1309_gt.txt"},
            {"name": "S1310", "path": "sequences/S1310", "startFrame": 1, "endFrame": 805, "nz": 6, "ext": "jpg","anno_path": "annotations/S1310_gt.txt"},
            {"name": "S1311", "path": "sequences/S1311", "startFrame": 1, "endFrame": 456, "nz": 6, "ext": "jpg","anno_path": "annotations/S1311_gt.txt"},
            {"name": "S1312", "path": "sequences/S1312", "startFrame": 1, "endFrame": 1919, "nz": 6, "ext": "jpg","anno_path": "annotations/S1312_gt.txt"},
            {"name": "S1313", "path": "sequences/S1313", "startFrame": 1, "endFrame": 2045, "nz": 6, "ext": "jpg","anno_path": "annotations/S1313_gt.txt"},
            {"name": "S1401", "path": "sequences/S1401", "startFrame": 1, "endFrame": 188, "nz": 6, "ext": "jpg","anno_path": "annotations/S1401_gt.txt"},
            {"name": "S1501", "path": "sequences/S1501", "startFrame": 1, "endFrame": 254, "nz": 6, "ext": "jpg","anno_path": "annotations/S1501_gt.txt"},
            {"name": "S1601", "path": "sequences/S1601", "startFrame": 1, "endFrame": 468, "nz": 6, "ext": "jpg","anno_path": "annotations/S1601_gt.txt"},
            {"name": "S1602", "path": "sequences/S1602", "startFrame": 1, "endFrame": 838, "nz": 6, "ext": "jpg","anno_path": "annotations/S1602_gt.txt"},
            {"name": "S1603", "path": "sequences/S1603", "startFrame": 1, "endFrame": 2969, "nz": 6, "ext": "jpg","anno_path": "annotations/S1603_gt.txt"},
            {"name": "S1604", "path": "sequences/S1604", "startFrame": 1, "endFrame": 624, "nz": 6, "ext": "jpg","anno_path": "annotations/S1604_gt.txt"},
            {"name": "S1605", "path": "sequences/S1605", "startFrame": 1, "endFrame": 605, "nz": 6, "ext": "jpg","anno_path": "annotations/S1605_gt.txt"},
            {"name": "S1606", "path": "sequences/S1606", "startFrame": 1, "endFrame": 655, "nz": 6, "ext": "jpg","anno_path": "annotations/S1606_gt.txt"},
            {"name": "S1607", "path": "sequences/S1607", "startFrame": 1, "endFrame": 563, "nz": 6, "ext": "jpg","anno_path": "annotations/S1607_gt.txt"},
            {"name": "S1701", "path": "sequences/S1701", "startFrame": 1, "endFrame": 324, "nz": 6, "ext": "jpg","anno_path": "annotations/S1701_gt.txt"},
            {"name": "S1702", "path": "sequences/S1702", "startFrame": 1, "endFrame": 329, "nz": 6, "ext": "jpg","anno_path": "annotations/S1702_gt.txt"}]

        return sequence_info_list
