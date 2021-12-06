class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = '/mnt/test/'    # Base directory for saving network checkpoints.
        self.tensorboard_dir = self.workspace_dir + '/tensorboard/'      # Directory for tensorboard files.
        self.pretrained_networks = self.workspace_dir + '/pretrained_networks/'
        self.lasot_dir = '/mnt/Datasets/LaSOTBenchmark/'
        self.got10k_dir = '/mnt/Datasets/got10k/train/'
        self.trackingnet_dir = '/mnt/Datasets/TrackingNet/'
        self.coco_dir = '/mnt/Datasets/coco2017/coco/'
        self.lvis_dir = ''
        self.sbd_dir = ''
        self.imagenet_dir = ''
        self.imagenetdet_dir = ''
        self.ecssd_dir = ''
        self.hkuis_dir = ''
        self.msra10k_dir = ''
        self.davis_dir = ''
        self.youtubevos_dir = ''
        self.pretrained_tracker_dir = '/mnt/MainCodes/BMVC_Submission/BMVC_Fair/pytracking/networks/prdimp50_main.pth.tar' #Pretrained Checkpoint of prdimp for search phase
        self.alpha_search_dir = '/mnt/MainCodes/BMVC_Submission/final_github/alpha_test/'