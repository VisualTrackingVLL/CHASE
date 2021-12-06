from pytracking.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.
    settings.vis_path='/mnt/Datasets/VisDrone2019-SOT-test-dev/'
    settings.uavdt_path= '/mnt/Datasets/UAVDT/'
    settings.dtb70_path = '/mnt/Datasets/DTB70/'
    settings.davis_dir = ''
    settings.got10k_path = '/mnt/Datasets/got10k/'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.lasot_path = '/mnt/Datasets/LaSOTBenchmark/'
    settings.network_path = '/mnt/MainCodes/BMVC_Submission/pytracking_new_NAS/pytracking/networks/'    # Where tracking networks are stored.
    settings.nfs_path = '/mnt/Datasets/NFS/'
    settings.otb_path = ''
    settings.result_plot_path = ''
    settings.results_path = '/mnt/MainCodes/BMVC_Submission/final_github/test2/'# Where to store tracking results
    settings.segmentation_path = ''
    settings.tn_packed_results_path = ''
    settings.tpl_path = ''
    settings.uav_path = '/mnt/Datasets/UAV123/'
    settings.vot_path = ''
    settings.youtubevos_dir = ''
    settings.trackingnet_path='/mnt/Datasets/TrackingNet'

    return settings

