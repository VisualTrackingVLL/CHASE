import os
import sys
import argparse

env_path = os.path.join(os.path.dirname(__file__), '..')
if env_path not in sys.path:
    sys.path.append(env_path)

from pytracking.evaluation import Tracker

tracker = Tracker('dimp', 'prdimp50_original', run_id=None)
print('Tracker Made successfully')
tracker.run_vot2020(debug=0, visdom_info=None)




