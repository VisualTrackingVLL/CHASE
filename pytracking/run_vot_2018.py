import os
import sys
import argparse

env_path = os.path.join(os.path.dirname(__file__), '..')
if env_path not in sys.path:
    sys.path.append(env_path)

from pytracking.evaluation import Tracker

print('salam')
tracker = Tracker('dimp', 'prdimp_vot18', run_id=0)
print('Tracker created')
tracker.run_vot()

