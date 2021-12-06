% Set path to the python in the pytracking conda environment
python_path = '/home/khaghani/anaconda3/envs/pytracking/bin/python';

% Set path to pytracking
pytracking_path = '/home/khaghani/Desktop/pytracking_new_NAS_New/pytracking';

% Set path to trax installation. Check
% https://trax.readthedocs.io/en/latest/tutorial_compiling.html for
% compilation information
trax_path = '/home/khaghani/Desktop/VOT_Toolkit/native/trax';

tracker_name = 'dimp';          % Name of the tracker to evaluate
runfile_name = 'dimp50_vot18';    % Name of the parameter file to use


%%
tracker_label = [tracker_name, '_', runfile_name];

% Generate python command
tracker_command = sprintf(['%s -c "import sys; sys.path.append(''%s'');', ...
                           'sys.path.append(''%s/support/python'');', ...
                           'import run_vot;', ...
                           'run_vot.run_vot(''%s'', ''%s'')"'],...
                           python_path, pytracking_path, trax_path, ...
                           tracker_name, runfile_name);


tracker_interpreter = python_path;

tracker_linkpath = {[trax_path, '/build'],...
		[trax_path, '/build/support/client'],...
		[trax_path, '/build/support/opencv']};
