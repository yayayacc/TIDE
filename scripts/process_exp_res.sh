# Whether to overwrite
REWRITE=False
python -m utils.analysis_files.analysis --all_exp_dir ./res/alfworld --rewrite $REWRITE
python -m utils.analysis_files.analysis --all_exp_dir ./res/frozen_lake --rewrite $REWRITE
python -m utils.analysis_files.analysis --all_exp_dir ./res/sodoku --rewrite $REWRITE
python -m utils.analysis_files.analysis --all_exp_dir ./res/blocksworld --rewrite $REWRITE
python -m utils.analysis_files.analysis --all_exp_dir ./res/webshop --rewrite $REWRITE