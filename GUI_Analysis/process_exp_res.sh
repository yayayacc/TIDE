# Whether to overwrite
REWRITE=True
python -m analysis_third_part --all_exp_dir ./cache_embeddings/androidworld --rewrite $REWRITE
python -m analysis_third_part --all_exp_dir ./cache_embeddings/osworld --rewrite $REWRITE
python -m analysis_third_part --all_exp_dir ./cache_embeddings/waa --rewrite $REWRITE