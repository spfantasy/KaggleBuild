#!/bin/bash
# nohup python -u script_preprocessing.py > output_prep
nohup python -u script_xgboost1.py > output_xgboost1
nohup python -u script_xgboost2.py > output_xgboost2
nohup python -u script_lgbm1.py > output_lgbm1
nohup python -u script_lgbm2.py > output_lgbm2
nohup python -u script_stacking.py > output_stacking
shutdown now