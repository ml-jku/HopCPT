## Dependencies
Use Python 3.9
```bash
conda env update -n <your-enviroment> --file ./conformal-conda-env.yaml
pip install -r ./conformal-pip-requirements.txt
pip install neuralhydrology
```

## Notes on Wandb:
The project has support for Wandb for extended logging but it is disabled by default.
If you want to use it make sure you are logged in wandb and specify your project and entity by adding
```
'config.experiment_data.project_name=<yourproject>' 'config.experiment_data.project_entity=<yourentity>' 'config.experiment_data.offline=false'
```
to the command.


## Run Evaluation Commands
Main runs (Table 1) and additional results (Appendix A3)

You need to insert for `<absolute_project_root_dir>` to the absolute root path of this repository on your machine (e.g. `/home/yourhome/hopcpt/`)

### Solar 3Y

#### Forest
```sh
##### HopCPT
python code/run_sweep_eval.py 'config/model_fc=darts_forest' 'config/model_uc=eps_pred_hopfield_3year' 'config/task=default_3year_gn' 'config/dataset=nsdb2018-20_60m' 'config.trainer.trainer_config.n_epochs=3000' 'config.experiment_data.experiment_name=forest_hf_solar3year' 'config.model_uc.eps_mem_size=5000' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'

##### EnbPI
python ./code/main.py 'config/model_fc=darts_forest' 'config/model_uc=enbpi' 'config/task=default_3year_gn' 'config/dataset=nsdb2018-20_60m' 'config.experiment_data.experiment_name=enbpi_forest_3yearSolar' 'config.model_uc.past_window_len=125' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'

##### SPCI
python ./code/run_sweep_spic.py 'config/model_fc=darts_forest' 'config/task=default_3year_gn' 'config/dataset=nsdb2018-20_60m' 'config.experiment_data.experiment_name=spic_forest_3yearSolar' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'

#### NexCP
python ./code/main.py 'config/model_fc=darts_forest' 'config/model_uc=nextcp' 'config/task=default_3year_gn' 'config/dataset=nsdb2018-20_60m' 'config.experiment_data.experiment_name=nextcp_forest_3yearSolar' 'config.model_uc.rho=0.99' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'

#### Standard CP
python ./code/main.py 'config/model_fc=darts_forest' 'config/model_uc=conf_default' 'config/task=default_3year_gn' 'config/dataset=nsdb2018-20_60m' 'config.experiment_data.experiment_name=confDefault_forest_3yearSolar' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'

#### CopulaCPTS
python ./code/main.py 'config/model_fc=darts_forest' 'config/model_uc=conf_default_plus_recent' 'config/task=default_3year_gn' 'config/dataset=nsdb2018-20_60m' 'config.experiment_data.experiment_name=copula_forest_3yearSolar' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'

```


#### LGBM
```sh
##### HopCPT
python code/run_sweep_eval.py 'config/model_fc=darts_lightgbm' 'config/model_uc=eps_pred_hopfield_3year' 'config/task=default_3year_gn' 'config/dataset=nsdb2018-20_60m' 'config.trainer.trainer_config.n_epochs=3000' 'config.experiment_data.experiment_name=lgbm_hf_solar3year' 'config.model_uc.eps_mem_size=5000' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'

##### EnbPI
python ./code/main.py 'config/model_fc=darts_lightgbm' 'config/model_uc=enbpi' 'config/task=default_3year_gn' 'config/dataset=nsdb2018-20_60m' 'config.experiment_data.experiment_name=enbpi_lgbm_3yearSolar' 'config.model_uc.past_window_len=200' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'

##### SPCI
python ./code/run_sweep_spic.py 'config/model_fc=darts_lightgbm' 'config/task=default_3year_gn' 'config/dataset=nsdb2018-20_60m' 'config.experiment_data.experiment_name=spic_lgbm_3yearSolar' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'

#### NexCP
python ./code/main.py 'config/model_fc=darts_lightgbm' 'config/model_uc=nextcp' 'config/task=default_3year_gn' 'config/dataset=nsdb2018-20_60m' 'config.experiment_data.experiment_name=nextcp_lgbm_3yearSolar' 'config.model_uc.rho=0.99' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'
``
#### Standard CP
python ./code/main.py 'config/model_fc=darts_lightgbm' 'config/model_uc=conf_default' 'config/task=default_3year_gn' 'config/dataset=nsdb2018-20_60m' 'config.experiment_data.experiment_name=confDefault_lgbm_3yearSolar' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'

#### CopulaCPTS
python ./code/main.py 'config/model_fc=darts_lightgbm' 'config/model_uc=conf_default_plus_recent' 'config/task=default_3year_gn' 'config/dataset=nsdb2018-20_60m' 'config.experiment_data.experiment_name=copula_lgbm_3yearSolar' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'

#### AdaptiveCI
python ./code/main.py 'config/model_fc=darts_lightgbm_quantile' 'config/model_uc=adaptiveci' 'config/task=default_3year_gn' 'config/dataset=nsdb2018-20_60m' 'config.experiment_data.experiment_name=adaptiveCI_dartsGB_3yearSolar' 'config.model_uc.mode=simple' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'
```


#### Ridge
```sh
##### HopCPT
python code/run_sweep_eval.py 'config/model_fc=reg_ridge' 'config/model_uc=eps_pred_hopfield_3year' 'config/task=default_3year_gn' 'config/dataset=nsdb2018-20_60m' 'config.trainer.trainer_config.n_epochs=3000' 'config.experiment_data.experiment_name=ridge_hf_solar3year' 'config.model_uc.eps_mem_size=5000' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'

##### EnbPI
python ./code/main.py 'config/model_fc=reg_ridge' 'config/model_uc=enbpi' 'config/task=default_3year_gn' 'config/dataset=nsdb2018-20_60m' 'config.experiment_data.experiment_name=enbpi_ridge_3yearSolar' 'config.model_uc.past_window_len=25' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'

##### SPCI
python ./code/run_sweep_spic.py 'config/model_fc=reg_ridge' 'config/task=default_3year_gn' 'config/dataset=nsdb2018-20_60m' 'config.experiment_data.experiment_name=spic_regridge_3yearSolar' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'

#### NexCP
python ./code/main.py 'config/model_fc=reg_ridge' 'config/model_uc=nextcp' 'config/task=default_3year_gn' 'config/dataset=nsdb2018-20_60m' 'config.experiment_data.experiment_name=nextcp_ridge_3yearSolar' 'config.model_uc.rho=0.999' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'

#### Standard CP
python ./code/main.py 'config/model_fc=reg_ridge' 'config/model_uc=conf_default' 'config/task=default_3year_gn' 'config/dataset=nsdb2018-20_60m' 'config.experiment_data.experiment_name=confDefault_ridge_3yearSolar' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>' 

#### CopulaCPTS
python ./code/main.py 'config/model_fc=reg_ridge' 'config/model_uc=conf_default_plus_recent' 'config/task=default_3year_gn' 'config/dataset=nsdb2018-20_60m' 'config.experiment_data.experiment_name=copula_ridge_3yearSolar' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>' 

```

### LSTM

For the LSTM model you need to train an LSTM first - and save its predictions.
The prediction output and the  last saved (=best) model need to bout inside the a folder `models_save/lstm_fc` in the `absolute_project_root_dir`. The prediciction file must be named `predicitons.pt` the model file `model`.

```sh
python ./code/run_sweep_lstm_train.py 'config/task=default_3year_gn' 'config/dataset=nsdb2018-20_60m' 'config.experiment_data.experiment_name=trainsweep_solar3Y''config.model_fc.model_params.plot_eval_after_train=false' 'config.trainer.trainer_config.n_epochs=150' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>' 'model_fc.model_params.lstm_conf.hidden_size=256' 'model_fc.model_params.dropout=0.25' 'model_fc.model_params.batch_size=512' 'trainer.trainer_config.optim.lr=0.0001'
```

```sh
##### HopCPT
python code/run_sweep_eval.py 'config/model_fc=global_lstm_solar3y' 'config/model_uc=eps_pred_hopfield_3year' 'config/task=default_3year_gn' 'config/dataset=nsdb2018-20_60m' 'config.trainer.trainer_config.n_epochs=3000' 'config.experiment_data.experiment_name=lstm_hf_solar3y_Eval' 'config.model_uc.eps_mem_size=5000' 'config.trainer.trainer_config.optim.lr=0.01' 'config.model_uc.pos_encode.mode=rel-simple' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'

#### EnbPI
python ./code/main.py 'config/model_fc=global_lstm_solar3y' 'config/model_uc=enbpi' 'config/task=default_3year_gn' 'config/dataset=nsdb2018-20_60m' 'config.experiment_data.experiment_name=enbpi_lstm_solar3y_EvalRun' 'config.model_uc.past_window_len=200' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>' 

#### SPCI
python ./code/run_sweep_spic.py 'config/model_fc=global_lstm_solar3y' 'config/task=default_3year_gn' 'config/dataset=nsdb2018-20_60m' 'config.experiment_data.experiment_name=spic_lstm_solar3y' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'

### NexCP
python ./code/main.py 'config/model_fc=global_lstm_solar3y' 'config/model_uc=nextcp' 'config/task=default_3year_gn' 'config/dataset=nsdb2018-20_60m' 'config.experiment_data.experiment_name=ncp_lstm_solar3y_EvalRun' 'config.model_uc.rho=0.995' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>' 

#### StandardCP/CF-RNN
python ./code/main.py 'config/model_fc=global_lstm_solar3y' 'config/model_uc=conf_default' 'config/task=default_3year_gn' 'config/dataset=nsdb2018-20_60m' 'config.experiment_data.experiment_name=confDef_lstm_solar3y_EvalRun' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'

#### CopulaCPTS
python ./code/main.py 'config/model_fc=global_lstm_solar3y' 'config/model_uc=conf_default_plus_recent' 'config/task=default_3year_gn' 'config/dataset=nsdb2018-20_60m' 'config.experiment_data.experiment_name=confDef_lstm_solar3y_EvalRun' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'
```

### Air 10

#### Forest
```sh
##### HopCPT
python ./code/run_sweep_eval.py 'config/model_fc=darts_forest' 'config/model_uc=eps_pred_hopfield_3year' 'config/task=default_3year_gn' 'config/dataset=bejing_air_pm10' 'config.experiment_data.experiment_name=forest_hf_air10full' 'config.trainer.trainer_config.n_epochs=3000' 'config.model_uc.batch_size=2' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'

##### EnbPI
python ./code/main.py 'config/model_fc=darts_forest' 'config/model_uc=enbpi' 'config/task=default_3year_gn' 'config/dataset=bejing_air_pm10' 'config.experiment_data.experiment_name=enbpi_forest_air10full' 'config.model_uc.past_window_len=200' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'

##### SPCI
python ./code/run_sweep_spic.py 'config/model_fc=darts_forest' 'config/task=default_3year_gn' 'config/dataset=bejing_air_pm10' 'config.experiment_data.experiment_name=spic_forest_air10full' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'

#### NexCP
python ./code/main.py 'config/model_fc=darts_forest' 'config/model_uc=nextcp' 'config/task=default_3year_gn' 'config/dataset=bejing_air_pm10' 'config.experiment_data.experiment_name=nextcp_forest_air10full' 'config.model_uc.rho=0.993' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'

#### Standard CP
python ./code/main.py 'config/model_fc=darts_forest' 'config/model_uc=conf_default' 'config/task=default_3year_gn' 'config/dataset=bejing_air_pm10' 'config.experiment_data.experiment_name=confDefault_forest_air10full' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'

#### CopulaCPTS
python ./code/main.py 'config/model_fc=darts_forest' 'config/model_uc=conf_default_plus_recent' 'config/task=default_3year_gn' 'config/dataset=bejing_air_pm10' 'config.experiment_data.experiment_name=copula_forest_air10full' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'
```

#### LGBM
```sh
##### HopCPT
python ./code/run_sweep_eval.py 'config/model_fc=darts_lightgbm' 'config/model_uc=eps_pred_hopfield_3year' 'config/task=default_3year_gn' 'config/dataset=bejing_air_pm10' 'config.experiment_data.experiment_name=lgbm_hf_air10full' 'config.trainer.trainer_config.n_epochs=3000' 'config.model_uc.batch_size=2' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'

##### EnbPI
python ./code/main.py 'config/model_fc=darts_lightgbm' 'config/model_uc=enbpi' 'config/task=default_3year_gn' 'config/dataset=bejing_air_pm10' 'config.experiment_data.experiment_name=enbpi_lgbm_air10full' 'config.model_uc.past_window_len=200' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'

##### SPCI
python ./code/run_sweep_spic.py 'config/model_fc=darts_lightgbm' 'config/task=default_3year_gn' 'config/dataset=bejing_air_pm10' 'config.experiment_data.experiment_name=spic_lgbm_air10full' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'

#### NexCP
python ./code/main.py 'config/model_fc=darts_lightgbm' 'config/model_uc=nextcp' 'config/task=default_3year_gn' 'config/dataset=bejing_air_pm10' 'config.experiment_data.experiment_name=nextcp_lgbm_air10full' 'config.model_uc.rho=0.993' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'

#### Standard CP
python ./code/main.py 'config/model_fc=darts_lightgbm' 'config/model_uc=conf_default' 'config/task=default_3year_gn' 'config/dataset=bejing_air_pm10' 'config.experiment_data.experiment_name=confDefault_lgbm_air10full' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'

#### AdaptiveCI
python ./code/main.py 'config/model_fc=darts_lightgbm_quantile' 'config/model_uc=adaptiveci' 'config/task=default_3year_gn' 'config/dataset=bejing_air_pm10' 'config.experiment_data.experiment_name=adaptiveCI_dartsGB_air10full' 'config.model_uc.mode=simple' 'config.model_uc.gamma=0.01' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'
```


#### Ridge
```sh
##### HopCPT
python ./code/run_sweep_eval.py 'config/model_fc=reg_ridge' 'config/model_uc=eps_pred_hopfield_3year' 'config/task=default_3year_gn' 'config/dataset=bejing_air_pm10' 'config.experiment_data.experiment_name=ridge_hf_air10full' 'config.trainer.trainer_config.n_epochs=3000' 'config.model_uc.batch_size=2' 'config.model_uc.ctx_encode_dropout=0.25' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'

##### EnbPI
python ./code/main.py 'config/model_fc=reg_ridge' 'config/model_uc=enbpi' 'config/task=default_3year_gn' 'config/dataset=bejing_air_pm10' 'config.experiment_data.experiment_name=enbpi_ridge_air10full' 'config.model_uc.past_window_len=200' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'

##### SPCI
python ./code/run_sweep_spic.py 'config/model_fc=reg_ridge' 'config/task=default_3year_gn' 'config/dataset=bejing_air_pm10' 'config.experiment_data.experiment_name=spic_regridge_air10full' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'

#### NexCP
python ./code/main.py 'config/model_fc=reg_ridge' 'config/model_uc=nextcp' 'config/task=default_3year_gn' 'config/dataset=bejing_air_pm10' 'config.experiment_data.experiment_name=nextcp_ridge_air10full' 'config.model_uc.rho=0.993' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'

#### Standard CP
python ./code/main.py 'config/model_fc=reg_ridge' 'config/model_uc=conf_default' 'config/task=default_3year_gn' 'config/dataset=bejing_air_pm10' 'config.experiment_data.experiment_name=confDefault_ridge_air10full' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'
```

### LSTM

Train LSTM Model (details see above)
```sh
python ./code/run_sweep_lstm_train.py 'config/task=default_3year_gn' 'config/dataset=bejing_air_pm10' 'config.experiment_data.experiment_name=trainsweep_air10' 'config.model_fc.model_params.plot_eval_after_train=false' 'config.trainer.trainer_config.n_epochs=150' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>' 'model_fc.model_params.lstm_conf.hidden_size=256' 'model_fc.model_params.dropout=0.1' 'model_fc.model_params.batch_size=512' 'trainer.trainer_config.optim.lr=0.0001'
```

```sh
##### HopCPT
python code/run_sweep_eval.py 'config/model_fc=global_lstm_air10' 'config/model_uc=eps_pred_hopfield_3year' 'config/task=default_3year_gn' 'config/dataset=bejing_air_pm10' 'config.trainer.trainer_config.n_epochs=3000' 'config.experiment_data.experiment_name=lstm_hf_air10full_Eval' 'config.model_uc.batch_size=2' 'config.model_uc.ctx_encode_dropout=null' 'config.trainer.trainer_config.optim.lr=0.001' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'

#### SPCI
python ./code/run_sweep_spic.py 'config/model_fc=global_lstm_air10' 'config/task=default_3year_gn' 'config/dataset=bejing_air_pm10' 'config.experiment_data.experiment_name=spic_lstm_air10' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'

#### EnbPI
python ./code/main.py 'config/model_fc=global_lstm_air10' 'config/model_uc=enbpi' 'config/task=default_3year_gn' 'config/dataset=bejing_air_pm10' 'config.experiment_data.experiment_name=enbpi_lstm_air10_EvalRun' 'config.model_uc.past_window_len=200' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'

#### NexCP
 python ./code/main.py 'config/model_fc=global_lstm_air10' 'config/model_uc=nextcp' 'config/task=default_3year_gn' 'config/dataset=bejing_air_pm10' 'config.experiment_data.experiment_name=ncp_lstm_air10_EvalRun' 'config.model_uc.rho=0.995' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'
 
#### Standard CP/CF-RNN
python ./code/main.py 'config/model_fc=global_lstm_air10' 'config/model_uc=conf_default' 'config/task=default_3year_gn' 'config/dataset=bejing_air_pm10' 'config.experiment_data.experiment_name=confDef_lstm_air10_EvalRun' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'

#### CopulaCPTS
 python ./code/main.py 'config/model_fc=global_lstm_air10' 'config/model_uc=conf_default_plus_recent' 'config/task=default_3year_gn' 'config/dataset=bejing_air_pm10' 'config.experiment_data.experiment_name=confDef_lstm_air10_EvalRun' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'
```

### Sapflux

#### Forest
```sh
##### HopCPT
python ./code/run_sweep_eval.py 'config/model_fc=darts_forest' 'config/model_uc=eps_pred_hopfield_3year' 'config/task=default_3year_gn' 'config/dataset=sapflux_solo3_large' 'config.trainer.trainer_config.n_epochs=3000' 'config.experiment_data.experiment_name=forest_hf_sapflux3' 'config.model_uc.eps_mem_size=4000' 'config.model_uc.pos_encode.mode=null' 'config.model_uc.batch_size=1' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'

##### EnbPI
python code/main.py 'config/model_fc=darts_forest' 'config/model_uc=enbpi' 'config/task=default_3year_gn' 'config/dataset=sapflux_solo3_large' 'config.experiment_data.experiment_name=forest_enbpi_sapflux3' 'config.model_uc.past_window_len=200' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'

##### SPCI
python code/run_sweep_spic.py 'config/model_fc=darts_forest' 'config/task=default_3year_gn' 'config/dataset=sapflux_solo3_large' 'config.experiment_data.experiment_name=forest_spic_sapflux3' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'

#### NexCP
python code/main.py 'config/model_fc=darts_forest' 'config/model_uc=nextcp' 'config/task=default_3year_gn' 'config/dataset=sapflux_solo3_large' 'config.experiment_data.experiment_name=forest_nextcp_sapflux3' 'config.model_uc.rho=0.995' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'

#### Standard CP
python code/main.py 'config/model_fc=darts_forest' 'config/model_uc=conf_default' 'config/task=default_3year_gn' 'config/dataset=sapflux_solo3_large' 'config.experiment_data.experiment_name=forest_confDefault_sapflux3' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'
```

#### LGBM
```sh
##### HopCPT
python ./code/run_sweep_eval.py 'config/model_fc=darts_lightgbm' 'config/model_uc=eps_pred_hopfield_3year' 'config/task=default_3year_gn' 'config/dataset=sapflux_solo3_large' 'config.trainer.trainer_config.n_epochs=3000' 'config.experiment_data.experiment_name=lgbm_hf_sapflux3' 'config.model_uc.eps_mem_size=4000' 'config.model_uc.pos_encode.mode=null' 'config.model_uc.batch_size=1' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'

##### EnbPI
python code/main.py 'config/model_fc=darts_lightgbm' 'config/model_uc=enbpi' 'config/task=default_3year_gn' 'config/dataset=sapflux_solo3_large' 'config.experiment_data.experiment_name=lgbm_enbpi_sapflux3' 'config.model_uc.past_window_len=200' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'

##### SPCI
python code/run_sweep_spic.py 'config/model_fc=darts_lightgbm' 'config/task=default_3year_gn' 'config/dataset=sapflux_solo3_large' 'config.experiment_data.experiment_name=lgbm_spic_sapflux3' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'

#### NexCP
python code/main.py 'config/model_fc=darts_lightgbm' 'config/model_uc=nextcp' 'config/task=default_3year_gn' 'config/dataset=sapflux_solo3_large' 'config.experiment_data.experiment_name=lgbm_nextcp_sapflux3' 'config.model_uc.rho=0.995' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'

#### Standard CP
python code/main.py 'config/model_fc=darts_lightgbm' 'config/model_uc=conf_default' 'config/task=default_3year_gn' 'config/dataset=sapflux_solo3_large' 'config.experiment_data.experiment_name=lgbm_confDefault_sapflux3' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'

#### AdaptiveCI
python ./code/main.py 'config/model_fc=darts_lightgbm_quantile' 'config/model_uc=adaptiveci' 'config/task=default_3year_gn' 'config/dataset=sapflux_solo3_large' 'config.experiment_data.experiment_name=adaptiveCI_dartsGB_sapflux3' 'config.model_uc.mode=simple' 'config.model_uc.gamma=0.002' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'
```


#### Ridge
```sh
##### HopCPT
python ./code/run_sweep_eval.py 'config/model_fc=reg_ridge' 'config/model_uc=eps_pred_hopfield_3year' 'config/task=default_3year_gn' 'config/dataset=sapflux_solo3_large' 'config.trainer.trainer_config.n_epochs=3000' 'config.experiment_data.experiment_name=ridgeReg_hf_sapflux3' 'config.model_uc.eps_mem_size=4000' 'config.model_uc.pos_encode.mode=null' 'config.model_uc.batch_size=1' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'

##### EnbPI
python code/main.py 'config/model_fc=reg_ridge' 'config/model_uc=enbpi' 'config/task=default_3year_gn' 'config/dataset=sapflux_solo3_large' 'config.experiment_data.experiment_name=ridge_enbpi_sapflux3' 'config.model_uc.past_window_len=150' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'

##### SPCI
python code/run_sweep_spic.py 'config/model_fc=reg_ridge' 'config/task=default_3year_gn' 'config/dataset=sapflux_solo3_large' 'config.experiment_data.experiment_name=regRidge_spic_sapflux3' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'

#### NexCP
python code/main.py 'config/model_fc=reg_ridge' 'config/model_uc=nextcp' 'config/task=default_3year_gn' 'config/dataset=sapflux_solo3_large' 'config.experiment_data.experiment_name=ridge_nextcp_sapflux3' 'config.model_uc.rho=0.995' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'

#### Standard CP
python code/main.py 'config/model_fc=reg_ridge' 'config/model_uc=conf_default' 'config/task=default_3year_gn' 'config/dataset=sapflux_solo3_large' 'config.experiment_data.experiment_name=ridge_confDefault_sapflux3' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'
```

### LSTM

Train LSTM Model (details see above)
```sh
python ./code/run_sweep_lstm_train.py 'config/task=default_3year_gn' 'config/dataset=sapflux_solo3_large' 'config.experiment_data.experiment_name=trainsweep_sapflux''config.model_fc.model_params.plot_eval_after_train=false' 'config.trainer.trainer_config.n_epochs=150' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>' 'model_fc.model_params.lstm_conf.hidden_size=256' 'model_fc.model_params.dropout=0' 'model_fc.model_params.batch_size=256' 'trainer.trainer_config.optim.lr=0.001'
```

```sh
#### HopCPT
python code/run_sweep_eval.py 'config/model_fc=global_lstm_sapflux' 'config/model_uc=eps_pred_hopfield_3year' 'config/task=default_3year_gn' 'config/dataset=sapflux_solo3_large' 'config.trainer.trainer_config.n_epochs=3000' 'config.experiment_data.experiment_name=lstm_hf_sapflux_Eval' 'config.model_uc.eps_mem_size=4000' 'config.model_uc.batch_size=1' 'config.model_uc.ctx_encode_dropout=0.25' 'config.trainer.trainer_config.optim.lr=0.01' 'config.model_uc.pos_encode.mode=rel-simple' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'

#### EnbPI
python ./code/main.py 'config/model_fc=global_lstm_sapflux' 'config/model_uc=enbpi' 'config/task=default_3year_gn' 'config/dataset=sapflux_solo3_large' 'config.experiment_data.experiment_name=enbpi_lstm_sapflux_EvalRun' 'config.model_uc.past_window_len=200' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'

### SPCI
python code/run_sweep_spic.py 'config/model_fc=global_lstm_sapflux' 'config/task=default_3year_gn' 'config/dataset=sapflux_solo3_large' 'config.experiment_data.experiment_name=spic_lstm_sapflux' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'

#### NexCP
ython ./code/main.py 'config/model_fc=global_lstm_sapflux' 'config/model_uc=nextcp' 'config/task=default_3year_gn' 'config/dataset=sapflux_solo3_large' 'config.experiment_data.experiment_name=ncp_lstm_sapflux_EvalRun' 'config.model_uc.rho=0.993' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'

#### Standard CP/CF-RNN
python ./code/main.py 'config/model_fc=global_lstm_sapflux' 'config/model_uc=conf_default' 'config/task=default_3year_gn' 'config/dataset=sapflux_solo3_large' 'config.experiment_data.experiment_name=confDef_lstm_sapflux_EvalRun' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'

#### CopulaCPTS
python ./code/main.py 'config/model_fc=global_lstm_sapflux' 'config/model_uc=conf_default_plus_recent' 'config/task=default_3year_gn' 'config/dataset=sapflux_solo3_large' 'config.experiment_data.experiment_name=confDef_lstm_sapflux_EvalRun' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'
````

### Streamflow

For the streamflow runs additional data is needed:
1) Download camel_us as described here https://neuralhydrology.readthedocs.io/en/latest/tutorials/data-prerequisites.html:
The relevant subsection are `CAMELS US catchment attributes` and `CAMELS US meteorological time series and streamflow data` 
Then you copy the extracted files in your local repo to `<repo-root>/data/hydrology/CAMELS_US`.
2) Download https://www.hydroshare.org/resource/17c896843cf940339c3c3496d0c1c077/ and place zip content into `<repo-root>/data/hydrology/CAMELS_US/basin_mean_forcing/maurer_extended`.
3) Download https://www.hydroshare.org/resource/0a68bfd7ddf642a8be9041d60f40868c/ and place tar content into `<repo-root>/data/hydrology/CAMELS_US/basin_mean_forcing/nldas_extended`.

```sh
##### HopCPT
python code/run_sweep_eval.py 'config/model_fc=precomputed_nh' 'config/model_uc=eps_pred_hopfield_3year' 'config/task=default_3year_gn' 'config/dataset=hydro' 'config.trainer.trainer_config.n_epochs=3000' 'config.experiment_data.experiment_name=forest_hf_streamflow' 'config.model_uc.eps_mem_size=4000' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'

##### EnbPI
python ./code/main.py 'config/model_fc=precomputed_nh' 'config/model_uc=enbpi' 'config/task=default_3year_gn' 'config/dataset=hydro' 'config.experiment_data.experiment_name=enbpi_forest_streamflow' 'config.model_uc.past_window_len=200' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'

##### SPCI
python ./code/run_sweep_spic.py 'config/model_fc=precomputed_nh' 'config/task=default_3year_gn' 'config/dataset=hydro' 'config.experiment_data.experiment_name=spic_forest_streamflow' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'

#### NexCP
python ./code/main.py 'config/model_fc=precomputed_nh' 'config/model_uc=nextcp' 'config/task=default_3year_gn' 'config/dataset=hydro' 'config.experiment_data.experiment_name=nextcp_forest_streamflow' 'config.model_uc.rho=0.993' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'

#### Standard CP
python ./code/main.py 'config/model_fc=precomputed_nh' 'config/model_uc=conf_default' 'config/task=default_3year_gn' 'config/dataset=hydro' 'config.experiment_data.experiment_name=confDefault_forest_streamflow' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'
```


### Solar 1Y

#### Forest
```sh
##### HopCPT
python ./code/run_sweep_eval.py 'config/model_fc=darts_forest' 'config/model_uc=eps_pred_hopfield_1year' 'config/task=default_1year_gn' 'config/dataset=nsdb2019_60m'  'config.trainer.trainer_config.n_epochs=3000' 'config.experiment_data.experiment_name=forest_hf_soalr1year' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'

##### EnbPI
python ./code/main.py 'config/model_fc=darts_forest' 'config/model_uc=enbpi' 'config/task=default_1year_gn' 'config/dataset=nsdb2019_60m' 'config.experiment_data.experiment_name=enbpi_forest_1yearSolar' 'config.model_uc.past_window_len=200' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'

##### SPCI
python ./code/run_sweep_spic.py 'config/model_fc=darts_forest' 'config/task=default_1year_gn' 'config/dataset=nsdb2019_60m' 'config.experiment_data.experiment_name=spic_forest_1yearSolar' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'

#### NexCP
python ./code/main.py 'config/model_fc=darts_forest' 'config/model_uc=nextcp' 'config/task=default_1year_gn' 'config/dataset=nsdb2019_60m' 'config.experiment_data.experiment_name=nextcp_forest_1yearSolar' 'config.model_uc.rho=0.99' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'

#### Standard CP
python ./code/main.py 'config/model_fc=darts_forest' 'config/model_uc=conf_default' 'config/task=default_1year_gn' 'config/dataset=nsdb2019_60m' 'config.experiment_data.experiment_name=confDefault_forest_1yearSolar' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'
```

#### LGBM
```sh
##### HopCPT
python ./code/run_sweep_eval.py 'config/model_fc=darts_lightgbm' 'config/model_uc=eps_pred_hopfield_1year' 'config/task=default_1year_gn' 'config/dataset=nsdb2019_60m'  'config.trainer.trainer_config.n_epochs=3000' 'config.experiment_data.experiment_name=lgbm_hf_solar1year' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'

##### EnbPI
python ./code/main.py 'config/model_fc=darts_lightgbm' 'config/model_uc=enbpi' 'config/task=default_1year_gn' 'config/dataset=nsdb2019_60m' 'config.experiment_data.experiment_name=enbpi_lgbm_1yearSolar' 'config.model_uc.past_window_len=200' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'

##### SPCI
python ./code/run_sweep_spic.py 'config/model_fc=darts_lightgbm' 'config/task=default_1year_gn' 'config/dataset=nsdb2019_60m' 'config.experiment_data.experiment_name=spic_lgbm_1yearSolar' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'

#### NexCP
python ./code/main.py 'config/model_fc=darts_lightgbm' 'config/model_uc=nextcp' 'config/task=default_1year_gn' 'config/dataset=nsdb2019_60m' 'config.experiment_data.experiment_name=nextcp_lgmb_1yearSolar_EvalRun' 'config.model_uc.rho=0.99' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'

#### Standard CP
python ./code/main.py 'config/model_fc=darts_lightgbm' 'config/model_uc=conf_default' 'config/task=default_1year_gn' 'config/dataset=nsdb2019_60m' 'config.experiment_data.experiment_name=confDefault_lgbm_1yearSolar' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'

#### AdaptiveCI
python ./code/main.py 'config/model_fc=darts_lightgbm_quantile' 'config/model_uc=adaptiveci' 'config/task=default_1year_gn' 'config/dataset=nsdb2019_60m' 'config.experiment_data.experiment_name=adaptiveCI_dartsGB_1yearSolar' 'config.model_uc.mode=simple' 'config.model_uc.gamma=0.02' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'
```


#### Ridge
```sh
##### HopCPT
python ./code/run_sweep_eval.py 'config/model_fc=reg_ridge' 'config/model_uc=eps_pred_hopfield_1year' 'config/task=default_1year_gn' 'config/dataset=nsdb2019_60m'  'config.trainer.trainer_config.n_epochs=3000' 'config.experiment_data.experiment_name=ridge_hf_solar1year' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'

##### EnbPI
python ./code/main.py 'config/model_fc=reg_ridge' 'config/model_uc=enbpi' 'config/task=default_1year_gn' 'config/dataset=nsdb2019_60m' 'config.experiment_data.experiment_name=enbpi_ridge_1yearSolar' 'config.model_uc.past_window_len=200' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'

##### SPCI
python ./code/run_sweep_spic.py 'config/model_fc=reg_ridge' 'config/task=default_1year_gn' 'config/dataset=nsdb2019_60m' 'config.experiment_data.experiment_name=spic_regridge_1yearSolar' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'

#### NexCP
python ./code/main.py 'config/model_fc=reg_ridge' 'config/model_uc=nextcp' 'config/task=default_1year_gn' 'config/dataset=nsdb2019_60m' 'config.experiment_data.experiment_name=nextcp_ridge_1yearSolar' 'config.model_uc.rho=0.999' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'

#### Standard CP
python ./code/main.py 'config/model_fc=reg_ridge' 'config/model_uc=conf_default' 'config/task=default_1year_gn' 'config/dataset=nsdb2019_60m' 'config.experiment_data.experiment_name=confDefault_ridge_1yearSolar' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'
```

### LSTM

Train LSTM Model (details see above)
```sh
python ./code/run_sweep_lstm_train.py 'config/task=default_1year_gn' 'config/dataset=nsdb2019_60m' 'config.experiment_data.experiment_name=trainsweep_solar1Y' 'config.model_fc.model_params.plot_eval_after_train=false''config.trainer.trainer_config.n_epochs=150' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'  'model_fc.model_params.lstm_conf.hidden_size=64' 'model_fc.model_params.dropout=0.25' 'model_fc.model_params.batch_size=256' 'trainer.trainer_config.optim.lr=0.001'
```

```sh
#### HopCPT
python code/run_sweep_eval.py 'config/model_fc=global_lstm_solar1y' 'config/model_uc=eps_pred_hopfield_1year' 'config/task=default_1year_gn' 'config/dataset=nsdb2019_60m' 'config.trainer.trainer_config.n_epochs=3000' 'config.experiment_data.experiment_name=lstm_hf_solar1y_Eval' 'config.model_uc.ctx_encode_dropout=0.5' 'config.trainer.trainer_config.optim.lr=0.01' 'config.model_uc.pos_encode.mode=null' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'

#### EnbPI
python ./code/main.py 'config/model_fc=global_lstm_solar1y' 'config/model_uc=enbpi' 'config/task=default_1year_gn' 'config/dataset=nsdb2019_60m' 'config.experiment_data.experiment_name=enbpi_lstm_solar1Y_EvalRun' 'config.model_uc.past_window_len=200' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'

#### SPCI
python ./code/run_sweep_spic.py 'config/model_fc=global_lstm_solar1y' 'config/task=default_1year_gn' 'config/dataset=nsdb2019_60m' 'config.experiment_data.experiment_name=spic_lstm_solar1y' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'

#### NexCP
python ./code/main.py 'config/model_fc=global_lstm_solar1y' 'config/model_uc=nextcp' 'config/task=default_1year_gn' 'config/dataset=nsdb2019_60m' 'config.experiment_data.experiment_name=ncp_lstm_solar1Y_EvalRun' 'config.model_uc.rho=0.993' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'

#### StandardCP /CF-RNN
python ./code/main.py 'config/model_fc=global_lstm_solar1y' 'config/model_uc=conf_default' 'config/task=default_1year_gn' 'config/dataset=nsdb2019_60m' 'config.experiment_data.experiment_name=confDef_lstm_solar1Y_EvalRun' 'config.model_uc.rho=0.993' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'

#### CopulaCPTS
python ./code/main.py 'config/model_fc=global_lstm_solar1y' 'config/model_uc=conf_default_plus_recent' 'config/task=default_1year_gn' 'config/dataset=nsdb2019_60m' 'config.experiment_data.experiment_name=confDef_lstm_solar1Y_EvalRun' 'config.model_uc.rho=0.993' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'
````

### Solar Small

#### Forest
```sh
##### HopCPT
python ./code/run_sweep_eval.py 'config/model_fc=darts_forest' 'config/model_uc=eps_pred_hopfield' 'config/task=default_gn' 'config/dataset=enbPI_solar_all'  'config.trainer.trainer_config.n_epochs=3000' 'config.experiment_data.experiment_name=forest_hf_smallSolar' 'config.model_uc.ctx_encode_dropout=0.50' 'config.trainer.trainer_config.optim.lr=0.001' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'

##### EnbPI
python ./code/main.py 'config/model_fc=darts_forest' 'config/model_uc=enbpi' 'config/task=default_gn' 'config/dataset=enbPI_solar_all' 'config.experiment_data.experiment_name=enbpi_forest_smallSolar' 'config.model_uc.past_window_len=150' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'

##### SPCI
python ./code/run_sweep_spic.py 'config/model_fc=darts_forest' 'config/task=default_gn' 'config/dataset=enbPI_solar_all' 'config.experiment_data.experiment_name=spic_forest_smallSolar' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'

#### NexCP
python ./code/main.py 'config/model_fc=darts_forest' 'config/model_uc=nextcp' 'config/task=default_gn' 'config/dataset=enbPI_solar_all' 'config.experiment_data.experiment_name=nextcp_forest_smallSolar' 'config.model_uc.rho=0.98' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'

#### Standard CP
python ./code/main.py 'config/model_fc=darts_forest' 'config/model_uc=conf_default' 'config/task=default_gn' 'config/dataset=enbPI_solar_all' 'config.experiment_data.experiment_name=confDefault_forest_smallSolar' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'
```

#### LGBM
```sh
##### HopCPT
python ./code/run_sweep_eval.py 'config/model_fc=darts_lightgbm' 'config/model_uc=eps_pred_hopfield' 'config/task=default_gn' 'config/dataset=enbPI_solar_all'  'config.trainer.trainer_config.n_epochs=3000' 'config.experiment_data.experiment_name=lgbm_hf_smallSolar' 'config.model_uc.ctx_encode_dropout=0.50' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'

##### EnbPI
python ./code/main.py 'config/model_fc=darts_lightgbm' 'config/model_uc=enbpi' 'config/task=default_gn' 'config/dataset=enbPI_solar_all' 'config.experiment_data.experiment_name=enbpi_lgbm_smallSolar' 'config.model_uc.past_window_len=150' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'

##### SPCI
python ./code/run_sweep_spic.py 'config/model_fc=darts_lightgbm' 'config/task=default_gn' 'config/dataset=enbPI_solar_all' 'config.experiment_data.experiment_name=spic_lgbm_smallSolar' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'

#### NexCP
python ./code/main.py 'config/model_fc=darts_lightgbm' 'config/model_uc=nextcp' 'config/task=default_gn' 'config/dataset=enbPI_solar_all' 'config.experiment_data.experiment_name=nextcp_lgbm_smallSolar_EvalRun' 'config.model_uc.rho=0.98' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'

#### Standard CP
python ./code/main.py 'config/model_fc=darts_lightgbm' 'config/model_uc=conf_default' 'config/task=default_gn' 'config/dataset=enbPI_solar_all' 'config.experiment_data.experiment_name=confDefault_lgbm_smallSolar' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'

#### AdaptiveCI
python ./code/main.py 'config/model_fc=darts_lightgbm_quantile' 'config/model_uc=adaptiveci' 'config/task=default_gn' 'config/dataset=enbPI_solar_all' 'config.experiment_data.experiment_name=adaptiveCI_dartsGB_smSolar' 'config.model_uc.mode=simple' 'config.model_uc.gamma=0.02' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'
```


#### Ridge
```sh
##### HopCPT
python ./code/run_sweep_eval.py 'config/model_fc=reg_ridge' 'config/model_uc=eps_pred_hopfield' 'config/task=default_gn' 'config/dataset=enbPI_solar_all'  'config.trainer.trainer_config.n_epochs=3000' 'config.experiment_data.experiment_name=ridge_hf_smallSolar'  'config.model_uc.ctx_encode_dropout=0.50' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'

##### EnbPI
python ./code/main.py 'config/model_fc=reg_ridge' 'config/model_uc=enbpi' 'config/task=default_gn' 'config/dataset=enbPI_solar_all' 'config.experiment_data.experiment_name=enbpi_ridge_smallSolar' 'config.model_uc.past_window_len=100' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'

##### SPCI
python ./code/run_sweep_spic.py 'config/model_fc=reg_ridge' 'config/task=default_gn' 'config/dataset=enbPI_solar_all' 'config.experiment_data.experiment_name=spic_ridge_smallSolar' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'

#### NexCP
python ./code/main.py 'config/model_fc=reg_ridge' 'config/model_uc=nextcp' 'config/task=default_gn' 'config/dataset=enbPI_solar_all' 'config.experiment_data.experiment_name=nextcp_ridge_smallSolar_EvalRun' 'config.model_uc.rho=0.999' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'

#### Standard CP
python ./code/main.py 'config/model_fc=reg_ridge' 'config/model_uc=conf_default' 'config/task=default_gn' 'config/dataset=enbPI_solar_all' 'config.experiment_data.experiment_name=confDefault_ridge_smallSolar' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'
```

### Air 25

#### Forest
```sh
##### HopCPT
python ./code/run_sweep_eval.py 'config/model_fc=darts_forest' 'config/model_uc=eps_pred_hopfield_3year' 'config/task=default_3year_gn' 'config/dataset=bejing_air_pm25' 'config.experiment_data.experiment_name=forest_hf_air25full' 'config.trainer.trainer_config.n_epochs=3000' 'config.model_uc.batch_size=2' 'config.model_uc.pos_encode.mode=rel-simple' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'

##### EnbPI
python ./code/main.py 'config/model_fc=darts_forest' 'config/model_uc=enbpi' 'config/task=default_3year_gn' 'config/dataset=bejing_air_pm25' 'config.experiment_data.experiment_name=enbpi_forest_air25full' 'config.model_uc.past_window_len=200' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'

##### SPCI
python ./code/run_sweep_spic.py 'config/model_fc=darts_forest' 'config/task=default_3year_gn' 'config/dataset=bejing_air_pm25' 'config.experiment_data.experiment_name=spic_forest_air25full' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'

#### NexCP
python ./code/main.py 'config/model_fc=darts_forest' 'config/model_uc=nextcp' 'config/task=default_3year_gn' 'config/dataset=bejing_air_pm25' 'config.experiment_data.experiment_name=nexcp_forest_air25full' 'config.model_uc.rho=0.99' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'

#### Standard CP
python ./code/main.py 'config/model_fc=darts_forest' 'config/model_uc=conf_default' 'config/task=default_3year_gn' 'config/dataset=bejing_air_pm25' 'config.experiment_data.experiment_name=confDefault_forest_air25full' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'
```

#### LGBM
```sh
##### HopCPT
python ./code/run_sweep_eval.py 'config/model_fc=darts_lightgbm' 'config/model_uc=eps_pred_hopfield_3year' 'config/task=default_3year_gn' 'config/dataset=bejing_air_pm25' 'config.experiment_data.experiment_name=lgbm_hf_air25full' 'config.trainer.trainer_config.n_epochs=3000' 'config.model_uc.batch_size=2' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'

##### EnbPI
python ./code/main.py 'config/model_fc=darts_lightgbm' 'config/model_uc=enbpi' 'config/task=default_3year_gn' 'config/dataset=bejing_air_pm25' 'config.experiment_data.experiment_name=enbpi_lgbm_air25full' 'config.model_uc.past_window_len=200' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'

##### SPCI
python ./code/run_sweep_spic.py 'config/model_fc=darts_lightgbm' 'config/task=default_3year_gn' 'config/dataset=bejing_air_pm25' 'config.experiment_data.experiment_name=spic_lgbm_air25full' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'

#### NexCP
python ./code/main.py 'config/model_fc=darts_lightgbm' 'config/model_uc=nextcp' 'config/task=default_3year_gn' 'config/dataset=bejing_air_pm25' 'config.experiment_data.experiment_name=nexcp_lgbm_air25full' 'config.model_uc.rho=0.995' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'

#### Standard CP
python ./code/main.py 'config/model_fc=darts_lightgbm' 'config/model_uc=conf_default' 'config/task=default_3year_gn' 'config/dataset=bejing_air_pm25' 'config.experiment_data.experiment_name=confDefault_lgbm_air25full' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'

#### AdaptiveCI
python ./code/main.py 'config/model_fc=darts_lightgbm_quantile' 'config/model_uc=adaptiveci' 'config/task=default_3year_gn' 'config/dataset=bejing_air_pm25' 'config.experiment_data.experiment_name=adaptiveCI_dartsGB_air25full' 'config.model_uc.mode=simple' 'config.model_uc.gamma=0.002' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'
```


#### Ridge
```sh
##### HopCPT
python ./code/run_sweep_eval.py 'config/model_fc=reg_ridge' 'config/model_uc=eps_pred_hopfield_3year' 'config/task=default_3year_gn' 'config/dataset=bejing_air_pm25' 'config.experiment_data.experiment_name=ridge_hf_air25full' 'config.trainer.trainer_config.n_epochs=3000' 'config.model_uc.batch_size=2' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'

##### EnbPI
python ./code/main.py 'config/model_fc=reg_ridge' 'config/model_uc=enbpi' 'config/task=default_3year_gn' 'config/dataset=bejing_air_pm25' 'config.experiment_data.experiment_name=enbpi_ridge_air25full' 'config.model_uc.past_window_len=200' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'

##### SPCI
python ./code/run_sweep_spic.py 'config/model_fc=reg_ridge' 'config/task=default_3year_gn' 'config/dataset=bejing_air_pm25' 'config.experiment_data.experiment_name=spic_regRidge_air25full' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'
``
#### NexCP
python ./code/main.py 'config/model_fc=reg_ridge' 'config/model_uc=nextcp' 'config/task=default_3year_gn' 'config/dataset=bejing_air_pm25' 'config.experiment_data.experiment_name=nexcp_ridge_air25full' 'config.model_uc.rho=0.995' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'

#### Standard CP
python ./code/main.py 'config/model_fc=reg_ridge' 'config/model_uc=conf_default' 'config/task=default_3year_gn' 'config/dataset=bejing_air_pm25' 'config.experiment_data.experiment_name=confDefault_ridge_air25full' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'
```

### LSTM

Train LSTM Model (details see above)
```sh
python ./code/run_sweep_lstm_train.py 'config/task=default_3year_gn' 'config/dataset=bejing_air_pm25' 'config.experiment_data.experiment_name=trainsweep_air25' 'config.model_fc.model_params.plot_eval_after_train=false' 'config.trainer.trainer_config.n_epochs=150' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>' 'model_fc.model_params.lstm_conf.hidden_size=64' 'model_fc.model_params.dropout=0.1' 'model_fc.model_params.batch_size=512' 'trainer.trainer_config.optim.lr=0.001'
```

```sh
#### HopCPT
python code/run_sweep_eval.py 'config/model_fc=global_lstm_air25' 'config/model_uc=eps_pred_hopfield_3year' 'config/task=default_3year_gn' 'config/dataset=bejing_air_pm25' 'config.trainer.trainer_config.n_epochs=3000' 'config.experiment_data.experiment_name=lstm_hf_air25full_Eval' 'config.model_uc.batch_size=2' 'config.model_uc.ctx_encode_dropout=null' 'config.trainer.trainer_config.optim.lr=0.001' 'config.model_uc.pos_encode.mode=null' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>' 

#### EnbPI
python ./code/main.py 'config/model_fc=global_lstm_air25' 'config/model_uc=enbpi' 'config/task=default_3year_gn' 'config/dataset=bejing_air_pm25' 'config.experiment_data.experiment_name=enbpi_lstm_air25_EvalRun' 'config.model_uc.past_window_len=200' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>' 

#### SPCI
python ./code/run_sweep_spic.py 'config/model_fc=global_lstm_air25' 'config/task=default_3year_gn' 'config/dataset=bejing_air_pm25' 'config.experiment_data.experiment_name=spic_lstm_air25' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>' 

#### NexCP
python ./code/main.py 'config/model_fc=global_lstm_air25' 'config/model_uc=nextcp' 'config/task=default_3year_gn' 'config/dataset=bejing_air_pm25' 'config.experiment_data.experiment_name=ncp_lstm_air25_EvalRun' 'config.model_uc.rho=0.993'  'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'
 
####  StandardCP/CF-RNN
python ./code/main.py 'config/model_fc=global_lstm_air25' 'config/model_uc=conf_default' 'config/task=default_3year_gn' 'config/dataset=bejing_air_pm25' 'config.experiment_data.experiment_name=confDef_lstm_air25_EvalRun' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'

#### CopulaCPTS
python ./code/main.py 'config/model_fc=global_lstm_air25' 'config/model_uc=conf_default_plus_recent' 'config/task=default_3year_gn' 'config/dataset=bejing_air_pm25' 'config.experiment_data.experiment_name=confDef_lstm_air25_EvalRun' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'
```

## Run KNN Commands
KNN runs from Appendix C

You need to insert for `<absolute_project_root_dir>` to the absolute root path of this repository on your machine (e.g. `/home/yourhome/hopcpt/`)

```
# Solar Small - Forest/LGBM/Ridge
python code/main.py 'config/model_fc=darts_forest' 'config/model_uc=eps_sel_stat_knn' 'config/task=default_gn' 'config/dataset=enbPI_solar_all' 'config.experiment_data.experiment_name=forest_knn_solarSmall' 'config.model_uc.topk_used_share=0.05' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'
python code/main.py 'config/model_fc=darts_lightgbm' 'config/model_uc=eps_sel_stat_knn' 'config/task=default_gn' 'config/dataset=enbPI_solar_all'  'config.experiment_data.experiment_name=lgbm_knn_solarSmall' 'config.model_uc.topk_used_share=0.15' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'
python code/main.py 'config/model_fc=reg_ridge' 'config/model_uc=eps_sel_stat_knn' 'config/task=default_gn' 'config/dataset=enbPI_solar_all' 'config.experiment_data.experiment_name=ridge_knn_solarSmall' 'config.model_uc.topk_used_share=0.05' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'

# Solar 1Y - Forest/LGBM/Ridge
python code/main.py 'config/model_fc=darts_forest' 'config/model_uc=eps_sel_stat_knn' 'config/task=default_1year_gn' 'config/dataset=nsdb2019_60m' 'config.experiment_data.experiment_name=forest_knn_1yearSolar' 'config.model_uc.topk_used_share=0.15' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'
python code/main.py 'config/model_fc=darts_lightgbm' 'config/model_uc=eps_sel_stat_knn' 'config/task=default_1year_gn' 'config/dataset=nsdb2019_60m'  'config.experiment_data.experiment_name=lgbm_knn_1yearSolar' 'config.model_uc.topk_used_share=0.25' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'
python code/main.py 'config/model_fc=reg_ridge' 'config/model_uc=eps_sel_stat_knn' 'config/task=default_1year_gn' 'config/dataset=nsdb2019_60m' 'config.experiment_data.experiment_name=ridge_knn_1yearSolar' 'config.model_uc.topk_used_share=0.20' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>' 

# Solar 3Y - Forest/LGBM/Ridge
python code/main.py 'config/model_fc=darts_forest' 'config/model_uc=eps_sel_stat_knn' 'config/task=default_3year_gn' 'config/dataset=nsdb2018-20_60m'  'config.experiment_data.experiment_name=forest_knn_3yearSolar' 'config.model_uc.topk_used_share=0.20' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'
python code/main.py 'config/model_fc=darts_lightgbm' 'config/model_uc=eps_sel_stat_knn' 'config/task=default_3year_gn' 'config/dataset=nsdb2018-20_60m' 'config.experiment_data.experiment_name=lgbm_knn_3yearSolar' 'config.model_uc.topk_used_share=0.35' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'
python code/main.py 'config/model_fc=reg_ridge' 'config/model_uc=eps_sel_stat_knn' 'config/task=default_3year_gn' 'config/dataset=nsdb2018-20_60m'  'config.experiment_data.experiment_name=ridge_knn_3yearSolar' 'config.model_uc.topk_used_share=0.25 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'

# Air 10PM - Forest/LGBM/Ridge
python ./code/main.py 'config/model_fc=darts_forest' 'config/model_uc=eps_sel_stat_knn' 'config/task=default_3year_gn' 'config/dataset=bejing_air_pm10' 'config.experiment_data.experiment_name=forest_knn_air10full' 'config.model_uc.topk_used_share=0.35' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'
python ./code/main.py 'config/model_fc=darts_lightgbm' 'config/model_uc=eps_sel_stat_knn' 'config/task=default_3year_gn' 'config/dataset=bejing_air_pm10' 'config.experiment_data.experiment_name=lgbm_knn_air10full'  'config.model_uc.topk_used_share=0.35' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'
python ./code/main.py 'config/model_fc=reg_ridge' 'config/model_uc=eps_sel_stat_knn' 'config/task=default_3year_gn' 'config/dataset=bejing_air_pm10' 'config.experiment_data.experiment_name=ridge_knn_air10full' 'config.model_uc.topk_used_share=0.35' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'

# Air 25PM - Forest/LGBM/Ridge
python ./code/main.py 'config/model_fc=darts_forest' 'config/model_uc=eps_sel_stat_knn' 'config/task=default_3year_gn' 'config/dataset=bejing_air_pm25' 'config.experiment_data.experiment_name=forest_knn_air25full' 'config.model_uc.topk_used_share=0.35' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'
python ./code/main.py 'config/model_fc=darts_lightgbm' 'config/model_uc=eps_sel_stat_knn' 'config/task=default_3year_gn' 'config/dataset=bejing_air_pm25' 'config.experiment_data.experiment_name=lgbm_knn_air25full' 'config.model_uc.topk_used_share=0.35' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'
python ./code/main.py 'config/model_fc=reg_ridge' 'config/model_uc=eps_sel_stat_knn' 'config/task=default_3year_gn' 'config/dataset=bejing_air_pm25' 'config.experiment_data.experiment_name=ridge_knn_air25full' 'config.model_uc.topk_used_share=0.35' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'

# Sapflux - Forest/LGBM/Ridge
python code/main.py 'config/model_fc=darts_forest' 'config/model_uc=eps_sel_stat_knn' 'config/task=default_3year_gn' 'config/dataset=sapflux_solo3_large' 'config.experiment_data.experiment_name=forest_knn_sapflux3' 'config.model_uc.topk_used_share=0.35' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'
python code/main.py 'config/model_fc=darts_lightgbm' 'config/model_uc=eps_sel_stat_knn' 'config/task=default_3year_gn' 'config/dataset=sapflux_solo3_large' 'config.experiment_data.experiment_name=lgbm_knn_sapflux3' 'config.model_uc.topk_used_share=0.35' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'
python code/main.py 'config/model_fc=reg_ridge' 'config/model_uc=eps_sel_stat_knn' 'config/task=default_3year_gn' 'config/dataset=sapflux_solo3_large' 'config.experiment_data.experiment_name=ridge_knn_sapflux3' 'config.model_uc.topk_used_share=0.35' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'
```

### Run Non-CP Models (Appendix A3)

### MCD
```sh
#### Solar 3Y
python ./code/run_sweep_lstm_train_and_eval.py 'config/model_fc=global_lstm_solar3y_nocal' 'config/task=default_3year_gn' 'config/dataset=nsdb2018-20_60m' 'config.experiment_data.experiment_name=solar3Y_mcd' 'config.model_fc.model_params.plot_eval_after_train=false' 'config.trainer.trainer_config.n_epochs=150' 'config.trainer.trainer_config.optim.lr=0.001'

#### Solar 1Y
python ./code/run_sweep_lstm_train_and_eval.py 'config/model_fc=global_lstm_solar1y_nocal' 'config/task=default_1year_gn' 'config/dataset=nsdb2019_60m' 'config.experiment_data.experiment_name=solar1Y_mcd' 'config.model_fc.model_params.plot_eval_after_train=false' 'config.trainer.trainer_config.n_epochs=150' 'config.trainer.trainer_config.optim.lr=0.005'

#### Sapflux
python ./code/run_sweep_lstm_train_and_eval.py 'config/model_fc=global_lstm_sapflux_nocal' 'config/task=default_3year_gn' 'config/dataset=sapflux_solo3_large' 'config.experiment_data.experiment_name=sapflux_mcd' 'config.model_fc.model_params.plot_eval_after_train=false' 'config.trainer.trainer_config.n_epochs=150' 'config.trainer.trainer_config.optim.lr=0.001'

#### Air10
python ./code/run_sweep_lstm_train_and_eval.py 'config/model_fc=global_lstm_air10_nocal' 'config/task=default_3year_gn' 'config/dataset=bejing_air_pm10' 'config.experiment_data.experiment_name=air10_mcd' 'config.model_fc.model_params.plot_eval_after_train=false' 'config.trainer.trainer_config.n_epochs=150' 'config.trainer.trainer_config.optim.lr=0.0001'

#### Air25
python ./code/run_sweep_lstm_train_and_eval.py 'config/model_fc=global_lstm_air25_nocal' 'config/task=default_3year_gn' 'config/dataset=bejing_air_pm25' 'config.experiment_data.experiment_name=air25_mcd' 'config.model_fc.model_params.plot_eval_after_train=false' 'config.trainer.trainer_config.n_epochs=150' 'config.trainer.trainer_config.optim.lr=0.005'
```

### MDN
```sh
#### Solar 3Y
python ./code/run_sweep_mdn_train_and_eval.py 'config/model_fc=global_lstmmdn_3year' 'config/task=default_3year_gn' 'config/dataset=nsdb2018-20_60m' 'config.experiment_data.experiment_name=solar3y_mdn'  'config.model_fc.model_params.plot_eval_after_train=false' 'config.trainer.trainer_config.n_epochs=150' 'config.trainer.trainer_config.optim.lr=0.001'

#### Solar 1Y
python ./code/run_sweep_mdn_train_and_eval.py 'config/model_fc=global_lstmmdn_1year' 'config/task=default_1year_gn' 'config/dataset=nsdb2019_60m' 'config.experiment_data.experiment_name=solar1Y_mdn' 'config.model_fc.model_params.plot_eval_after_train=false' 'config.trainer.trainer_config.n_epochs=150'  'config.trainer.trainer_config.optim.lr=0.001'

#### Sapflux
python ./code/run_sweep_mdn_train_and_eval.py 'config/model_fc=global_lstmmdn_sapflux' 'config/task=default_3year_gn' 'config/dataset=sapflux_solo3_large' 'config.experiment_data.experiment_name=sapflux_mdn' 'config.model_fc.model_params.plot_eval_after_train=false' 'config.trainer.trainer_config.n_epochs=150'  'config.trainer.trainer_config.optim.lr=0.001'
 
##### Air10
python ./code/run_sweep_mdn_train_and_eval.py 'config/model_fc=global_lstmmdn_air10' 'config/task=default_3year_gn' 'config/dataset=bejing_air_pm10' 'config.experiment_data.experiment_name=air10_mdn' 'config.model_fc.model_params.plot_eval_after_train=false' 'config.trainer.trainer_config.n_epochs=150' 'config.trainer.trainer_config.optim.lr=0.005'

#### Air25
python ./code/run_sweep_mdn_train_and_eval.py 'config/model_fc=global_lstmmdn_air25' 'config/task=default_3year_gn' 'config/dataset=bejing_air_pm25' 'config.experiment_data.experiment_name=air25_mdn' 'config.model_fc.model_params.plot_eval_after_train=false' 'config.trainer.trainer_config.n_epochs=150' 'config.trainer.trainer_config.optim.lr=0.005'
```

### Gauss
```sh
#### Solar 3Y
python ./code/run_sweep_mdn_train_and_eval.py 'config/model_fc=global_lstm1DGaus_3year' 'config/task=default_3year_gn' 'config/dataset=nsdb2018-20_60m' 'config.experiment_data.experiment_name=solar3y_lstm1gaus'  'config.model_fc.model_params.plot_eval_after_train=false' 'config.trainer.trainer_config.n_epochs=150' 'config.trainer.trainer_config.optim.lr=0.005'

#### Solar 1Y
python ./code/run_sweep_mdn_train_and_eval.py 'config/model_fc=global_lstm1DGaus_1year' 'config/task=default_1year_gn' 'config/dataset=nsdb2019_60m' 'config.experiment_data.experiment_name=solar1Y_lstmGaus' 'config.model_fc.model_params.plot_eval_after_train=false' 'config.trainer.trainer_config.n_epochs=150'  'config.trainer.trainer_config.optim.lr=0.005'

#### Sapflux
python ./code/run_sweep_mdn_train_and_eval.py 'config/model_fc=global_lstm1DGaus_sapflux' 'config/task=default_3year_gn' 'config/dataset=sapflux_solo3_large' 'config.experiment_data.experiment_name=sapflux_lstm1gaus' 'config.model_fc.model_params.plot_eval_after_train=false' 'config.trainer.trainer_config.n_epochs=150' 'config.trainer.trainer_config.optim.lr=0.001'
 
#### Air25
python ./code/run_sweep_mdn_train_and_eval.py 'config/model_fc=global_lstm1DGaus_air25' 'config/task=default_3year_gn' 'config/dataset=bejing_air_pm25' 'config.experiment_data.experiment_name=air25_lstmGaus' 'config.model_fc.model_params.plot_eval_after_train=false' 'config.trainer.trainer_config.n_epochs=150' 'config.trainer.trainer_config.optim.lr=0.001'

#### Air10
python ./code/run_sweep_mdn_train_and_eval.py 'config/model_fc=global_lstm1DGaus_air10' 'config/task=default_3year_gn' 'config/dataset=bejing_air_pm10' 'config.experiment_data.experiment_name=air10_lstm1gaus' 'config.model_fc.model_params.plot_eval_after_train=false' 'config.trainer.trainer_config.n_epochs=150' 'config.trainer.trainer_config.optim.lr=0.001'
```

### Models + AdaptiveCI

To run a model together with AdaptiveCI (Appendix F) simply add the following two arguments to your pyhton call
```
'config.use_adaptiveci=True' 'config.model_uc.rho=<EnterYourDecayLevel>'
```

### Runs SPCI Ablations
For running with a Window Size 25 set additional command (Appendix A1)
```
'config.model_uc.past_window_len=25'
```

For running with retrain set additional command (Appendix A2), set model_uc spec (`config/model_uc=`) to
```
'config/model_uc=spic_retrain'
```
