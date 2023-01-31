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
##### HopCPT
```sh
python code/run_sweep_eval.py 'config/model_fc=darts_forest' 'config/model_uc=eps_pred_hopfield_3year' 'config/task=default_3year_gn' 'config/dataset=nsdb2018-20_60m' 'config.trainer.trainer_config.n_epochs=3000' 'config.experiment_data.experiment_name=forest_hf_solar3year' 'config.model_uc.eps_mem_size=5000' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'
```
##### EnbPI
```sh
python ./code/main.py 'config/model_fc=darts_forest' 'config/model_uc=enbpi' 'config/task=default_3year_gn' 'config/dataset=nsdb2018-20_60m' 'config.experiment_data.experiment_name=enbpi_forest_3yearSolar' 'config.model_uc.past_window_len=125' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'
```
##### SPCI
```sh
python ./code/run_sweep_spic.py 'config/model_fc=darts_forest' 'config/task=default_3year_gn' 'config/dataset=nsdb2018-20_60m' 'config.experiment_data.experiment_name=spic_forest_3yearSolar' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'
```
#### NexCP
```sh
python ./code/main.py 'config/model_fc=darts_forest' 'config/model_uc=nextcp' 'config/task=default_3year_gn' 'config/dataset=nsdb2018-20_60m' 'config.experiment_data.experiment_name=nextcp_forest_3yearSolar' 'config.model_uc.rho=0.99' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'
```
#### Standard CP
```sh
python ./code/main.py 'config/model_fc=darts_forest' 'config/model_uc=conf_default' 'config/task=default_3year_gn' 'config/dataset=nsdb2018-20_60m' 'config.experiment_data.experiment_name=confDefault_forest_3yearSolar' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'
```


#### LGBM
##### HopCPT
```sh
python code/run_sweep_eval.py 'config/model_fc=darts_lightgbm' 'config/model_uc=eps_pred_hopfield_3year' 'config/task=default_3year_gn' 'config/dataset=nsdb2018-20_60m' 'config.trainer.trainer_config.n_epochs=3000' 'config.experiment_data.experiment_name=lgbm_hf_solar3year' 'config.model_uc.eps_mem_size=5000' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'
```
##### EnbPI
```sh
python ./code/main.py 'config/model_fc=darts_lightgbm' 'config/model_uc=enbpi' 'config/task=default_3year_gn' 'config/dataset=nsdb2018-20_60m' 'config.experiment_data.experiment_name=enbpi_lgbm_3yearSolar' 'config.model_uc.past_window_len=200' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'
```
##### SPCI
```sh
python ./code/run_sweep_spic.py 'config/model_fc=darts_lightgbm' 'config/task=default_3year_gn' 'config/dataset=nsdb2018-20_60m' 'config.experiment_data.experiment_name=spic_lgbm_3yearSolar' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'
```
#### NexCP
```sh
python ./code/main.py 'config/model_fc=darts_lightgbm' 'config/model_uc=nextcp' 'config/task=default_3year_gn' 'config/dataset=nsdb2018-20_60m' 'config.experiment_data.experiment_name=nextcp_lgbm_3yearSolar' 'config.model_uc.rho=0.99' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'
```
#### Standard CP
```sh
python ./code/main.py 'config/model_fc=darts_lightgbm' 'config/model_uc=conf_default' 'config/task=default_3year_gn' 'config/dataset=nsdb2018-20_60m' 'config.experiment_data.experiment_name=confDefault_lgbm_3yearSolar' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'
```
#### AdaptiveCI
```sh
python ./code/main.py 'config/model_fc=darts_lightgbm_quantile' 'config/model_uc=adaptiveci' 'config/task=default_3year_gn' 'config/dataset=nsdb2018-20_60m' 'config.experiment_data.experiment_name=adaptiveCI_dartsGB_3yearSolar' 'config.model_uc.mode=simple' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'
```


#### Ridge
##### HopCPT
```sh
python code/run_sweep_eval.py 'config/model_fc=reg_ridge' 'config/model_uc=eps_pred_hopfield_3year' 'config/task=default_3year_gn' 'config/dataset=nsdb2018-20_60m' 'config.trainer.trainer_config.n_epochs=3000' 'config.experiment_data.experiment_name=ridge_hf_solar3year' 'config.model_uc.eps_mem_size=5000' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'
```
##### EnbPI
```sh
python ./code/main.py 'config/model_fc=reg_ridge' 'config/model_uc=enbpi' 'config/task=default_3year_gn' 'config/dataset=nsdb2018-20_60m' 'config.experiment_data.experiment_name=enbpi_ridge_3yearSolar' 'config.model_uc.past_window_len=25' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'
```
##### SPCI
```sh
python ./code/run_sweep_spic.py 'config/model_fc=reg_ridge' 'config/task=default_3year_gn' 'config/dataset=nsdb2018-20_60m' 'config.experiment_data.experiment_name=spic_regridge_3yearSolar' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'
```
#### NexCP
```sh
python ./code/main.py 'config/model_fc=reg_ridge' 'config/model_uc=nextcp' 'config/task=default_3year_gn' 'config/dataset=nsdb2018-20_60m' 'config.experiment_data.experiment_name=nextcp_ridge_3yearSolar' 'config.model_uc.rho=0.999' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'
```
#### Standard CP
```sh
python ./code/main.py 'config/model_fc=reg_ridge' 'config/model_uc=conf_default' 'config/task=default_3year_gn' 'config/dataset=nsdb2018-20_60m' 'config.experiment_data.experiment_name=confDefault_ridge_3yearSolar' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>' 
```

### Air 10

#### Forest
##### HopCPT
```sh
python ./code/run_sweep_eval.py 'config/model_fc=darts_forest' 'config/model_uc=eps_pred_hopfield_3year' 'config/task=default_3year_gn' 'config/dataset=bejing_air_pm10' 'config.experiment_data.experiment_name=forest_hf_air10full' 'config.trainer.trainer_config.n_epochs=3000' 'config.model_uc.batch_size=2' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'
```
##### EnbPI
```sh
python ./code/main.py 'config/model_fc=darts_forest' 'config/model_uc=enbpi' 'config/task=default_3year_gn' 'config/dataset=bejing_air_pm10' 'config.experiment_data.experiment_name=enbpi_forest_air10full' 'config.model_uc.past_window_len=200' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'
```
##### SPCI
```sh
python ./code/run_sweep_spic.py 'config/model_fc=darts_forest' 'config/task=default_3year_gn' 'config/dataset=bejing_air_pm10' 'config.experiment_data.experiment_name=spic_forest_air10full' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'
```
#### NexCP
```sh
python ./code/main.py 'config/model_fc=darts_forest' 'config/model_uc=nextcp' 'config/task=default_3year_gn' 'config/dataset=bejing_air_pm10' 'config.experiment_data.experiment_name=nextcp_forest_air10full' 'config.model_uc.rho=0.993' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'
```
#### Standard CP
```sh
python ./code/main.py 'config/model_fc=darts_forest' 'config/model_uc=conf_default' 'config/task=default_3year_gn' 'config/dataset=bejing_air_pm10' 'config.experiment_data.experiment_name=confDefault_forest_air10full' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'
```

#### LGBM
##### HopCPT
```sh
python ./code/run_sweep_eval.py 'config/model_fc=darts_lightgbm' 'config/model_uc=eps_pred_hopfield_3year' 'config/task=default_3year_gn' 'config/dataset=bejing_air_pm10' 'config.experiment_data.experiment_name=lgbm_hf_air10full' 'config.trainer.trainer_config.n_epochs=3000' 'config.model_uc.batch_size=2' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'
```
##### EnbPI
```sh
python ./code/main.py 'config/model_fc=darts_lightgbm' 'config/model_uc=enbpi' 'config/task=default_3year_gn' 'config/dataset=bejing_air_pm10' 'config.experiment_data.experiment_name=enbpi_lgbm_air10full' 'config.model_uc.past_window_len=200' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'
```
##### SPCI
```sh
python ./code/run_sweep_spic.py 'config/model_fc=darts_lightgbm' 'config/task=default_3year_gn' 'config/dataset=bejing_air_pm10' 'config.experiment_data.experiment_name=spic_lgbm_air10full' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'
```
#### NexCP
```sh
python ./code/main.py 'config/model_fc=darts_lightgbm' 'config/model_uc=nextcp' 'config/task=default_3year_gn' 'config/dataset=bejing_air_pm10' 'config.experiment_data.experiment_name=nextcp_lgbm_air10full' 'config.model_uc.rho=0.993' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'
```
#### Standard CP
```sh
python ./code/main.py 'config/model_fc=darts_lightgbm' 'config/model_uc=conf_default' 'config/task=default_3year_gn' 'config/dataset=bejing_air_pm10' 'config.experiment_data.experiment_name=confDefault_lgbm_air10full' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'
```
#### AdaptiveCI
```sh
python ./code/main.py 'config/model_fc=darts_lightgbm_quantile' 'config/model_uc=adaptiveci' 'config/task=default_3year_gn' 'config/dataset=bejing_air_pm10' 'config.experiment_data.experiment_name=adaptiveCI_dartsGB_air10full' 'config.model_uc.mode=simple' 'config.model_uc.gamma=0.01' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'
```


#### Ridge
##### HopCPT
```sh
python ./code/run_sweep_eval.py 'config/model_fc=reg_ridge' 'config/model_uc=eps_pred_hopfield_3year' 'config/task=default_3year_gn' 'config/dataset=bejing_air_pm10' 'config.experiment_data.experiment_name=ridge_hf_air10full' 'config.trainer.trainer_config.n_epochs=3000' 'config.model_uc.batch_size=2' 'config.model_uc.ctx_encode_dropout=0.25' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'
```
##### EnbPI
```sh
python ./code/main.py 'config/model_fc=reg_ridge' 'config/model_uc=enbpi' 'config/task=default_3year_gn' 'config/dataset=bejing_air_pm10' 'config.experiment_data.experiment_name=enbpi_ridge_air10full' 'config.model_uc.past_window_len=200' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'
```
##### SPCI
```sh
python ./code/run_sweep_spic.py 'config/model_fc=reg_ridge' 'config/task=default_3year_gn' 'config/dataset=bejing_air_pm10' 'config.experiment_data.experiment_name=spic_regridge_air10full' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'
```
#### NexCP
```sh
python ./code/main.py 'config/model_fc=reg_ridge' 'config/model_uc=nextcp' 'config/task=default_3year_gn' 'config/dataset=bejing_air_pm10' 'config.experiment_data.experiment_name=nextcp_ridge_air10full' 'config.model_uc.rho=0.993' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'
```
#### Standard CP
```sh
python ./code/main.py 'config/model_fc=reg_ridge' 'config/model_uc=conf_default' 'config/task=default_3year_gn' 'config/dataset=bejing_air_pm10' 'config.experiment_data.experiment_name=confDefault_ridge_air10full' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'
```

### Sapflux

#### Forest
##### HopCPT
```sh
python ./code/run_sweep_eval.py 'config/model_fc=darts_forest' 'config/model_uc=eps_pred_hopfield_3year' 'config/task=default_3year_gn' 'config/dataset=sapflux_solo3_large' 'config.trainer.trainer_config.n_epochs=3000' 'config.experiment_data.experiment_name=forest_hf_sapflux3' 'config.model_uc.eps_mem_size=4000' 'config.model_uc.pos_encode.mode=rel-simple' 'config.model_uc.batch_size=1' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'
```
##### EnbPI
```sh
python code/main.py 'config/model_fc=darts_forest' 'config/model_uc=enbpi' 'config/task=default_3year_gn' 'config/dataset=sapflux_solo3_large' 'config.experiment_data.experiment_name=forest_enbpi_sapflux3' 'config.model_uc.past_window_len=200' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'
```
##### SPCI
```sh
python code/run_sweep_spic.py 'config/model_fc=darts_forest' 'config/task=default_3year_gn' 'config/dataset=sapflux_solo3_large' 'config.experiment_data.experiment_name=forest_spic_sapflux3' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'
```
#### NexCP
```sh
python code/main.py 'config/model_fc=darts_forest' 'config/model_uc=nextcp' 'config/task=default_3year_gn' 'config/dataset=sapflux_solo3_large' 'config.experiment_data.experiment_name=forest_nextcp_sapflux3' 'config.model_uc.rho=0.995' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'
```
#### Standard CP
```sh
python code/main.py 'config/model_fc=darts_forest' 'config/model_uc=conf_default' 'config/task=default_3year_gn' 'config/dataset=sapflux_solo3_large' 'config.experiment_data.experiment_name=forest_confDefault_sapflux3' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'
```

#### LGBM
##### HopCPT
```sh
python ./code/run_sweep_eval.py 'config/model_fc=darts_lightgbm' 'config/model_uc=eps_pred_hopfield_3year' 'config/task=default_3year_gn' 'config/dataset=sapflux_solo3_large' 'config.trainer.trainer_config.n_epochs=3000' 'config.experiment_data.experiment_name=lgbm_hf_sapflux3' 'config.model_uc.eps_mem_size=4000' 'config.model_uc.pos_encode.mode=rel-simple' 'config.model_uc.batch_size=1' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'
```
##### EnbPI
```sh
python code/main.py 'config/model_fc=darts_lightgbm' 'config/model_uc=enbpi' 'config/task=default_3year_gn' 'config/dataset=sapflux_solo3_large' 'config.experiment_data.experiment_name=lgbm_enbpi_sapflux3' 'config.model_uc.past_window_len=200' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'

```
##### SPCI
```sh
python code/run_sweep_spic.py 'config/model_fc=darts_lightgbm' 'config/task=default_3year_gn' 'config/dataset=sapflux_solo3_large' 'config.experiment_data.experiment_name=lgbm_spic_sapflux3' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'
```
#### NexCP
```sh
python code/main.py 'config/model_fc=darts_lightgbm' 'config/model_uc=nextcp' 'config/task=default_3year_gn' 'config/dataset=sapflux_solo3_large' 'config.experiment_data.experiment_name=lgbm_nextcp_sapflux3' 'config.model_uc.rho=0.995' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'
```
#### Standard CP
```sh
python code/main.py 'config/model_fc=darts_lightgbm' 'config/model_uc=conf_default' 'config/task=default_3year_gn' 'config/dataset=sapflux_solo3_large' 'config.experiment_data.experiment_name=lgbm_confDefault_sapflux3' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'
```
#### AdaptiveCI
```sh
python ./code/main.py 'config/model_fc=darts_lightgbm_quantile' 'config/model_uc=adaptiveci' 'config/task=default_3year_gn' 'config/dataset=sapflux_solo3_large' 'config.experiment_data.experiment_name=adaptiveCI_dartsGB_sapflux3' 'config.model_uc.mode=simple' 'config.model_uc.gamma=0.002' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'
```


#### Ridge
##### HopCPT
```sh
python ./code/run_sweep_eval.py 'config/model_fc=reg_ridge' 'config/model_uc=eps_pred_hopfield_3year' 'config/task=default_3year_gn' 'config/dataset=sapflux_solo3_large' 'config.trainer.trainer_config.n_epochs=3000' 'config.experiment_data.experiment_name=ridgeReg_hf_sapflux3' 'config.model_uc.eps_mem_size=4000' 'config.model_uc.pos_encode.mode=rel-simple' 'config.model_uc.batch_size=1' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'
```
##### EnbPI
```sh
python code/main.py 'config/model_fc=reg_ridge' 'config/model_uc=enbpi' 'config/task=default_3year_gn' 'config/dataset=sapflux_solo3_large' 'config.experiment_data.experiment_name=ridge_enbpi_sapflux3' 'config.model_uc.past_window_len=150' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'
```
##### SPCI
```sh
python code/run_sweep_spic.py 'config/model_fc=reg_ridge' 'config/task=default_3year_gn' 'config/dataset=sapflux_solo3_large' 'config.experiment_data.experiment_name=regRidge_spic_sapflux3' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'
```
#### NexCP
```sh
python code/main.py 'config/model_fc=reg_ridge' 'config/model_uc=nextcp' 'config/task=default_3year_gn' 'config/dataset=sapflux_solo3_large' 'config.experiment_data.experiment_name=ridge_nextcp_sapflux3' 'config.model_uc.rho=0.995' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'
```
#### Standard CP
```sh
python code/main.py 'config/model_fc=reg_ridge' 'config/model_uc=conf_default' 'config/task=default_3year_gn' 'config/dataset=sapflux_solo3_large' 'config.experiment_data.experiment_name=ridge_confDefault_sapflux3' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'
```

### Streamflow

For the streamflow runs additional data is needed:
1) Download camel_us as described here https://neuralhydrology.readthedocs.io/en/latest/tutorials/data-prerequisites.html:
The relevant subsection are `CAMELS US catchment attributes` and `CAMELS US meteorological time series and streamflow data` 
Then you copy the extracted files in your local repo to `<repo-root>/data/hydrology/CAMELS_US`.
2) Download https://www.hydroshare.org/resource/17c896843cf940339c3c3496d0c1c077/ and place zip content into `<repo-root>/data/hydrology/CAMELS_US/basin_mean_forcing/maurer_extended`.
3) Download https://www.hydroshare.org/resource/0a68bfd7ddf642a8be9041d60f40868c/ and place tar content into `<repo-root>/data/hydrology/CAMELS_US/basin_mean_forcing/nldas_extended`.


##### HopCPT
```sh
python code/run_sweep_eval.py 'config/model_fc=precomputed_nh' 'config/model_uc=eps_pred_hopfield_3year' 'config/task=default_3year_gn' 'config/dataset=hydro' 'config.trainer.trainer_config.n_epochs=3000' 'config.experiment_data.experiment_name=forest_hf_streamflow' 'config.model_uc.eps_mem_size=4000' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'
```
##### EnbPI
```sh
python ./code/main.py 'config/model_fc=precomputed_nh' 'config/model_uc=enbpi' 'config/task=default_3year_gn' 'config/dataset=hydro' 'config.experiment_data.experiment_name=enbpi_forest_streamflow' 'config.model_uc.past_window_len=200' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'
```
##### SPCI
```sh
python ./code/run_sweep_spic.py 'config/model_fc=precomputed_nh' 'config/task=default_3year_gn' 'config/dataset=hydro' 'config.experiment_data.experiment_name=spic_forest_streamflow' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'
```
#### NexCP
```sh
python ./code/main.py 'config/model_fc=precomputed_nh' 'config/model_uc=nextcp' 'config/task=default_3year_gn' 'config/dataset=hydro' 'config.experiment_data.experiment_name=nextcp_forest_streamflow' 'config.model_uc.rho=0.993' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'
```
#### Standard CP
```sh
python ./code/main.py 'config/model_fc=precomputed_nh' 'config/model_uc=conf_default' 'config/task=default_3year_gn' 'config/dataset=hydro' 'config.experiment_data.experiment_name=confDefault_forest_streamflow' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'
```


### Solar 1Y

#### Forest
##### HopCPT
```sh
python ./code/run_sweep_eval.py 'config/model_fc=darts_forest' 'config/model_uc=eps_pred_hopfield_1year' 'config/task=default_1year_gn' 'config/dataset=nsdb2019_60m'  'config.trainer.trainer_config.n_epochs=3000' 'config.experiment_data.experiment_name=forest_hf_soalr1year' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'
```
##### EnbPI
```sh
python ./code/main.py 'config/model_fc=darts_forest' 'config/model_uc=enbpi' 'config/task=default_1year_gn' 'config/dataset=nsdb2019_60m' 'config.experiment_data.experiment_name=enbpi_forest_1yearSolar' 'config.model_uc.past_window_len=200' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'
```
##### SPCI
```sh
python ./code/run_sweep_spic.py 'config/model_fc=darts_forest' 'config/task=default_1year_gn' 'config/dataset=nsdb2019_60m' 'config.experiment_data.experiment_name=spic_forest_1yearSolar' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'
```
#### NexCP
```sh
python ./code/main.py 'config/model_fc=darts_forest' 'config/model_uc=nextcp' 'config/task=default_1year_gn' 'config/dataset=nsdb2019_60m' 'config.experiment_data.experiment_name=nextcp_forest_1yearSolar' 'config.model_uc.rho=0.99' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'
```
#### Standard CP
```sh
python ./code/main.py 'config/model_fc=darts_forest' 'config/model_uc=conf_default' 'config/task=default_1year_gn' 'config/dataset=nsdb2019_60m' 'config.experiment_data.experiment_name=confDefault_forest_1yearSolar' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'
```

#### LGBM
##### HopCPT
```sh
python ./code/run_sweep_eval.py 'config/model_fc=darts_lightgbm' 'config/model_uc=eps_pred_hopfield_1year' 'config/task=default_1year_gn' 'config/dataset=nsdb2019_60m'  'config.trainer.trainer_config.n_epochs=3000' 'config.experiment_data.experiment_name=lgbm_hf_solar1year' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'
```
##### EnbPI
```sh
python ./code/main.py 'config/model_fc=darts_lightgbm' 'config/model_uc=enbpi' 'config/task=default_1year_gn' 'config/dataset=nsdb2019_60m' 'config.experiment_data.experiment_name=enbpi_lgbm_1yearSolar' 'config.model_uc.past_window_len=200' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'
```
##### SPCI
```sh
python ./code/run_sweep_spic.py 'config/model_fc=darts_lightgbm' 'config/task=default_1year_gn' 'config/dataset=nsdb2019_60m' 'config.experiment_data.experiment_name=spic_lgbm_1yearSolar' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'
```
#### NexCP
```sh
python ./code/main.py 'config/model_fc=darts_lightgbm' 'config/model_uc=nextcp' 'config/task=default_1year_gn' 'config/dataset=nsdb2019_60m' 'config.experiment_data.experiment_name=nextcp_lgmb_1yearSolar_EvalRun' 'config.model_uc.rho=0.99' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'
```
#### Standard CP
```sh
python ./code/main.py 'config/model_fc=darts_lightgbm' 'config/model_uc=conf_default' 'config/task=default_1year_gn' 'config/dataset=nsdb2019_60m' 'config.experiment_data.experiment_name=confDefault_lgbm_1yearSolar' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'
```
#### AdaptiveCI
```sh
python ./code/main.py 'config/model_fc=darts_lightgbm_quantile' 'config/model_uc=adaptiveci' 'config/task=default_1year_gn' 'config/dataset=nsdb2019_60m' 'config.experiment_data.experiment_name=adaptiveCI_dartsGB_1yearSolar' 'config.model_uc.mode=simple' 'config.model_uc.gamma=0.02' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'
```


#### Ridge
##### HopCPT
```sh
python ./code/run_sweep_eval.py 'config/model_fc=reg_ridge' 'config/model_uc=eps_pred_hopfield_1year' 'config/task=default_1year_gn' 'config/dataset=nsdb2019_60m'  'config.trainer.trainer_config.n_epochs=3000' 'config.experiment_data.experiment_name=ridge_hf_solar1year' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'
```
##### EnbPI
```sh
python ./code/main.py 'config/model_fc=reg_ridge' 'config/model_uc=enbpi' 'config/task=default_1year_gn' 'config/dataset=nsdb2019_60m' 'config.experiment_data.experiment_name=enbpi_ridge_1yearSolar' 'config.model_uc.past_window_len=200' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'
```
##### SPCI
```sh
on ./code/run_sweep_spic.py 'config/model_fc=reg_ridge' 'config/task=default_1year_gn' 'config/dataset=nsdb2019_60m' 'config.experiment_data.experiment_name=spic_regridge_1yearSolar' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'
```
#### NexCP
```sh
python ./code/main.py 'config/model_fc=reg_ridge' 'config/model_uc=nextcp' 'config/task=default_1year_gn' 'config/dataset=nsdb2019_60m' 'config.experiment_data.experiment_name=nextcp_ridge_1yearSolar' 'config.model_uc.rho=0.999' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'
```
#### Standard CP
```sh
python ./code/main.py 'config/model_fc=reg_ridge' 'config/model_uc=conf_default' 'config/task=default_1year_gn' 'config/dataset=nsdb2019_60m' 'config.experiment_data.experiment_name=confDefault_ridge_1yearSolar' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'
```

### Solar Small

#### Forest
##### HopCPT
```sh
python ./code/run_sweep_eval.py 'config/model_fc=darts_forest' 'config/model_uc=eps_pred_hopfield' 'config/task=default_gn' 'config/dataset=enbPI_solar_all'  'config.trainer.trainer_config.n_epochs=3000' 'config.experiment_data.experiment_name=forest_hf_smallSolar' 'config.model_uc.ctx_encode_dropout=0.50' 'config.trainer.trainer_config.optim.lr=0.001' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'
```
##### EnbPI
```sh
python ./code/main.py 'config/model_fc=darts_forest' 'config/model_uc=enbpi' 'config/task=default_gn' 'config/dataset=enbPI_solar_all' 'config.experiment_data.experiment_name=enbpi_forest_smallSolar' 'config.model_uc.past_window_len=150' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'
```
##### SPCI
```sh
python ./code/run_sweep_spic.py 'config/model_fc=darts_forest' 'config/task=default_gn' 'config/dataset=enbPI_solar_all' 'config.experiment_data.experiment_name=spic_forest_smallSolar' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'
```
#### NexCP
```sh
python ./code/main.py 'config/model_fc=darts_forest' 'config/model_uc=nextcp' 'config/task=default_gn' 'config/dataset=enbPI_solar_all' 'config.experiment_data.experiment_name=nextcp_forest_smallSolar' 'config.model_uc.rho=0.98' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'
```
#### Standard CP
```sh
python ./code/main.py 'config/model_fc=darts_forest' 'config/model_uc=conf_default' 'config/task=default_gn' 'config/dataset=enbPI_solar_all' 'config.experiment_data.experiment_name=confDefault_forest_smallSolar' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'
```

#### LGBM
##### HopCPT
```sh
python ./code/run_sweep_eval.py 'config/model_fc=darts_lightgbm' 'config/model_uc=eps_pred_hopfield' 'config/task=default_gn' 'config/dataset=enbPI_solar_all'  'config.trainer.trainer_config.n_epochs=3000' 'config.experiment_data.experiment_name=lgbm_hf_smallSolar' 'config.model_uc.ctx_encode_dropout=0.50' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'
```
##### EnbPI
```sh
python ./code/main.py 'config/model_fc=darts_lightgbm' 'config/model_uc=enbpi' 'config/task=default_gn' 'config/dataset=enbPI_solar_all' 'config.experiment_data.experiment_name=enbpi_lgbm_smallSolar' 'config.model_uc.past_window_len=150' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'
```
##### SPCI
```sh
python ./code/run_sweep_spic.py 'config/model_fc=darts_lightgbm' 'config/task=default_gn' 'config/dataset=enbPI_solar_all' 'config.experiment_data.experiment_name=spic_lgbm_smallSolar' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'
```
#### NexCP
```sh
python ./code/main.py 'config/model_fc=darts_lightgbm' 'config/model_uc=nextcp' 'config/task=default_gn' 'config/dataset=enbPI_solar_all' 'config.experiment_data.experiment_name=nextcp_lgbm_smallSolar_EvalRun' 'config.model_uc.rho=0.98' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'
```
#### Standard CP
```sh
python ./code/main.py 'config/model_fc=darts_lightgbm' 'config/model_uc=conf_default' 'config/task=default_gn' 'config/dataset=enbPI_solar_all' 'config.experiment_data.experiment_name=confDefault_lgbm_smallSolar' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'
```
#### AdaptiveCI
```sh
python ./code/main.py 'config/model_fc=darts_lightgbm_quantile' 'config/model_uc=adaptiveci' 'config/task=default_gn' 'config/dataset=enbPI_solar_all' 'config.experiment_data.experiment_name=adaptiveCI_dartsGB_smSolar' 'config.model_uc.mode=simple' 'config.model_uc.gamma=0.02' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'
```


#### Ridge
##### HopCPT
```sh
python ./code/run_sweep_eval.py 'config/model_fc=reg_ridge' 'config/model_uc=eps_pred_hopfield' 'config/task=default_gn' 'config/dataset=enbPI_solar_all'  'config.trainer.trainer_config.n_epochs=3000' 'config.experiment_data.experiment_name=ridge_hf_smallSolar'  'config.model_uc.ctx_encode_dropout=0.50' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'
```
##### EnbPI
```sh
python ./code/main.py 'config/model_fc=reg_ridge' 'config/model_uc=enbpi' 'config/task=default_gn' 'config/dataset=enbPI_solar_all' 'config.experiment_data.experiment_name=enbpi_ridge_smallSolar' 'config.model_uc.past_window_len=100' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'

```
##### SPCI
```sh
python ./code/run_sweep_spic.py 'config/model_fc=reg_ridge' 'config/task=default_gn' 'config/dataset=enbPI_solar_all' 'config.experiment_data.experiment_name=spic_ridge_smallSolar' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'
```
#### NexCP
```sh
python ./code/main.py 'config/model_fc=reg_ridge' 'config/model_uc=nextcp' 'config/task=default_gn' 'config/dataset=enbPI_solar_all' 'config.experiment_data.experiment_name=nextcp_ridge_smallSolar_EvalRun' 'config.model_uc.rho=0.999' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'
```
#### Standard CP
```sh
python ./code/main.py 'config/model_fc=reg_ridge' 'config/model_uc=conf_default' 'config/task=default_gn' 'config/dataset=enbPI_solar_all' 'config.experiment_data.experiment_name=confDefault_ridge_smallSolar' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'
```

### Air 25

#### Forest
##### HopCPT
```sh
python ./code/run_sweep_eval.py 'config/model_fc=darts_forest' 'config/model_uc=eps_pred_hopfield_3year' 'config/task=default_3year_gn' 'config/dataset=bejing_air_pm25' 'config.experiment_data.experiment_name=forest_hf_air25full' 'config.trainer.trainer_config.n_epochs=3000' 'config.model_uc.batch_size=2' 'config.model_uc.pos_encode.mode=rel-simple' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'
```
##### EnbPI
```sh
python ./code/main.py 'config/model_fc=darts_forest' 'config/model_uc=enbpi' 'config/task=default_3year_gn' 'config/dataset=bejing_air_pm25' 'config.experiment_data.experiment_name=enbpi_forest_air25full' 'config.model_uc.past_window_len=200' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'
```
##### SPCI
```sh
python ./code/run_sweep_spic.py 'config/model_fc=darts_forest' 'config/task=default_3year_gn' 'config/dataset=bejing_air_pm25' 'config.experiment_data.experiment_name=spic_forest_air25full' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'
```
#### NexCP
```sh
python ./code/main.py 'config/model_fc=darts_forest' 'config/model_uc=nextcp' 'config/task=default_3year_gn' 'config/dataset=bejing_air_pm25' 'config.experiment_data.experiment_name=nexcp_forest_air25full' 'config.model_uc.rho=0.99' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'
```
#### Standard CP
```sh
python ./code/main.py 'config/model_fc=darts_forest' 'config/model_uc=conf_default' 'config/task=default_3year_gn' 'config/dataset=bejing_air_pm25' 'config.experiment_data.experiment_name=confDefault_forest_air25full' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'
```

#### LGBM
##### HopCPT
```sh
python ./code/run_sweep_eval.py 'config/model_fc=darts_lightgbm' 'config/model_uc=eps_pred_hopfield_3year' 'config/task=default_3year_gn' 'config/dataset=bejing_air_pm25' 'config.experiment_data.experiment_name=lgbm_hf_air25full' 'config.trainer.trainer_config.n_epochs=3000' 'config.model_uc.batch_size=2' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'
```
##### EnbPI
```sh
python ./code/main.py 'config/model_fc=darts_lightgbm' 'config/model_uc=enbpi' 'config/task=default_3year_gn' 'config/dataset=bejing_air_pm25' 'config.experiment_data.experiment_name=enbpi_lgbm_air25full' 'config.model_uc.past_window_len=200' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'
```
##### SPCI
```sh
python ./code/run_sweep_spic.py 'config/model_fc=darts_lightgbm' 'config/task=default_3year_gn' 'config/dataset=bejing_air_pm25' 'config.experiment_data.experiment_name=spic_lgbm_air25full' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'
```
#### NexCP
```sh
python ./code/main.py 'config/model_fc=darts_lightgbm' 'config/model_uc=nextcp' 'config/task=default_3year_gn' 'config/dataset=bejing_air_pm25' 'config.experiment_data.experiment_name=nexcp_lgbm_air25full' 'config.model_uc.rho=0.995' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'
```
#### Standard CP
```sh
python ./code/main.py 'config/model_fc=darts_lightgbm' 'config/model_uc=conf_default' 'config/task=default_3year_gn' 'config/dataset=bejing_air_pm25' 'config.experiment_data.experiment_name=confDefault_lgbm_air25full' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'
```
#### AdaptiveCI
```sh
python ./code/main.py 'config/model_fc=darts_lightgbm_quantile' 'config/model_uc=adaptiveci' 'config/task=default_3year_gn' 'config/dataset=bejing_air_pm25' 'config.experiment_data.experiment_name=adaptiveCI_dartsGB_air25full' 'config.model_uc.mode=simple' 'config.model_uc.gamma=0.002' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'
```


#### Ridge
##### HopCPT
```sh
python ./code/run_sweep_eval.py 'config/model_fc=reg_ridge' 'config/model_uc=eps_pred_hopfield_3year' 'config/task=default_3year_gn' 'config/dataset=bejing_air_pm25' 'config.experiment_data.experiment_name=ridge_hf_air25full' 'config.trainer.trainer_config.n_epochs=3000' 'config.model_uc.batch_size=2' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'
```
##### EnbPI
```sh
python ./code/main.py 'config/model_fc=reg_ridge' 'config/model_uc=enbpi' 'config/task=default_3year_gn' 'config/dataset=bejing_air_pm25' 'config.experiment_data.experiment_name=enbpi_ridge_air25full' 'config.model_uc.past_window_len=200' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'
```
##### SPCI
```sh
python ./code/run_sweep_spic.py 'config/model_fc=reg_ridge' 'config/task=default_3year_gn' 'config/dataset=bejing_air_pm25' 'config.experiment_data.experiment_name=spic_regRidge_air25full' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'
```
#### NexCP
```sh
python ./code/main.py 'config/model_fc=reg_ridge' 'config/model_uc=nextcp' 'config/task=default_3year_gn' 'config/dataset=bejing_air_pm25' 'config.experiment_data.experiment_name=nexcp_ridge_air25full' 'config.model_uc.rho=0.995' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'
```
#### Standard CP
```sh
python ./code/main.py 'config/model_fc=reg_ridge' 'config/model_uc=conf_default' 'config/task=default_3year_gn' 'config/dataset=bejing_air_pm25' 'config.experiment_data.experiment_name=confDefault_ridge_air25full' 'config.experiment_data.base_proj_dir=<absolute_project_root_dir>'
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


### Runs SPCI Ablations
For running with a Window Size 25 set additional command (Appendix A1)
```
'config.model_uc.past_window_len=25'
```

For running with retrain set additional command (Appendix A2), set model_uc spec (`config/model_uc=`) to
```
'config/model_uc=spic_retrain'
```

