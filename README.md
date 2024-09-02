# cdata_utils

This repository contains the python code which as used to clean, preprocess and analyze clinical data of patients suffering from Porto-Sinusoidal Vascular Disorder (PSVD). The analysis of the PSVD data (related to reference ...) was done with [this version](https://github.com/ClemensWatzenboeck/cdata_utils/commit/0986fc8d49c3ac1e1aa7734ec807dcea48c2877c) of the code.



**Note:** Since the patient data is confidential, we show here only the scripts and jupyter notebooks which were used to analyze the data. The data itself is not available.


## Installation with conda:
The majority of the code is in the python package `cdata_utils`.
If you use a package mananger, like anaconda you might want to use the following commands.

```bash
# make new env: 
conda create --name py3-10-cdata_utils -c conda-forge python=3.10
conda activate py3-10-cdata_utils

# After that the other requirements should be handeled by the `pyproject.toml`
pip install -e . 

```


## More notes on the environemnt: 
We also provide a frozen environment, which can be used to recreate the environment *exactly* in the form we used to analyze the data.
See [./env/environment.yml](./env/environment.yml). 


It was created with: 
```bash 
conda env export -n py3-10-cdata_utils > ./env/environment.yml
```

And could be used to recreate the environment with

```bash
conda env create -f ./env/environment.yml
```





