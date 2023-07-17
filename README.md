# Diffusion model in ICU false alarm judgement

Code/Script for IJCAI 2023 paper A Diffusion Model with Contrastive Learning for ICU False Arrhythmia Alarm Reduction.


## Requirements

- Ubuntu 18.04
- pytorch 1.12.1+cu113


Dependencies can be installed by:

```
    pip install -r requirements.txt
```
## Dataset

Due to the requirements of the data providers, we are unable to publicly share our dataset.

## Usage
To train our model, you need modify the dataset path in config/config.json file. Then you can run the train_c.py.

```
python train_c.py
```



## Acknowledgments
We would like to thank [Diffwave](https://github.com/philsyn/DiffWave-Vocoder) and [SSSD](https://github.com/AI4HealthUOL/SSSD). our code is based on their implementations.




Coming soon
