# HopCPT - Conformal Prediction for Time Series with Modern Hopfield Networks
[![arXiv](https://img.shields.io/badge/arXiv-2306.14884-b31b1b.svg)](https://arxiv.org/abs/2303.12783)
[![Paper](https://img.shields.io/badge/Neurips2023-Paper-red.svg)](https://neurips.cc/virtual/2023/poster/72007)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Andreas Auer<sup>**1**</sup>, Martin Gauch<sup>**1,2**</sup>, Daniel Klotz<sup>**1**</sup>, Sepp Hochreiter<sup>**1**</sup> 

<sup>**1**</sup>ELLIS Unit Linz and LIT AI Lab, Institute for Machine Learning, Johannes Kepler University Linz, Austria\
<sup>**2**</sup>Google Research, Linz, Austria\

This repository contains the source code for **"Conformal Prediction for Time Series with Modern Hopfield Networks"** accepted at the at Neurips 2023.
The paper is available [here](https://arxiv.org/abs/2303.12783). 

---

##### Blogpost comming soon - stay tuned!

---


![Overview HopCPT](./Figure1_Animated_higherRes.gif)


## Install Dependencies 
```
conda env update -n <your-enviroment> --file ./conformal-conda-env.yaml
pip install -r ./conformal-pip-requirements.txt
pip install neuralhydrology
```

## Reproduce Experiments

#### Neuips 2023 - "Conformal Prediction for Time Series with Modern Hopfield Networks"

To re-run the experiments of *Conformal Prediction for Time Series with Modern Hopfield Networks* see [experiments_neurips23.md](https://github.com/ml-jku/HopCPT/blob/master/experiments_neurips23.md).



## 📚 Cite
If you find this work helpful, please cite

```bibtex
@inproceedings{auer2023conformal,
    author={Auer, Andreas and Gauch, Martin and Klotz, Daniel and Hochreiter, Sepp},
    title={Conformal Prediction for Time Series with Modern Hopfield Networks},
    booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
    institution = {Institute for Machine Learning, Johannes Kepler University, Linz},
    year={2023},
    url={https://openreview.net/forum?id=KTRwpWCMsC}
}
```

## Keywords
Time Series, Uncertainty, Conformal Prediction, Machine Learning, Deep Learning,  

