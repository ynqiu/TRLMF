# Imbalanced Low-Rank Tensor Completion via Latent Matrix Factorization

This paper proposes an imbalanced low-rank tensor completion method using latent tensor ring components and proximal alternating minimization, achieving better results with less computational cost.

## Main Functions

- `src/trfold.m`: Implements the tensor ring folding method.
- `src/trunfold.m`: Implements the tensor ring unfolding method.
- `src/TRLMF_PAM.m`: Implements the imbalanced low-rank tensor completion method based on the proximal alternating minimization algorithm.

## Usage

An example is provided in the `test_TRLMF_color_image.m` file, demonstrating how to use the above functions for tensor completion. Running this file will show a comparison of the original image, the observed image, and the recovered image.

## Citation

If you use this code in your research, please cite the following paper:
```
@article{qiu2022imbalanced, 
title={Imbalanced low-rank tensor completion via latent matrix factorization}, 
author={Qiu, Yuning and Zhou, Guoxu and Zeng, Junhua and Zhao, Qibin and Xie, Shengli},
journal={Neural Networks}, 
volume={155}, 
pages={369--382}, 
year={2022}, 
publisher={Elsevier}
}
```
