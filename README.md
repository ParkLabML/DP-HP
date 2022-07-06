# DP-HP

Code for Hermite Polynomial Features for Private Data Generation (published at ICML2022)

### Dependencies
Versions numbers are based on our system and may not need to be exact matches. 

    python 3.6
    torch 1.3.1              
    torchvision 0.4.2
    numpy 1.16.4
    scipy 1.3.1
    pandas 1.0.1
    scikit-learn 0.21.2
    matplotlib 3.1.0 (plotting)
    seaborn 0.10.0 (more plotting)
    sdgym 0.1.0 (handling tabular datasets)
    autodp 0.1 (privacy analysis)
    backpack-for-pytorch 1.0.1 (efficient DP-SGD for DP-MERF+AE)
    tensorboardX 1.7 (some logging)
    tensorflow-gpu 1.14.0 (DP-CGAN)


# Repository Structure


## Comparison between HP features ans RF features (Fig 1)

Execute `dp_mehp/error_comaprison_multiple_runs.py`.

## 2D data (Fig 2)

Run `dp_mehp/dp_mehp_synth_data_2d.py`

## Experiments on image data

To run DP-HP experiments, use the following commands:

1. Obtain .npz files needed: run `extract_numpy_data_mats()` function in `dp_mehp/aux.py`

2. Creating the generated samples and predictive models evaluation

### MNIST

- ` python3 prod_and_sum_kernel_image.py --log-name *experiment name* --data digits -bs 200  --seed 0 --model-name FC -ep 10  -lr 0.01 --order-hermite-sum 100 --order-hermite-prod 20 --kernel-length-sum 0.005 --kernel-length-prod 0.005 --gamma 5 --prod-dimension 2` for the non-private case

- ` python3 prod_and_sum_kernel_image.py --log-name *experiment name* --data digits -bs 200  --seed 0  --ep 10 --lr 0.01 --order-hermite-sum 100 --order-hermite-prod 20 --model-name FC --kernel-length-sum 0.005 --kernel-length-prod 0.005 --gamma 20 --prod-dimension 2 --split --split-sum-ratio 0.8 --is-private` for $(1, 10^{-5})$-DP case

### FashionMNIST

- ` python3 prod_and_sum_kernel_image.py--log-name *experiment name* --data fashion -bs 200  --seed 0 --model-name CNN -ep 10  -lr 0.01 --order-hermite-sum 100 --order-hermite-prod 20 --kernel-length-sum 0.15 --kernel-length-prod 0.15 --gamma 20 --prod-dimension 2` for the non-private case

- ` python3 prod_and_sum_kernel_image.py --log-name *experiment name* --data fashion -bs 200  --seed 0 --model-name CNN -ep 10  -lr 0.01 --order-hermite-sum 100 --order-hermite-prod 20 --kernel-length-sum 0.15 --kernel-length-prod 0.15 --gamma 10 --prod-dimension 2  --split --split-sum-ratio 0.8 --is-private` for $(1, 10^{-5})$-DP case


3. Repoducing Fig. 3 and Fig. 6

-Run `code_balanced/plot_results.py` that loads the results from different models from `code_balanced/plots/` folder.

## Experiments on tabular data

1. Results in table 1. are obtained with `dp_mehp/discretized_datasets.py`

2. Results in table 2. are obatained with `dp_mehp/run_sum_prod_kernel_tabular_data.py`

