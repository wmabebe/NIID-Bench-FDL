# NIID-Bench
This repository clonned and transformed the <a href="https://github.com/Xtra-Computing/NIID-Bench">NIID-Bench </a> repository.
Here is the original paper on Non-IID FL [Federated Learning on Non-IID Data Silos: An Experimental Study](https://arxiv.org/pdf/2102.02079.pdf).


This code implements a Fully Decentralized Learning graph using 3 types of non-IID settings (label distribution skew, feature distribution skew & quantity skew) and 4 datasets (MNIST, Cifar-10, FEMNIST, Cifar-100).


## Non-IID Settings
### Label Distribution Skew
* **Quantity-based label imbalance**: each party owns data samples of a fixed number of labels.
* **Distribution-based label imbalance**: each party is allocated a proportion of the samples of each label according to Dirichlet distribution.
### Feature Distribution Skew
* **Noise-based feature imbalance**: We first divide the whole datasetinto multiple parties randomly and equally. For each party, we adddifferent levels of Gaussian noises.
* **Synthetic feature imbalance**: For generated 3D data set, we allocate two parts which are symmetric of(0,0,0) to a subset for each party.
* **Real-world feature imbalance**: For FEMNIST, we divide and assign thewriters (and their characters) into each party randomly and equally.
### Quantity Skew
* While the data distribution may still be consistent amongthe parties, the size of local dataset varies according to Dirichlet distribution.



## Usage
Here is one example to run this code:
```
python fdl.py --model=simple-cnn \
    --dataset=mnist \
    --lr=0.01 \
    --batch-size=64 \
    --epochs=10 \
    --n_parties=128 \
    --mu=0.01 \
    --rho=0.9 \
    --comm_round=50 \
    --partition=noniid-labeldir \
    --beta=0.5 \
    --device='cuda:0' \
    --datadir='./data/' \
    --logdir='./logs/' \
    --noise=0 \
    --sample=1 \
    --init_seed=0 \
    --local_acc=0 \
    --topology=pcc \
    --strategy=optim \
    --similarity=optim \
```

| Parameter                      | Description                                 |
| ----------------------------- | ---------------------------------------- |
| `model` | The model architecture. Options: `simple-cnn`,`simple-ffnn`, `vgg`, `res18`, `mlp`. Default = `mlp`. |
| `dataset`      | Dataset to use. Options: `mnist`, `cifar10`, `femnist`, `cifar100`. Default = `mnist`. |
| `lr` | Learning rate for the local models, default = `0.01`. |
| `batch-size` | Batch size, default = `64`. |
| `epochs` | Number of local training epochs, default = `5`. |
| `n_parties` | Number of parties, default = `2`. |
| `mu` | The proximal term parameter for FedProx, default = `1`. |
| `rho` | The parameter controlling the momentum SGD, default = `0`. |
| `comm_round`    | Number of communication rounds to use, default = `50`. |
| `partition`    | The partition way. Options: `homo`, `noniid-labeldir`, `noniid-#label1` (or 2, 3, ..., which means the fixed number of labels each party owns), `real`, `iid-diff-quantity`. Default = `homo` |
| `beta` | The concentration parameter of the Dirichlet distribution for heterogeneous partition, default = `0.5`. |
| `device` | Specify the device to run the program, default = `cuda:0`. |
| `datadir` | The path of the dataset, default = `./data/`. |
| `logdir` | The path to store the logs, default = `./logs/`. |
| `noise` | Maximum variance of Gaussian noise we add to local party, default = `0`. |
| `sample` | Ratio of parties that participate in each communication round, default = `1`. |
| `init_seed` | The initial seed, default = `0`. |
| `local_acc` | Whether you want the local accuracies displayed at the end of local training, default = `1`. |
| `topology` | The DL topology. Options: `tree`, `ring`, `clique`, `pcc`, default = `tree`|
| `strategy` | Local arrangement of nodes, heterogeneous (optim) or homogeneous (rand). Options: `optim`, `rand` |
| `similarity` | Similarity function. Options `kl`, `grad`, default = `kl` |
| `cut` | The number of partitions to create. Default = `0` i.e. No partitions |
| `pre_epochs` | The number of local pre train (prologue) epochs. Default = `10` |




## Data Partition Map
You can call function `get_partition_dict()` in `experiments.py` to access `net_dataidx_map`. `net_dataidx_map` is a dictionary. Its keys are party ID, and the value of each key is a list containing index of data assigned to this party. For our experiments, we usually set `init_seed=0`. When we repeat experiments of some setting, we change `init_seed` to 1 or 2. The default value of `noise` is 0 unless stated. We list the way to get our data partition as follow.
* **Quantity-based label imbalance**: `partition`=`noniid-#label1`, `noniid-#label2` or `noniid-#label3`
* **Distribution-based label imbalance**: `partition`=`noniid-labeldir`, `beta`=`0.5` or `0.1`
* **Noise-based feature imbalance**: `partition`=`homo`, `noise`=`0.1` (actually noise does not affect `net_dataidx_map`)
* **Synthetic feature imbalance & Real-world feature imbalance**: `partition`=`real`
* **Quantity Skew**: `partition`=`iid-diff-quantity`, `beta`=`0.5` or `0.1`
* **IID Setting**: `partition`=`homo`
* **Mixed skew**: `partition` = `mixed` for mixture of distribution-based label imbalance and quantity skew; `partition` = `noniid-labeldir` and `noise` = `0.1` for mixture of distribution-based label imbalance and noise-based feature imbalance.

Here is explanation of parameter for function `get_partition_dict()`. 

| Parameter                      | Description                                 |
| ----------------------------- | ---------------------------------------- |
| `dataset`      | Dataset to use. Options: `mnist`, `cifar10`, `femnist`, `cifar-100`. |
| `partition`    | Tha partition way. Options: `homo`, `noniid-labeldir`, `noniid-#label1` (or 2, 3, ..., which means the fixed number of labels each party owns), `real`, `iid-diff-quantity` |
| `n_parties` | Number of parties. |
| `init_seed` | The initial seed. |
| `datadir` | The path of the dataset. |
| `logdir` | The path to store the logs. |
| `beta` | The concentration parameter of the Dirichlet distribution for heterogeneous partition. |

## Experiments


<p><img src="figures/plots.pdf" width="800" height="600" /><p>

Blue lines show experiments run with `strategy=optim` to create locally hetergeneous DL graph. Corresponding yellow lines are experiments
run wth `strategy=rand` to create locally homogeneous DL graph. Furthermore, the table below show nodes in the form `m/n`. When running the
experiemnts, the `n_parties` argument is set to the `n` value. While the `m` value is automatically set. This happens when setting the `topology`=pcc.


| Experiment    |   Dataset      |  Partition |     Non-IID                   |    Model     |   Nodes        | Delta Acc.  |
| --------------|--------------- | ---------- |------------------------------ | ------------ | -------------- | ------------|
| A             | `CIFAR-10`     | `cut`= 0   | #label2                       | `ResNet-18`  |   18/20        | +6.28%      |
| B             |  >>            |  >>        | labeldir `beta`=0.5           |  >>          |   15/20        | -1.98%      |
| C             |  >>            |  >>        | feat-noise `noise`=0.2        |  >>          |   20/24        | +1.13%      |
| D             |  >>            |  >>        | mixed `beta`=0.5,`noise`=0.3  |  >>          |   20/24        | +1.59%      |
| E             | `MNIST`        |  >>        | #label2                       | `simple-cnn` |   22/24        | +25.9%      |
| F             |  >>            | `cut`= 2   | labeldir `beta`=0.5           |  >>          |   22/24        | +29.6%      |
| G             |  >>            | `cut`= 3   | feat-noise `noise`=0.2        |  >>          |   24/24        | +25.8%      |
| H             |  >>            | `cut`= 4   | mixed `beta`=0.5,`noise`=0.3  |  >>          |   24/24        | +22.6%      |
| I             | `FEMNIST`      | `cut`= 0   | feat-noise `noise`=0.2        | `simple-ffnn`|   122/1200     | +12.4%      |
| J             |  >>            | >>         | mixed `beta`=0.5,`noise`=0.3  |  >>          |   122/1200     | +16.0%      |
| K             | `CIFAR-100`    | >>         | feat-noise `noise`=0.2        | `ResNet-18`  |   22/24        | +1.31%      |
| L             |  >>            | >>         | mixed `beta`=0.5,`noise`=0.3  |  >>          |   22/24        | +2.86%      |





## Citation
To cite the original paper
```
@inproceedings{li2022federated,
      title={Federated Learning on Non-IID Data Silos: An Experimental Study},
      author={Li, Qinbin and Diao, Yiqun and Chen, Quan and He, Bingsheng},
      booktitle={IEEE International Conference on Data Engineering},
      year={2022}
}
```
