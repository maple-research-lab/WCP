# WCP
Source code for "WCP: Worst-Case Perturbations for Semi-Supervised Deep Learning" in CPVR 2020.

## Abstract
In this paper, we present a novel regularization mechanism for training deep networks by minimizing the Worse-Case Perturbation (WCP). It is based on the idea that a robust model is least likely to be affected by small perturbations, such that its output decisions should be as stable as possible on both labeled and unlabeled examples. We will consider two forms of WCP regularizations -- additive and DropConnect perturbations, which impose additive noises on network weights, and make structural changes by dropping the network connections, respectively. We will show that the worse cases of both perturbations can be derived by solving respective optimization problems with spectral methods. The WCP can be minimized on both labeled and unlabeled data so that networks can be trained in a semi-supervised fashion.  This leads to a novel paradigm of semi-supervised classifiers by stabilizing the predicted outputs in presence of the worse-case perturbations imposed on the network weights and structures.

## Motivation
| ![Toy example](https://github.com/maple-research-lab/WCP/blob/master/resource/aWCP_ex.png) |
|:--:| 
| *(a) toy example of a sigmoid unit for four datapoints* |
| ![Relation](https://github.com/maple-research-lab/WCP/blob/master/resource/aWCP_relation.png) |
| *(b) The relation between φ and the correspondingWCP value.  * |
| *Figure 1. From the “teeth” curve, the minimum WCP regularizer occurs at φ = 0(similary at φ=±π) with the corresponding boundary x1= 0. Two local minima of the WCP regularizer occur at ±π/2, corresponding to x2 = 0.* |

The idea of minimizing the impact of worst-case model perturbation is well connected with the large margin principle, since a maximum-margin classi-fier is least likely to change its predictions whenit is maximally perturbed. In Figure 1, we  use  a  toy  example  to  show  the  insightinto how the WCP regularizes the training of deep net-works. For details, please refer to our paper.

## Run our codes
### Requirements
- Python == 2.7
- Chainer == 5.1.0

### CIFAR10
    cd dataset
    python cifar10.py
    cd ..
    cd WCP
    python train_semisup.py --data_dir=../dataset/cifar10/ --log_dir=./log/cifar10_seed1/ --num_epochs=1000 --epoch_decay_start=800 --aug_flip --aug_trans --epsilon=8.0 --gpu 0 --dataset_seed 1
    
### SVHN
    cd dataset
    python svhn.py
    cd ..
    cd WCP
    python train_semisup.py --data_dir=../dataset/svhn/ --log_dir=./log/svhn_seed1/ --num_epochs=500 --epoch_decay_start=400 --epsilon=3.5 --aug_trans --top_bn --gpu 0 --dataset_seed 1
    
## Citation

Liheng Zhang, Guo-Jun Qi. WCP: Worst-Case Perturbations for Semi-Supervised Deep Learning in Proceedings of IEEE Conference on Computer Vision and Pattern Recognition (CVPR 2020), Seattle, WA, June 16th - June 20th, 2020. 
    
## Disclaimer

Some of our codes reuse the github project [vat_chainer](https://github.com/takerum/vat_chainer).  

## License

This code is released under the MIT License.


