# ICICLE: Interpretable Class Incremental Continual Learning
### D. Rymarczyk, J. van de Weijer, B. Zieliński, B. Twardowski
This repository contains the code for the paper [ICICLE: Interpretable Class Incremental Continual Learning](https://arxiv.org/abs/2303.07811). The paper was accepted at ICCV 2023.

### Requirements

Required packages:

```
- python>=3.9
- pytorch>=1.7
- torchvision>=0.8.2
- opencv-python==4.5.3.56
- matplotlib
- numpy
- tensorboard
```

This codebase is built on FACIL framework https://github.com/mmasana/FACIL
The prototypical parts-based models are adapted from the following repositories:

- ProtoPNet: https://github.com/cfchen-duke/ProtoPNet
- TesNet: https://github.com/JackeyWang96/TesNet

### Data preparation

Please use data preparation method from ProtoPNet repository https://github.com/cfchen-duke/ProtoPNet
In the next step prepare text file describing the data as in `data_utils`
Lastly, change path in the file `src/datasets/dataset_config.py`

### Running the code 

To run ICICLE and obtain main results use the following command:
```
python src/main_incremental.py --results-path [path] --exp-name [exp_name] --save-models --dataset cub_200_2011_cropped --batch-size 80 --num-tasks 4 --network {protopnet,tesnet}_resnet34 --approach icicle --eval-on-train --pretrained --gpu 0 --proto_num_per_class 10 --proto_depth 256 --num_classes 50 --nepochs 21 --lr 0.001 --num-workers 6 --weight-decay 1e-3 --push_at 10 --num_warm 5 --num_push_tune 0 --ppnet_eval --save-models --ppnet_sim log --sep_weight -0.08 --incorrect_weight 0 --lamb 0.01 --similarity_reg --perc 3 --lr_old 1e-3
```

For the other approaches, you can use:
- EWC: `python src/main_incremental.py --results-path [path] --exp-name [exp_name] --dataset cub_200_2011_cropped --batch-size 80 --num-tasks 4 --network {protopnet,tesnet}_resnet34 --approach ewc_protopnet --eval-on-train --pretrained --gpu 0 --proto_num_per_class 10 --proto_depth 256 --num_classes 50 --nepochs 21 --lr 0.001 --num-workers 6 --weight-decay 1e-3 --push_at 10 --num_warm 5 --num_push_tune 0 --ppnet_eval --ppnet_sim log --sep_weight -0.08 --incorrect_weight 0 --seed 0`
- LWF: `python main_incremental.py --results-path [path] --exp-name [exp_name] --dataset cub_200_2011_cropped --batch-size 80 --num-tasks 4 --network {protopnet,tesnet}_resnet34 --approach lwf_protopnet --eval-on-train --pretrained --gpu 0 --proto_num_per_class 10 --proto_depth 256 --num_classes 50 --nepochs 21 --lr 0.001 --num-workers 6 --weight-decay 1e-3 --push_at 10 --num_warm 5 --num_push_tune 0 --ppnet_eval --ppnet_sim log --sep_weight -0.08 --incorrect_weight 0 --seed 0 `
- LWM: `python main_incremental.py --results-path [path] --exp-name [exp_name] --dataset cub_200_2011_cropped --batch-size 80 --num-tasks 4 --network {protopnet,tesnet}_resnet34 --approach lwm_protopnet --eval-on-train --pretrained --gpu 0 --proto_num_per_class 10 --proto_depth 256 --num_classes 50 --nepochs 21 --lr 0.001 --num-workers 6 --weight-decay 1e-3 --push_at 10 --num_warm 5 --num_push_tune 0 --ppnet_eval --ppnet_sim log --sep_weight -0.08 --incorrect_weight 0 --seed 0 --gamma 0.001`
- Finetuning: `python main_incremental.py --results-path [path] --exp-name [exp_name] --dataset cub_200_2011_cropped --batch-size 80 --num-tasks 4 --network {protopnet,tesnet}_resnet34 --approach finetuning_protopnet --eval-on-train --pretrained --gpu 0 --proto_num_per_class 10 --proto_depth 256 --num_classes 50 --nepochs 21 --lr 0.001 --num-workers 6 --weight-decay 1e-3 --push_at 10 --num_warm 5 --num_push_tune 0 --ppnet_eval --ppnet_sim log --sep_weight -0.08 --incorrect_weight 0 --seed 0`
- Freezing: `python main_incremental.py --results-path [path] --exp-name [exp_name] --dataset cub_200_2011_cropped_gmum --batch-size 80 --num-tasks 4 --network {protopnet,tesnet}_resnet34 --approach freezing_protopnet --eval-on-train --pretrained --gpu 0 --proto_num_per_class 10 --proto_depth 64 --num_classes 20 --nepochs 15 --lr 0.001 --num-workers 6 --weight-decay 1e-3 --push_at 7 --num_warm 5 --num_push_tune 0 --ppnet_eval --ppnet_sim log --sep_weight -0.08 --incorrect_weight 0 --seed 0  --fix-bn`

The non-interpretable baselines as PASS and FeTrIL were obtained using code from repo: PyCIL https://github.com/G-U-N/PyCIL

To perform task-recency bias compensation after training use notebook: `balance_tasks.ipynb`

## Citation

If you find this work useful, please cite our paper:
```bibtex
@article{rymarczyk2023icicle,
  title={ICICLE: Interpretable Class Incremental Continual Learning},
  author={Rymarczyk, Dawid and van de Weijer, Joost and Zieli{\'n}ski, Bartosz and Twardowski, Bart{\l}omiej},
  journal={Intenational Conference on Computer Vision [ICCV]},
  year={2023}
}

```