# Predicting Viral Rumors and Vulnerable Users with Graph-based Neural Multi-task Learning for Infodemic Surveillance (IP&M 2023)

Official implementation of paper "[Predicting Viral Rumors and Vulnerable Users with Graph-based Neural Multi-task Learning for Infodemic Surveillance](https://www.sciencedirect.com/science/article/pii/S0306457323002571)".

## Introduction

1. A unified framework for rumor detection, virality & user vulnerability scoring.
2. The three tasks are learned jointly via inductive GNN and hierarchical graph pooling.
3. The framework is trained in two multi-task settings to handle training conflicts.
   
![Overview of the proposed multi-task model.](https://github.com/jadeCurl/Predicting-Viral-Rumors-and-Vulnerable-Users/blob/main/pics/model.png)

## Datasets

For our dataset, we have built upon two publicly available datasets: [TWITTER](https://aclanthology.org/P17-1066/) and [WEIBO](https://dl.acm.org/doi/10.5555/3061053.3061153). These original datasets were initially created for rumor detection, featuring annotations that categorize data at the graph level as either rumor or non-rumor. To adapt these datasets for our specific research needs, we have further processed them to derive two additional labels: 'virality' and 'vulnerability'. The reconstructed datasets can be found in the [datasets](https://github.com/jadeCurl/Predicting-Viral-Rumors-and-Vulnerable-Users/tree/main/datasets) folder.

## Models

### Training and Inferring
>python run.py --time 0.8
   
TBD

## Citation

If you find this paper helpful or intriguing and decide to use it, kindly acknowledge the paper by citing it and consider starring this repo, thanks!
```bibtex
@article{zhang2024predicting,
  title={Predicting viral rumors and vulnerable users with graph-based neural multi-task learning for infodemic surveillance},
  author={Zhang, Xuan and Gao, Wei},
  journal={Information Processing \& Management},
  volume={61},
  number={1},
  pages={103520},
  year={2024},
  publisher={Elsevier}
}
