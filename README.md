# DexGraspSyn: A differentiable grasp optimizer for universal grasp synthesis.

This repository contains the code for synthesizing power grasps, which mainly draws on our previous work DexFG [Learning Human-like Functional Grasping for Multi-finger Hands from Few Demonstrations](https://ieeexplore.ieee.org/abstract/document/10577462). 
[[project page]](https://v-wewei.github.io/sr_dexgrasp/)

## Introduction

This repository provides an extremely fast method for synthesizing universal (powerful) grasp based on  gradient-based optimization.

## Qualitative Results

According to our experiments, on an A100 GPU, DexGraspSyn can synthesize 10k grasps in 2-5 minutes, with the time difference mainly depending on the number of sampling points on the object. The speed is about 15 times faster than DexGraspNet, and can be used for quickly synthesizing large-scale multi-finger grasp datasets.

## Overview

This repository provides:

  Toolkit for processing MuJoCo xml file. See `xml_processing/write_obj_xml.py`.
  Differentiable Grasp optimizer based on Pytorch. See `graspsyn/hand_optimizer.py`. 
  Toolkit for differentiable Hand Kinematics Layer. See Leap_hand_layer (https://github.com/v-wewei/leap_hand_layer).
  - The gradient-based hand grasp optimizer is implemented with AdamW optimizer with following objectives.
    - Collision Avoidance. Hand-Object Collision and Self Collision based on point to point singed distance. 
    - Contact Encourage. Hand(Anchor)-object distance.  
    - Contact Normal Alignment. The alignment is calculated based on anchor normal and object surface normal.
    - Abnormal Joint Avoidance. Preventing abnormal finger overlap.
  Grasp optimization with obstacle avoidance, such as an object placed on top of the table.
  Parallel-Jaw gripper contact grasp to Multi-fingered hand grasp mapping with gradient. (This function will be enriched in September 2024.)

## Quick Example

```bash
conda create -n diffgraspsyn python=3.8
conda activate diffgraspsyn
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

pip install -r requirements.txt


cd hand_layers/
git clone https://github.com/DexGraspOpt/leap_hand_layer

cd ..
# for quick example using our test data
python syn_unigrasp.py
```

## Citation

If you find this code useful in your research, please consider citing the following papers:

```
[1]@article{wei2024learning,
  title={Learning Human-like Functional Grasping for Multi-finger Hands from Few Demonstrations},
  author={Wei, Wei and Wang, Peng and Wang, Sizhe and Luo, Yongkang and Li, Wanyi and Li, Daheng and Huang, Yayu and Duan, Haonan},
  journal={IEEE Transactions on Robotics},
  year={2024},
  publisher={IEEE}
}

[2]@article{wei2022dvgg,
  title={DVGG: Deep variational grasp generation for dextrous manipulation},
  author={Wei, Wei and Li, Daheng and Wang, Peng and Li, Yiming and Li, Wanyi and Luo, Yongkang and Zhong, Jun},
  journal={IEEE Robotics and Automation Letters},
  volume={7},
  number={2},
  pages={1659--1666},
  year={2022},
  publisher={IEEE}
}

```

## Contact

If you have any questions, please do not hesitate to open an issue or contact me:

Email address: Wei Wei <weiwei72607260@gmail.com>.

## Acknowledge

The authors express sincere gratitude for the contribution and help of Sizhe Wang <sizhe_wang@163.com>.
