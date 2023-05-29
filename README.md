## Introduction

This repository contains my unofficial reimplementation of the standard [ECAPA-TDNN](https://arxiv.org/pdf/2005.07143.pdf), which is the speaker recognition in VoxCeleb2 dataset.

This repository is modified based on [voxceleb_trainer](https://github.com/clovaai/voxceleb_trainer).

## Best Performance in this project (with AS-norm)

| Dataset |  Vox1_O  |  Vox1_E  |  Vox1_H  |
| ------- |  ------  |  ------  |  ------  |
|  EER    |   0.86   |  1.18    |  2.17    |
|  minDCF |  0.0686  | 0.0765   |  0.1295  |

Notice, this result is in the Vox1_O clean list, for Vox1_O Noise list: EER is 1.00 and minDCF is 0.0713.
***

## System Description

I have uploaded the [system description](https://arxiv.org/pdf/2111.06671.pdf), please check the Session 3, `ECAPA-TDNN SYSTEM`.

### Dependencies

Note: That is the setting based on my device, you can modify the torch and torchaudio version based on your device.

Start from building the environment
```
conda create -n ECAPA python=3.7.9 anaconda
conda activate ECAPA
pip install -r requirements.txt
```

Start from the existing environment
```
pip install -r requirements.txt
```

### Data preparation

Please follow the official code to perpare your VoxCeleb2 dataset from the 'Data preparation' part in [this repository](https://github.com/clovaai/voxceleb_trainer).

Dataset for training usage: 

1) VoxCeleb2 training set;

2) MUSAN dataset;

3) RIR dataset.

Dataset for evaluation: 

1) VoxCeleb1 test set for [Vox1_O](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/veri_test2.txt) 

2) VoxCeleb1 train set for [Vox1_E](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/list_test_all2.txt) and [Vox1_H](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/list_test_hard2.txt) (Optional)

### Training

Then you can change the data path in the `trainECAPAModel.py`. Train ECAPA-TDNN model end-to-end by using:

```
python trainECAPAModel.py --save_path exps/exp1 
```

Every `test_step` epoches, system will be evaluated in Vox1_O set and print the EER. 

The result will be saved in `exps/exp1/score.txt`. The model will saved in `exps/exp1/model`

In my case, I trained 80 epoches in one 3090 GPU. Each epoch takes 37 mins, the total training time is about 48 hours.

### Pretrained model

Our pretrained model performs `EER: 0.96` in Vox1_O set without AS-norm, you can check it by using: 

```
python trainECAPAModel.py --eval --initial_model exps/pretrain.model
```

With AS-norm, this system performs `EER: 0.86`. We will not update this code recently since no enough time for this work. I suggest you the following paper if you want to add AS-norm or other norm methods:

```
Matejka, Pavel, et al. "Analysis of Score Normalization in Multilingual Speaker Recognition." INTERSPEECH. 2017.
```

We also update the score.txt file in `exps/pretrain_score.txt`, it contains the training loss, training acc and EER in Vox1_O in each epoch for your reference.

***


### Reference

Original ECAPA-TDNN paper
```
@inproceedings{desplanques2020ecapa,
  title={{ECAPA-TDNN: Emphasized Channel Attention, propagation and aggregation in TDNN based speaker verification}},
  author={Desplanques, Brecht and Thienpondt, Jenthe and Demuynck, Kris},
  booktitle={Interspeech 2020},
  pages={3830--3834},
  year={2020}
}
```

Our reimplement report
```
@article{das2021hlt,
  title={HLT-NUS SUBMISSION FOR 2020 NIST Conversational Telephone Speech SRE},
  author={Das, Rohan Kumar and Tao, Ruijie and Li, Haizhou},
  journal={arXiv preprint arXiv:2111.06671},
  year={2021}
}
```

VoxCeleb_trainer paper
```
@inproceedings{chung2020in,
  title={In defence of metric learning for speaker recognition},
  author={Chung, Joon Son and Huh, Jaesung and Mun, Seongkyu and Lee, Minjae and Heo, Hee Soo and Choe, Soyeon and Ham, Chiheon and Jung, Sunghwan and Lee, Bong-Jin and Han, Icksang},
  booktitle={Interspeech},
  year={2020}
}
```

### Acknowledge

We study many useful projects in our codeing process, which includes:

[clovaai/voxceleb_trainer](https://github.com/clovaai/voxceleb_trainer).

[lawlict/ECAPA-TDNN](https://github.com/lawlict/ECAPA-TDNN/blob/master/ecapa_tdnn.py).

[speechbrain/speechbrain](https://github.com/speechbrain/speechbrain/blob/96077e9a1afff89d3f5ff47cab4bca0202770e4f/speechbrain/lobes/models/ECAPA_TDNN.py)

[ranchlai/speaker-verification](https://github.com/ranchlai/speaker-verification)

Thanks for these authors to open source their code!

### Notes

If you meet the problems about this repository, **Please ask me from the 'issue' part in Github (using English) instead of sending the messages to me from bilibili, so others can also benifit from it.** Thanks for your understanding!

If you improve the result based on this repository by some methods, please let me know. Thanks!

### Cooperation

If you are interested to work on this topic and have some ideas to implement, I am glad to collaborate and contribute with my experiences & knowlegde in this topic. Please contact me with ruijie.tao@u.nus.edu.
