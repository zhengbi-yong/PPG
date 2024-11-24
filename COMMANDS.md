# 常用命令
## 下载数据集
```zsh
python extracode/download_dataset/PPG_FieldStudy.py
```
## 可视化数据集
```zsh
python extracode/visualize/PPG_FieldStudy_pickle.py
```
## 训练

### WINDOWS+GPU

```bash
python src/train.py experiment=PPG_FieldStudy_Base callbacks=PPG_FieldStudy logger=tensorboard

python src/train.py experiment=PPG_FieldStudy_Base callbacks=PPG_FieldStudy logger=tensorboard trainer=gpu

python src/train.py experiment=PPG_FieldStudy_Base logger=tensorboard callbacks=PPG_FieldStudy trainer=gpu debug=fdr
```

### WINDOWS+CPU

```powershell
python src/train.py experiment=PPG_FieldStudy_Base callbacks=PPG_FieldStudy logger=tensorboard trainer.accelerator=cpu

python src/train.py experiment=PPG_FieldStudy_LSTM callbacks=PPG_FieldStudy logger=tensorboard trainer.accelerator=cpu
```

## 实验结果查看

### tensorboard

```bash
tensorboard --logdir=./logs/train/runs/
```

## 评估模型

```bash
python src/eval.py ckpt_path="D:\lightning-hydra-template\logs\train\runs\2024-10-06_11-35-20\checkpoints\epoch_005.ckpt"
```
