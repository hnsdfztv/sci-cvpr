# SCI(CVPR 2022) CPU 复现与代码导读（完成版，已实测）

> 适用目录：`SCI/CVPR`
>
> 本文档是“可直接执行”的完成版：包含已完成改动、已跑通命令、产物位置、当前项目结构和下一步建议。

---

## 0. 目前状态（结论先看）

你这个项目已经在 **CPU** 上完成了以下三类验证：

1. `test.py`：预训练模型推理 ✅
2. `finetune.py`：小步数微调冒烟 ✅
3. `train.py`：1 epoch 训练冒烟 ✅

并且已经给环境起了短名：`sci-cvpr`（不再用超长路径环境名）。

---

## 1. 论文在做什么（超简洁复述）

SCI 的核心是：
- 训练时用“**多阶段 + 自校准**”让 illumination 学得更稳；
- 测试时只保留轻量增强块，所以推理快。

Retinex 关系：

$$
\mathbf{y} = \mathbf{z} \otimes \mathbf{x}
$$

代码里对应：先估计 illumination `i`，再做 `r = input / i` 得增强结果。

---

## 2. 已做代码改动（你问的“做了哪些改动”）

以下改动都在 `SCI/CVPR` 下完成：

### 2.1 `test.py`（CPU 推理）
- 新增 `--device`（默认 `cpu`）
- 删除“无 GPU 直接退出”
- `.cuda()` 改为 `.to(device)`
- 推理权重加载改为 `Finetunemodel(..., map_location=device)`

### 2.2 `model.py`
- `Finetunemodel.__init__` 增加 `map_location=None`
- `torch.load(weights)` 改为 `torch.load(weights, map_location=map_location)`

### 2.3 `finetune.py`（CPU 微调）
- 新增 `--device`（默认 `cpu`）
- 删除 GPU 强制退出逻辑
- `.cuda()` 改为 `.to(device)`
- `pin_memory` 按设备动态设置
- `--steps` 从 `float` 改为 `int`（避免 `range(float)` 报错）

### 2.4 `loss.py`
- `SmoothLoss` 里原本写死 `.cuda()` 的常量 tensor 改为：
  - `device=input_im.device`
  - `dtype=im_flat.dtype`
- 这样 CPU/CUDA 都能跑

### 2.5 `train.py`（CPU 训练）
- 新增 `--device`（默认 `cpu`）
- 新增 `--train_data_path`、`--test_data_path`
- 删除 GPU 强制退出逻辑
- `.cuda()` 改为 `.to(device)`
- `pin_memory` 按设备动态设置
- 去掉 `Variable(..., volatile=True)` 旧写法

---

## 3. 已做复现（你问的“复现了哪些、产出什么”）

## 3.1 推理复现（CPU）
执行过：

```powershell
D:/Software/miniconda3/python.exe test.py --device cpu --data_path ./data/medium --save_path ./results/medium_cpu --model ./weights/medium.pt
```

结果目录：`CVPR/results/medium_cpu`

生成文件（10 张）：
- `00001.png`
- `00051.png`
- `00079.png`
- `00091.png`
- `2062.png`
- `2064.png`
- `3008.png`
- `3018.png`
- `3020.png`
- `NPE_71.png`

## 3.2 微调冒烟（CPU）
执行过：

```powershell
D:/Software/miniconda3/python.exe finetune.py --device cpu --steps 10 --save ./results/finetune_cpu_smoke --model ./weights/difficult.pt
```

结果目录：`CVPR/results/finetune_cpu_smoke`

生成文件：
- `LIME_9_10_ref_.png`

## 3.3 训练冒烟（CPU，1 epoch）
执行过：

```powershell
D:/Software/miniconda3/python.exe train.py --device cpu --epochs 1 --batch_size 1 --train_data_path ./data/medium --test_data_path ./data/medium --save EXP_CPU_SMOKE
```

结果目录：`CVPR/EXP_CPU_SMOKE/Train-20260327-193825`（时间戳目录会变）

关键产物：
- 权重：`model_epochs/weights_0.pt`
- 可视化：`image_epochs/*.png`（10 张）
- 日志：`log.txt`

---

## 4. 当前项目结构（你问的“项目结构讲解”）

你的工作区主要是：

- `SCI/`（仓库根）
  - `CVPR/`（当前主战场）
    - `data/`：输入图片目录（`easy`/`medium`/`difficult`/`finetune`）
    - `weights/`：预训练权重（`easy.pt`/`medium.pt`/`difficult.pt`）
    - `results/`：推理和微调输出
      - `medium_cpu/`：CPU 推理结果
      - `finetune_cpu_smoke/`：CPU 微调冒烟结果
    - `EXP_CPU_SMOKE/`：CPU 训练冒烟实验目录
    - `test.py`：推理入口
    - `finetune.py`：微调入口
    - `train.py`：训练入口
    - `model.py`：网络结构
    - `loss.py`：无监督损失
    - `multi_read_data.py`：数据读取
    - `CVPR_CPU_复现与代码导读_小白版.md`：早期指南
    - `CVPR_CPU_复现与代码导读_完成版.md`：本文件
  - `TPAMI/`：下一阶段可再看（你现在先不读）

---

## 5. conda 环境（已命名）

你原来的环境路径很长：
- `d:\Zi_Liao\LEARING\DL\reproduce\dut\SCI\.conda`

已经创建短名环境：
- `sci-cvpr`

并已验证关键依赖可用：
- `torch 2.11.0+cpu`
- `torchvision 0.26.0+cpu`
- `PIL 12.1.1`
- `numpy 2.4.3`

使用方式：

```powershell
conda activate sci-cvpr
cd d:\Zi_Liao\LEARING\DL\reproduce\dut\SCI\SCI\CVPR
```

---

## 6. 一键可复制命令（建议你就按这个跑）

## 6.1 CPU 推理

```powershell
conda activate sci-cvpr
cd d:\Zi_Liao\LEARING\DL\reproduce\dut\SCI\SCI\CVPR
python test.py --device cpu --data_path ./data/medium --save_path ./results/medium_cpu --model ./weights/medium.pt
```

## 6.2 CPU 微调（小步）

```powershell
conda activate sci-cvpr
cd d:\Zi_Liao\LEARING\DL\reproduce\dut\SCI\SCI\CVPR
python finetune.py --device cpu --steps 10 --save ./results/finetune_cpu_smoke --model ./weights/difficult.pt
```

## 6.3 CPU 训练（流程验证）

```powershell
conda activate sci-cvpr
cd d:\Zi_Liao\LEARING\DL\reproduce\dut\SCI\SCI\CVPR
python train.py --device cpu --epochs 1 --batch_size 1 --train_data_path ./data/medium --test_data_path ./data/medium --save EXP_CPU_SMOKE
```

---

## 7. 常见提示与说明

1. `torch.set_default_tensor_type` 的警告：
   - 是 PyTorch 2.x 的弃用提醒，不影响当前复现跑通。
2. CPU 很慢：
   - 当前阶段你只需要“流程可跑 + 代码理解”，不是追求速度。
3. 训练目录会带时间戳：
   - 属于正常行为，方便分实验管理。

---

## 8. 你下一步最优路线（给 0 基础）

1. 先固定只用 `sci-cvpr` 环境，避免混用。
2. 用你自己的 5~20 张低光图跑 `test.py`，看三套权重差异。
3. 对照 `model.py` 与 `loss.py`，逐行读你不懂的部分（每次 20 行以内）。
4. 等你觉得流程清晰了，再开始碰 TPAMI 版本。

---

如果你愿意，下一步我可以再给你一份“**答辩版讲稿**”：
- 3 分钟讲清论文
- 5 分钟讲清代码
- 2 分钟讲清你复现做了什么
（适合你和学长汇报用）
