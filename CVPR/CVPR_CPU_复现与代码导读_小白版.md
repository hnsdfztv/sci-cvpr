# SCI(CVPR 2022) 小白友好复现指南（仅 CVPR，CPU 版）

> 适用仓库：`SCI/CVPR`
> 
> 目标：先搞懂论文在做什么，再在 **CPU** 上跑通推理，并具备后续微调/训练的操作框架。

---

## 0. 你现在要达成的 3 个目标

1. **看懂论文核心想法**：SCI 为什么快、为什么鲁棒。
2. **先跑通推理**：用仓库提供的 `easy.pt / medium.pt / difficult.pt` 在 CPU 生成增强结果。
3. **读懂代码职责**：知道每个 `.py` 文件在做什么，后续能让 AI/Agent 帮你改代码。

---

## 1. 这篇论文到底在做什么？（一句话 + 细化）

### 1.1 一句话

这篇论文提出了一个非常轻量的低照度增强方法 SCI：
- 训练时用“多阶段 + 自校准模块”把模型学稳；
- 测试时只用一个很小的增强模块，所以 **速度非常快**、参数量很小。

### 1.2 任务建模（Retinex 视角）

论文使用经典关系：

$$
\mathbf{y} = \mathbf{z} \otimes \mathbf{x}
$$

- $\mathbf{y}$：低光输入图像
- $\mathbf{z}$：理想反射/清晰结果
- $\mathbf{x}$：光照分量（illumination）

做法是先估计光照，再通过除法恢复结果（代码里就是 `r = input / i`）。

### 1.3 SCI 的核心创新（你最该记住）

1. **多阶段、参数共享**：
   同一个增强块反复用在多个 stage（训练时）。
2. **自校准模块（calibrate）**：
   每个 stage 额外预测一个校正项，让后续 stage 输入更合理、结果更容易收敛。
3. **训练-推理解耦**：
   训练复杂一点（多 stage + calibrate），推理时只保留增强块（单 stage），因此很快。
4. **无监督损失**：
   不强依赖成对标注，使用保真项 + 平滑项约束光照图。

### 1.4 为啥它“快、灵活、鲁棒”

- **快**：测试只走 `EnhanceNetwork`（几层卷积）。
- **灵活**：训练框架可迁移到其他 illumination-based 方法。
- **鲁棒**：自校准让多阶段输出趋同，降低过曝/不稳定风险。

---

## 2. 仓库代码地图（按你当前 `CVPR` 目录）

- `model.py`
  - `EnhanceNetwork`：估计 illumination（核心推理模块）
  - `CalibrateNetwork`：训练阶段做自校准
  - `Network`：训练总网络（stage 循环）
  - `Finetunemodel`：加载预训练权重，推理/微调常用
- `loss.py`
  - `LossFunction = 1.5 * Fidelity + Smooth`
  - `SmoothLoss`：基于颜色相似度加权的空间平滑
- `multi_read_data.py`
  - `MemoryFriendlyLoader`：读取目录下图片并转 tensor
- `train.py`
  - 训练主入口（当前代码默认强依赖 CUDA）
- `test.py`
  - 推理入口（当前代码默认强依赖 CUDA）
- `finetune.py`
  - 小数据微调（当前代码默认强依赖 CUDA）
- `weights/`
  - `easy.pt` / `medium.pt` / `difficult.pt` 预训练模型
- `data/`
  - `easy`、`medium`、`difficult`、`finetune` 示例数据目录

---

## 3. 先说清楚“复现”分层（非常重要）

### Level A（推荐先做）

**CPU 跑通预训练模型推理**，拿到增强图。这是你当前最优先目标。

### Level B

在你自己的低光图片上做微调（CPU 可做但会慢）。

### Level C（论文完整训练）

用 MIT/LSRW 等数据从头训练并做完整指标复现。**CPU 上非常慢**，通常只做流程验证。

> 对 0 基础，建议路线：A -> 读代码 -> B，最后再碰 C。

---

## 4. CPU 复现实战（一步一步）

## 4.1 环境准备（Windows PowerShell）

在 `SCI/CVPR` 目录执行：

```powershell
cd d:\Zi_Liao\LEARING\DL\reproduce\dut\SCI\SCI\CVPR
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install numpy pillow
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

> 说明：
> - 仓库 README 写的是 `torch==1.8.0 + cuda`，但你现在目标是 CPU 跑通，可先用 CPU 版 torch。
> - 如果后续出现 API 兼容问题，再让 AI/Agent 定点修补（下面有模板）。

## 4.2 准备测试图片

- 把低光图片放到 `data/medium`（仓库已有示例图）。
- 输出目录建议用 `results/medium_cpu`。

## 4.3 关键现实：原始 `test.py` 默认只走 CUDA

仓库当前 `test.py` 有以下限制：
- `if not torch.cuda.is_available(): sys.exit(1)`
- 模型和输入都强制 `.cuda()`

所以 **CPU 不能直接跑**，你需要做“CPU 兼容改造”。

---

## 5. CPU 兼容改造清单（建议让 AI/Agent 自动改）

下面是最小必要改动（优先推理）：

1. `test.py`
   - 增加 `--device` 参数（默认 `cpu`）
   - 使用 `device = torch.device(args.device)`
   - `model.to(device)`，`input.to(device)`
   - 删除/绕过 `no gpu device available` 强退出
2. `model.py`
   - `Finetunemodel` 里 `torch.load(weights)` 改成 `torch.load(weights, map_location=device)` 逻辑
3. `loss.py`（训练/微调才需要）
   - 把 `.cuda()` 常量改成跟随输入 `device`
4. `train.py` / `finetune.py`（可后做）
   - 去掉无 GPU 时 `sys.exit(1)`
   - 所有 `.cuda()` 改 `.to(device)`

> 经验：先改 `test.py + model.py` 跑通推理，再考虑训练相关脚本。

---

## 6. 一次最小可用验证（你应看到什么）

成功标准：

1. 命令执行无报错；
2. 控制台打印 `processing xxx.png`；
3. 在 `results/medium_cpu` 看到输出图片；
4. 输出图相比输入图明显变亮，且尽量不过曝。

建议对比：
- 输入目录：`data/medium`
- 输出目录：`results/medium_cpu`
- 模型优先试：`weights/medium.pt`

---

## 7. 如何选择模型（`easy/medium/difficult`）

- `easy.pt`：轻度低光，提亮温和。
- `medium.pt`：通用场景优先试。
- `difficult.pt`：极暗场景更激进，可能更亮也更容易偏色。

实操建议：
- 同一张图分别跑 3 个模型，人工选观感最佳结果。

---

## 8. 每个代码文件“在算法链路中的位置”

按执行顺序理解（推理）：

1. `test.py` 解析参数 -> 组 DataLoader
2. `model.py` 的 `Finetunemodel` 加载权重
3. 前向计算：
   - `i = enhance(input)`（光照图）
   - `r = input / i`（增强结果）
4. `save_images` 保存 `r`

训练时（`train.py`）：

1. `Network(stage=3)` 进入循环 stage
2. 每 stage：`enhance -> reflectance -> calibrate -> 更新下一 stage 输入`
3. `loss.py` 计算无监督损失
4. 反向传播更新参数

---

## 9. 常见报错与定位

1. `no gpu device available`
   - 原因：脚本硬编码 GPU。
   - 处理：按第 5 节改成 `device` 分支。

2. `Attempting to deserialize object on a CUDA device...`
   - 原因：`torch.load` 载入 GPU 权重到 CPU。
   - 处理：`torch.load(..., map_location='cpu')`。

3. `Variable(..., volatile=True)` 相关告警/报错
   - 原因：旧版写法。
   - 处理：删 `volatile=True`，并在 `torch.no_grad()` 下推理。

4. 结果偏灰、过曝、颜色怪
   - 先换模型：`easy/medium/difficult`
   - 再检查输入是否归一化正确（当前 loader 是 `ToTensor`，范围 `[0,1]`）

---

## 10. 你可以直接复制给 AI/Agent 的任务模板

### 模板 A：先改到 CPU 可推理

```text
请在 SCI/CVPR 中做最小改动，实现 test.py 可在 CPU 推理：
1) 新增 --device 参数，默认 cpu；
2) 删除无 GPU 直接退出逻辑；
3) 所有 .cuda() 改为 .to(device)；
4) model.py 中 Finetunemodel 的 torch.load 支持 map_location；
5) 保持原有参数与输出行为不变；
6) 运行一次 data/medium -> results/medium_cpu 验证。
```

### 模板 B：继续支持 CPU 微调

```text
在已支持 CPU 推理的基础上，继续改造 finetune.py 与 loss.py：
1) 删除强制 CUDA 退出；
2) 常量 tensor 跟随输入 device；
3) 提供最小微调命令示例；
4) 只做必要改动，不重构。
```

### 模板 C：做代码讲解笔记

```text
请基于当前 CVPR 代码，逐函数解释：
- model.py: EnhanceNetwork/CalibrateNetwork/Network/Finetunemodel
- loss.py: LossFunction/SmoothLoss
- train.py 与 test.py 的执行流程
并输出成 markdown 学习笔记，适合 0 基础阅读。
```

---

## 11. 学习顺序建议（给 0 基础）

1. 先读本文件第 1、2、4 节；
2. 跑通一次 CPU 推理；
3. 对照第 8 节读 `model.py`；
4. 再读 `loss.py`，理解无监督约束；
5. 最后再看 `train.py`。

---

## 12. 你现在就可以做的下一步

- 第一步：让 AI/Agent按“模板 A”改 CPU 推理。
- 第二步：用 `weights/medium.pt` 在 `data/medium` 批量跑图。
- 第三步：把你最不懂的 20 行代码贴给 AI，让它逐行解释。

如果你按这个顺序做，通常 1~2 天内就能从“不会”到“能讲清 SCI 在干什么，并能自己跑结果”。
