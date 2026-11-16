# NL4Opt Subtask 2 — BART + Copy 基线迁移到 Transformers 5.x

本仓库将 NeurIPS 2022「NL4Opt」竞赛 Subtask 2 的官方基线（BART + Copy Mechanism 的
text-to-text 生成模型）从旧版 transformers（4.x）迁移到 **transformers 5.20.2**，并在
GPU（CUDA）上完成训练与评测的复现。

> 原始基线：[nl4opt/nl4opt-subtask2-baseline](https://github.com/nl4opt/nl4opt-subtask2-baseline)
> （原始说明保留在 [README.baseline.md](README.baseline.md)）

## 任务

给定线性规划（LP）题目的自然语言描述，生成其形式化建模（目标函数 + 约束的 XML 表示）。
官方基线用 BART 编码器-解码器 + copy 机制 + 声明级 prompt 逐条生成，评分口径为
`scoring.overall_score = 1 - Σ(fp+fn)/Σd`（见 `scoring.py`，未改动）。

## 迁移内容（4 个文件，+37 / −35 行）

transformers 5.x 移除了一批旧 API，迁移改动如下：

| 文件 | 改动 | 原因 |
|---|---|---|
| `train.py` | `AdamW` 改为 `from torch.optim import AdamW` | 5.x 不再从 `transformers` 导出 `AdamW` |
| `model.py` | 解码改用公开 `model.generate()`，不再调用已删除的 `greedy_search` / `beam_search` / `BeamSearchScorer`；`num_beams` 不再硬编码为 1 | 5.x 删除了这些内部方法 |
| `generation_bart.py` | 在 forward 内强制 `output_attentions = True` | 5.x 默认 SDPA attention 不支持 `output_attentions=True`，而 copy 机制依赖 cross-attention；加载时用 `attn_implementation="eager"` |
| `parsers.py` | `LIMIT` 缺失时 `limit=''` 改为 `0.0`；`parse_number(None)` 增加保护 | 修复训练第 54 epoch 触发的 `TypeError` 崩溃 |

## 环境

- Python 3.10+
- PyTorch 2.x（CUDA）
- transformers 5.20.2

```bash
pip install torch transformers rouge beautifulsoup4 lxml word2number jsonlines
```

## 复现

1. 下载数据（本仓库**不含**数据集）：从
   [nl4opt-competition](https://github.com/nl4opt/nl4opt-competition) 的 `generation_task`
   目录取 `train.jsonl`、`dev.jsonl` 放入 `data/`。
2. 训练（官方配方 `configs/baseline.json`，bart-base + copy + prompt，200 epoch，贪心解码）：
   ```bash
   python train.py --config configs/baseline.json
   ```
3. 评测：
   ```bash
   python test.py --gpu 0 --checkpoint output/baseline/<run>/best-checkpoint.mdl --test-file data/dev.jsonl
   ```

## 结果

在共享 dev 集（99 例）上复现的 per-declaration accuracy：

| 设置 | dev accuracy |
|---|---|
| 贪心解码（num_beams=1，官方配方） | **0.4598** |

结论：
- 官方报告的 `Acc = 0.60` 是在竞赛**隐藏测试集**上的结果（该集合未公开），与共享 dev 集
  不可直接比较；本复现在共享 dev 集上为 0.4598。
- 评分口径与官方完全一致：`scoring.py` / `test_utils.py` 未改动，`overall_score` 公式同上。
- beam search 反而显著变差（num_beams=1→0.46、3→0.155、5→0.149、8→0.124）：copy 机制的
  输出是**概率**而非 logits，beam 打分时再做 `log_softmax` 造成二次归一化，beam 排序失真。

## 目录结构

```
.
├── configs/baseline.json   # 官方训练配方
├── generation_bart.py      # BART + copy 机制（已迁移）
├── model.py                # 训练 / 解码封装（已迁移）
├── parsers.py              # XML 输出解析（已修复崩溃）
├── train.py / test.py      # 训练 / 评测入口
├── scoring.py              # 官方评分（未改动）
├── data/                   # 数据集（需自行下载）
├── analysis/               # 迁移验证与错误分析脚本（见 analysis/README.md）
└── train_*.log             # 200-epoch 训练日志
```

## 参考

- 官方基线：<https://github.com/nl4opt/nl4opt-subtask2-baseline>
- 竞赛与数据：<https://github.com/nl4opt/nl4opt-competition>
- 竞赛冠军方法（NER 高亮 + BART-large）：<https://arxiv.org/abs/2212.13201>
