# Team Long（第 3 名）迁移说明

NeurIPS 2022 NL4Opt **subtask 2**（自然语言 → 形式化 LP 建模）第 3 名方案
（`bigdata-ustc/nl4opt`）的复现代码，已迁移到 **transformers 5.20.2** 可直接运行。

## 与 baseline 的关系

在官方 baseline（BART + Copy Mechanism，见仓库根目录）基础上加三点：

1. **FGM 对抗训练** —— `attack_utils.py` / `train_attack.py`
2. **span 输入** —— `data_per_declaration_span.py` / `train_span.py`
3. **双模型集成** —— `test_mix.py`（span 输出目标函数 OBJ_DIR、attack 输出约束 CONST_DIR，再加两条后处理规则）

## 迁移改动（transformers 4.x → 5.20.2）

- `AdamW`：`from transformers import AdamW` → `from torch.optim import AdamW`
- 删掉 `BeamSearchScorer` / `greedy_search` / `beam_search` / `get_encoder` → 改用公开 `model.generate()`
- copy 机制依赖 cross-attention，`output_attentions=True` 时 SDPA 不支持 → `attn_implementation="eager"`
- `generation_bart.py` 的 forward 内强制 `output_attentions = True`

## 训练结果（WSL2 + RTX 4060 Laptop GPU）

- **attack（FGM）**：dev 最优 0.5757 @ epoch 29（baseline 0.4598）；FGM 分数震荡大，约 epoch 124 提前停
- **span**：batch_size 8 / accumulate_step 4（有效 batch 32 不变）；训练中（目标 200 epoch）
- 评分口径：`accuracy` = per-declaration，`1 - Σ(fp+fn)/Σd`

## 运行

```bash
# 数据/模型缓存：bert/ 和 data/ 需从 baseline 软链或自行下载（见根目录 README.baseline.md）
python train_attack.py --config configs/prompt-in-attack_fix_const_2.json
python train_span.py   --config configs/span-in.json
python test_mix.py --checkpoint <span> --checkpoint-c <attack>   # 集成
```
