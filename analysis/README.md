# 分析 / 验证脚本

迁移与结果定位过程中使用的实验脚本，不属于 train/test 主流程。这些脚本硬编码了本机 WSL
的 checkpoint 与数据路径（`output/` 与 `data/` 未纳入仓库），因此主要作为**参考材料**，
用于说明验证与诊断过程；如需复跑，请按自己的环境修改路径并准备对应的 checkpoint / 数据。

## 迁移等价性验证

| 脚本 | 用途 |
|---|---|
| `equiv_harness.py` | 对比新旧 transformers 版 `forward` 的输出，验证迁移未改变数值 |
| `copy_check.py` | 校验 copy 机制：copy 分布、`p_gen`、最终输出概率和 ≈ 1 |
| `compare.py` | 迁移前后解码结果对比 |
| `verify_fix.py` | 验证 `parsers.py` 崩溃修复（`parse_number(None)`、缺失 `LIMIT`） |

## 解码 / 搜索实验

| 脚本 | 用途 |
|---|---|
| `beam_eval.py` | 扫描 `num_beams ∈ {1,3,5,8}`，得到 beam 反而变差的结论 |
| `badword_check.py` | 检查 bad-words 解码约束是否生效 |
| `debug_acc.py` | 精度调试辅助 |

## 错误分析（定位 0.46 vs 0.60 差距）

| 脚本 | 用途 |
|---|---|
| `err_analyze.py` | 统计表面 token 错误（约 68.1% 声明有表面错误） |
| `err_parsecheck.py` | 检查 XML 解析层面的错误 |
| `err_categorize.py` | 对错误分类计数 |
| `err_canonical.py` / `err_canonical2.py` | 分析 canonical 转换阶段的对齐 / 误差 |
| `err_reason.py` | 逐声明归因：objective 45.5% / constraint 32.0% 正确率 |

## 其他

| 脚本 | 用途 |
|---|---|
| `make_200.py` | 生成 / 切换 200-epoch 训练配置 |
