# Home Credit Track B：LightGBM / XGBoost 训练与 Rolling OOF 手册

## 1. 本轮交付是什么

本轮只继续 **Track B 模型训练与时间验证**，不运行 Competition Forensics，也不导入或评价 Track A 的 Notebook、分数或后处理结果。

新增入口：

```text
home-credit-diagnostics train-track-b
```

它直接消费现有客户级中间表：

```text
data/kaggle/base_100features.parquet
```

已核实的输入规模：

| 项目 | 数值 |
|---|---:|
| 客户行数 | 1,526,659 |
| 候选特征 | 100 |
| 完整周 | 92（0–91） |
| `target=1` | 47,994 |
| 总体坏账率 | 3.1437% |

`case_id` 只用于对齐，`WEEK_NUM` 只用于时间切分和周指标，`target` 只作为已经成熟的违约标签。模型矩阵不会包含这三个元数据字段。

> 当前 Parquet 是已经加工、筛选后的 100 特征中间表。本轮能检验的是“给定这张中间表时，模型能否在未来完整周保持区分能力”，不是重新执行原始多表到 472+ 特征的端到端验证。

## 2. 时间回放口径

主情景设定：

```text
label_maturity_lag = 8 complete weeks
validation_block = 4 complete weeks
refresh_step = 4 complete weeks
outer_quarantine = 12 complete weeks
```

八周成熟延迟是模拟参数，不代表 Home Credit 的真实生产标签一定在八周后成熟。

每折包含四个互斥区间：

```text
Fit weeks
    ↓ 只用于拟合模型
Inner early-stopping weeks
    ↓ 已成熟，只用于确定树的轮数
Eight-week shadow gap
    ↓ 输入已发生、标签在回放时点尚未成熟，不进入 fit
Four complete future validation weeks
    ↓ 先生成分数，标签成熟后才做 OOF 评价
```

实际 92 周会生成以下 8 折：

| Fold | Fit | 折内 Early Stop | 成熟训练整体 | 未成熟 Shadow | Rolling OOF |
|---|---|---|---|---|---|
| B01 | 0–27 | 28–31 | 0–31 | 32–39 | 40–43 |
| B02 | 0–31 | 32–35 | 0–35 | 36–43 | 44–47 |
| B03 | 0–35 | 36–39 | 0–39 | 40–47 | 48–51 |
| B04 | 0–39 | 40–43 | 0–43 | 44–51 | 52–55 |
| B05 | 0–43 | 44–47 | 0–47 | 48–55 | 56–59 |
| B06 | 0–47 | 48–51 | 0–51 | 52–59 | 60–63 |
| B07 | 0–51 | 52–55 | 0–55 | 56–63 | 64–67 |
| B08 | 0–55 | 56–59 | 0–59 | 60–67 | 68–71 |

因此，真正的 pooled rolling OOF 只覆盖 week 40–71。week 0–39 是暖启动训练区，代码不会用训练内预测给这些客户补分。

最终隔离关系为：

```text
可用于最终冻结模型的成熟周：0–71
冻结前 Shadow：72–79
Outer Quarantine：80–91
```

当前命令只生成 inner rolling OOF，明确保留但不评价 week 80–91。静态 Parquet 仍由同一离线进程加载并做字段库存盘点，因此这里保证的是 Outer 标签、指标和行没有参与拟合、early stopping 或评价，不是物理权限隔离的数据保险库。

## 3. 两个模型怎样训练

### LightGBM

默认配置强调可复现的 CPU 基线：

- `learning_rate=0.03`
- `num_leaves=31`
- `min_child_samples=1500`
- 行、列采样均为 0.8
- `reg_alpha=0.1`、`reg_lambda=5`
- 最多 2,000 棵树
- 折内成熟时间尾段 early stopping 150 轮
- 固定随机种子，串行 Fold，避免同时占用多份大矩阵

代码使用规范的 `reg_alpha` / `reg_lambda` 参数名，保留原 Notebook 的类别分列、固定随机种子、双类别检查和逐折内存释放意图。

### XGBoost

默认 CPU Histogram 基线：

- `tree_method=hist`
- `learning_rate=0.03`
- `max_depth=6`
- `min_child_weight=50`
- 行、列采样均为 0.8
- `reg_alpha=0.1`、`reg_lambda=5`
- `max_bin=256`
- 原生类别切分，不做高维 one-hot
- 最多 2,000 棵树
- 同样只用折内成熟周 early stopping

本机已验证 XGBoost CPU 路径。CLI 默认写明 `--xgboost-device cpu`；只有在目标运行环境的驱动、CUDA 与 XGBoost wheel 已单独烟测通过后，才应改成 `cuda`。

### 类别处理

当前 100 特征中，名称以 `MODE(`、`MONTH(`、`SEASON(` 或 `WEEKDAY(` 开头的编码字段按类别处理。

每一折都独立执行：

1. 只用 Fit 周建立类别词表；
2. Early-stop 与未来 OOF 中的新类别映射为 missing/unknown；
3. 类别数超过 10,000 的字段改用训练折频率编码；
4. 未见类别频率为 0；
5. 不使用全量类别词表，不进行 10 万维 one-hot。

每折的新类别比例会写到：

```text
categorical_unseen_by_fold.csv
```

## 4. 安装和真实数据

安装代码与模型依赖：

```bash
python -m pip install -e ".[training,test]"
```

本项目不会把 Kaggle Parquet 提交到 GitHub。若本地尚无文件，可执行：

```bash
kaggle datasets download \
  -d diarray/deep-feature-synthesis-home-credit-stability \
  -f base_100features.parquet \
  -p data/kaggle \
  --unzip
```

Kaggle CLI 需要单独安装，并完成必要的登录或规则确认。

## 5. 先做真实数据烟测

以下命令仍然读取真实 Kaggle Parquet，但每周最多抽取 1,000 行，只运行首折和 50 棵树：

```bash
home-credit-diagnostics train-track-b \
  --train data/kaggle/base_100features.parquet \
  --out track_b_outputs_smoke \
  --models lightgbm,xgboost \
  --max-folds 1 \
  --max-rows-per-week 1000 \
  --n-estimators 50 \
  --early-stopping-rounds 10 \
  --xgboost-device cpu \
  --lightgbm-device cpu \
  --run-id track-b-real-smoke-v1
```

本机已经完成这条烟测，输出 4,000 条 week 40–43 的 OOF 分数：

| 模型 | AUC | KS | AP | Lift@5% | Lift@10% |
|---|---:|---:|---:|---:|---:|
| LightGBM | 0.720675 | 0.341630 | 0.098130 | 3.829114 | 3.016322 |
| XGBoost | 0.733085 | 0.404236 | 0.103473 | 3.417722 | 2.848101 |
| 50/50 Blend | 0.735726 | 0.407360 | 0.104633 | 3.544304 | 2.848101 |

这些数值只证明真实数据加载、折内类别转换、两个模型、OOF 回写和指标计算能够贯通。它们使用了行抽样、单折和极少树，**不能作为最终模型有效性结论**。

## 6. 运行完整 Rolling OOF

完整命令不设置 `--max-folds` 和 `--max-rows-per-week`：

```bash
home-credit-diagnostics train-track-b \
  --train data/kaggle/base_100features.parquet \
  --out track_b_outputs_full \
  --models lightgbm,xgboost \
  --label-maturity-lag-weeks 8 \
  --validation-weeks 4 \
  --step-weeks 4 \
  --min-mature-train-weeks 32 \
  --inner-early-stopping-weeks 4 \
  --outer-quarantine-weeks 12 \
  --n-estimators 2000 \
  --early-stopping-rounds 150 \
  --n-jobs 12 \
  --xgboost-device cpu \
  --lightgbm-device cpu \
  --run-id track-b-lag8-full-v1 \
  --dataset-label "Kaggle base_100features.parquet full rolling OOF"
```

默认使用当前 100 个预建特征，因为本轮按用户要求先验证模型训练和 OOF。若后续需要一个更严格的日历代理对照组，可额外加：

```text
--exclude-time-proxies
```

但不要根据 Outer 结果再决定是否使用这个选项；对照组应在 inner rolling OOF 阶段预先比较并冻结。

16 GB 内存环境建议顺序运行 Fold，不要同时开两条训练命令，也不要做全量 one-hot 或大规模 Optuna。完整双模型 8 折可能需要较长时间，先用首折全量行测速，再估算整轮耗时。

### 已完成的完整 8 折结果

本机随后使用全量 1,526,659 行、全部 8 折、每折最多 800 棵树和 100 轮折内 early stopping 完成了完整 inner rolling OOF。运行约 11 分钟，生成 559,660 条 week 40–71 的 OOF：

| 模型 | AUC | KS | AP | Lift@5% | Lift@10% | Lift@20% | Stability |
|---|---:|---:|---:|---:|---:|---:|---:|
| LightGBM | 0.801606 | 0.458083 | 0.148646 | 4.947032 | 4.016746 | 3.050146 | 0.577893 |
| XGBoost | 0.799394 | 0.453157 | 0.146993 | 4.884119 | 4.001018 | 3.024009 | 0.575973 |
| 固定 50/50 Blend | 0.801969 | 0.457923 | 0.149348 | 4.932229 | 4.017671 | 3.037424 | 0.579424 |

八个 Fold 的 LightGBM AUC 范围为 0.795858–0.831089，XGBoost 为 0.794871–0.831129；pooled 32 周 Gini 斜率均接近 0 且非负。结果支持两个模型在 inner 时间回放中具有稳定区分能力，但 Blend 相比 LightGBM 的增量很小。

该完整运行仍包含全部 100 个预建特征，即 `exclude_time_proxies=false`，因为本轮按要求聚焦训练和 OOF；它是时间验证兼容性基线，不代表字段治理、Outer 或生产审批已经完成。轻量可审查快照见 [`track_b_results/`](../track_b_results/README.md)。

Fold 8 的 XGBoost 最佳轮数为 769/800，接近本次树上限；这不影响当前固定 800 棵树基线的有效性，但若继续做 inner 优化，应提高树上限并让 early stopping 充分结束，再冻结下一版，而不是根据 Outer 调整。

## 7. OOF 与模型产物

每次运行会生成：

| 产物 | 作用 |
|---|---|
| `oof_predictions.parquet` | 不含标签的 `case_id` 对齐模型分数 |
| `oof_labels.parquet` | 独立保存成熟标签和可用周 |
| `oof_evaluation.parquet` | 标签成熟后的一对一评价连接表 |
| `fold_manifest.csv` | 每折 Fit/Early-stop/Shadow/Validation 周界、样本和正例数 |
| `overall_metrics.csv` | pooled rolling OOF AUC、KS、AP、Lift 和 stability |
| `fold_metrics.csv` | 每折模型指标 |
| `weekly_metrics.csv` | 每个真实周的指标 |
| `categorical_unseen_by_fold.csv` | 验证期新类别比例 |
| `feature_manifest.csv` | 元数据排除、模型特征和折内编码方式 |
| `feature_importance_by_fold.csv` | LGB gain/split 与 XGB gain/weight，折内归一化 |
| `feature_importance_summary.csv` | 分模型、分口径汇总后的重要性与平均排名 |
| `models/` | 每折 LightGBM 文本模型和 XGBoost JSON 模型 |
| `preprocessing/fold_XX.json` | 每折类别词表和高基数训练频率映射 |
| `run_config.json` | 完整参数、样本量和周范围 |
| `RUN_REPORT.md` | 自动生成的运行摘要 |

`oof_predictions.parquet` 只含被未来完整周真正评分的行。暖启动周没有分数，这是时间 OOF 的正常边界，不应填 0，也不应填训练集预测。

Track B 的时间 OOF 只覆盖 week 40–71，不能直接传给 Diagnostics V3 的旧 `run --oof` 入口；后者要求整张训练矩阵的 `case_id` 精确覆盖，这是刻意保留的不同安全边界。

## 8. 怎样判断模型是否有效

完整运行后至少同时检查：

- pooled OOF AUC、KS 和 Average Precision；
- Lift@5% / 10% / 20%；
- 八个 Fold 是否多数保持正向区分能力；
- week 40–71 的周 Gini 均值、斜率和残差波动；
- 最差周与最差 Fold；
- LightGBM 与 XGBoost 的结果方向是否一致；
- 新类别比例上升时，模型性能是否同步下降；
- Feature Importance 是否只在单一 Fold 突然集中。

当前代码预注册了简单的 50/50 概率平均，只作为附加结果。不要查看 Outer 后再调整 Blend 权重。

同一离线进程会在模型评分完成后连接已经模拟成熟的验证标签来计算指标；这是严格的时间顺序回放，不是独立标签系统或物理权限隔离。当前也没有最终 week 0–71 部署模型、Outer 分数、概率校准、阈值选择、业务成本优化或候选特征 paired ablation。

在完整 inner OOF 结果稳定、特征和参数冻结之前，不应解封 week 80–91。若以后执行一次性 Outer 验证，需要把它单独记录为新的固定版本；不能用同一隔离区反复调参后仍称其为未见数据。
