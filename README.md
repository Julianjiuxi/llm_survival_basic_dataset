# LLM 生成长度：基础生存分析探索数据集

这是一个面向基础生存分析方法探索的公开 CSV 数据集。它把 50,000 次 Qwen3-8B 生成请求整理成两层：

- `data/requests.csv`：一行一个请求，适合 Kaplan–Meier、Cox、AFT、删失敏感性和静态协变量分析；
- `data/landmarks/*.csv`：一行一个“仍在生成的请求 × landmark”，适合 time-dependent covariates、剩余生存时间和未来 horizon 终止预测。

公开文件不含 prompt/response 全文、token IDs、源记录 ID、原始请求 ID或可对 WildChat 做字典匹配的哈希。`case_id` 和 `prompt_group_id` 由未公开密钥生成的 HMAC 伪名组成。

## 文件

```text
data/
  requests.csv
  landmarks/
    landmark_0032.csv
    landmark_0064.csv
    landmark_0128.csv
    landmark_0256.csv
    landmark_0512.csv
    landmark_1024.csv
    landmark_1536.csv
  data_dictionary.csv
  dataset_summary.csv
  manifest.json
```

完整字段解释见 [FIELD_DICTIONARY.md](FIELD_DICTIONARY.md)，研究设计与删失口径见 [METHODS.md](METHODS.md)，来源与限制见 [DATA_CARD.md](DATA_CARD.md)。

## Python 快速开始

```python
import pandas as pd
from lifelines import KaplanMeierFitter, CoxPHFitter

df = pd.read_csv("data/requests.csv")

km = KaplanMeierFitter()
km.fit(df["survival_time"], event_observed=df["event"])

cox_cols = [
    "survival_time", "event", "log1p_prompt_tokens",
    "code_switching_score", "paragraph_count", "digit_ratio"
]
CoxPHFitter().fit(df[cox_cols].dropna(), "survival_time", "event")
```

## R 快速开始

```r
library(survival)
d <- read.csv("data/requests.csv")
fit <- survfit(Surv(survival_time, event) ~ language_group, data=d)
summary(fit)

lm128 <- read.csv("data/landmarks/landmark_0128.csv")
coxph(Surv(remaining_followup_tokens, remaining_event) ~
      eos_logprob_last + entropy_raw_mean + repetition_mean,
      data=lm128)
```

## 使用时必须注意

1. `event=1` 表示观察到 EOS；`event=0` 表示在 2,048 token 上限处右删失。
2. 有放回抽样使同一 prompt 可能重复。划分和置信区间应按 `prompt_group_id` 分组。
3. 不要重新按单行随机切分。优先使用冻结的 `split`。
4. horizon 标签的空白不是 0，而是因行政删失而无法判定。
5. landmark 表只包含在该时点尚未结束的请求，已满足 `survival_time > landmark`。

## 重建与校验

`scripts/build_survival_exploration_release.py` 从本项目只读归档重新聚合公开 CSV；`scripts/validate_survival_exploration_release.py` 检查主键、风险集、标签、隐私字段和 SHA-256。

```bash
python scripts/validate_survival_exploration_release.py --root .
```

该仓库面向基础生存分析、动态协变量探索和方法复现，不适合事实性、偏好、安全性或个体层面的结论。
