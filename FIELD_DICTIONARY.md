# 字段说明

机器可读的完整字典在 `data/data_dictionary.csv`。下面给出字段组和最容易误用的变量。

## 主键与设计

- `case_id`：HMAC 伪名请求键，可连接 requests 和 landmark 表。
- `prompt_group_id`：同一规范化 prompt 的 HMAC 伪名组键，用于防泄漏切分和聚类推断。
- `sampling_block`：十个有放回抽样子集之一，不是自然人或会话 ID。
- `split`：冻结的 train/validation/test。

## prompt 特征

- `prompt_tokens`、`model_input_tokens`：chat template 前后 token 数。
- `log1p_prompt_tokens`：`log(1 + prompt_tokens)`。
- `was_truncated`：是否在 4,096 prompt token 处截断。
- `language_*`：规则语言识别结果、置信度和状态。
- `code_switching*`：多语言混合启发式标记和分数。
- `length_instruction_type`、`requested_length_*`：是否以及如何提出数字长度要求。
- `has_*`、`*_count`、`digit_ratio`、`repetition_bigram_ratio`：纯文本结构统计，不包含原文。

## 结局与成本

- `survival_time`、`event`：生存分析的时间和事件列。
- `censored`：`1-event` 的便捷列。
- `first_eos_token_index`：自然结束时 EOS 位置；删失请求为空。
- `endpoint_class_18`、`endpoint_interval_18`：最终终止位置的固定 18 类；最后一类为 `>1920`。
- `time_to_first_token_ms`、`total_latency_ms`：批处理上下文时延，仅用于本次系统配置内分析。

## 动态特征

- `eos_logprob_*`：raw 分布中 EOS log probability 的窗口摘要。
- `log_eos_rank_*`：EOS logit rank 取对数后的摘要。
- `entropy_raw_*`、`processed_entropy_mean`：采样变换前后分布熵。
- `margin_*`：raw top-1 与 top-2 概率差。
- `surprisal_*`：实际采样 token 的 processed surprisal。
- `support_rate`：EOS 进入 processed sampling support 的比例。
- `repetition_*`、`sentence_rate`、`paragraph_rate`：重复和局部边界信号。

`last` 是 landmark 前最后一步，`mean/max/min` 是窗口聚合，`slope` 是窗口内线性趋势。具体窗口由冻结特征构建代码定义。

## horizon 标签

每个 `end_within_h` 都配有 `end_within_h_observed`。只在 observed 为 1 时使用标签；CSV 空白代表未知而非否。
