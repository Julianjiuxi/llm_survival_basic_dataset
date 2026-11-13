# 研究设计与变量口径

## 生存结局

- `survival_time`：从第一个生成 token 风险时刻起计数。自然结束时包含最终 EOS 风险行。
- `event=1`：模型采样到 EOS。
- `event=0`：在 `censor_limit=2048` 前未采样到 EOS，属于行政右删失。
- 可见输出长度使用 `output_tokens_excluding_eos`；不能用它替代生存时间。

## 静态协变量

所有 prompt 特征都由规则、tokenizer 或字符串统计产生，不依赖额外大模型标注。语言、code-switching 和长度指令是启发式标签，适合作为协变量或分层变量，不应当作人工金标准。

## Landmark 口径

landmark `a` 的一行满足 `survival_time > a`，且动态特征只使用 `token_index <= a` 的历史。`remaining_followup_tokens = survival_time - a`，`remaining_event` 继承请求最终事件状态。

未来 horizon 标签 `end_within_h` 定义为从 landmark 起 `h` token 内是否观察到 EOS：

- 若 EOS 已在随访中观察到，标签可判定为 0 或 1；
- 若请求删失且删失边界仍覆盖整个 horizon，标签为 0；
- 若请求删失且 horizon 超过 2,048 token 边界，标签未知，CSV 留空，同时 `end_within_h_observed=0`。

## 重复 prompt 与切分

抽样有放回。同一 prompt 的重复请求共享 `prompt_group_id`，并被分配到同一冻结 split。课程作业应：

- 使用现成 split；
- cluster bootstrap 或稳健标准误按 `prompt_group_id` 聚类；
- 不把重复 prompt 跨 train/test 随机拆开。

## 推荐 baseline

1. 无协变量 KM 与语言/截断分层 KM。
2. Cox PH：静态非文本 prompt 特征。
3. AFT：Weibull 或 log-normal，显式处理删失。
4. landmark Cox：在 128 或 256 token 风险集加入 EOS/熵等动态特征。
5. 二元模型：`end_within_128`，仅使用 `end_within_128_observed=1` 的行。

报告至少包含样本数、事件率、风险集大小、缺失处理、重复 prompt 处理、比例风险诊断和删失敏感性。
