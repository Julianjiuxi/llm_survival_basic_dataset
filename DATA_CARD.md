# 数据卡

## 概览

本数据集记录 Qwen3-8B 在 50,000 个 WildChat 派生 prompt 上的生成长度、EOS 结局、静态非文本 prompt 特征以及七个 landmark 的运行时摘要。

| 项目 | 值 |
|---|---|
| 请求数 | 50,000 |
| 唯一 prompt 组 | 36,289 |
| 观察到 EOS | 46,653（93.306%） |
| 右删失 | 3,347（6.694%） |
| 行政删失上限 | 2,048 新 token |
| 生成模型 | Qwen3-8B，thinking disabled |
| 采样 | temperature 0.7, top-k 20, top-p 0.8 |
| landmark | 32, 64, 128, 256, 512, 1024, 1536 |
| 来源 prompt | `allenai/WildChat` 的本地清洗与有放回抽样 |

## 组成

`requests.csv` 包含匿名请求键、匿名 prompt 组键、冻结 split、静态 prompt 特征、长度结局、事件/删失状态以及请求级时延。

七个 landmark CSV 只保留在相应 landmark 尚未终止的请求，包含截至该点可在线获得的 EOS、熵、margin、surprisal、support、repetition 与文本边界摘要，以及清楚标记可观测性的未来 horizon 标签。

## 脱敏

公开 CSV 明确移除：

- prompt 和 response 全文；
- output、sampled、top-1 token IDs；
- WildChat source record ID、候选池 ID；
- 原始项目请求 ID、pool prompt ID、dedup hash；
- prompt 文本哈希和 model-input 哈希；
- generation seed、创建时间、服务器路径和账户信息。

`case_id` 和 `prompt_group_id` 使用未随仓库发布的密钥做 HMAC 伪名。数据仍不应被解释为绝对匿名，也不得用于重建、关联或识别原始用户内容。

## 预处理与抽样

清洗池排除了空首条用户消息、项目规则匹配的 PII-like 文本、`redacted=true` 和标记为 toxic 的来源记录，并进行规范化与去重。十个 5,000 条子集采用固定种子独立有放回抽样。模型 prompt 在 4,096 tokenizer token 处截断；50,000 个请求中 194 条被截断。

自动过滤和规则标签可能漏检，不能证明所有敏感、版权或不良内容已从内部全文归档中清除。本公开探索版本不分发全文，从根本上降低了这一风险。

## 推荐用途

- KM 曲线、log-rank、Cox PH、AFT 和删失敏感性；
- 静态 prompt 协变量与长度结局的关系；
- landmark 动态生存分析；
- 未来多时间尺度终止概率和校准；
- 重复 prompt 的 cluster bootstrap 或 frailty 扩展。

## 限制

- 一个模型快照和一个解码配置，不能代表所有 LLM。
- prompt 不是总体用户请求的概率样本，且仅来自本地处理的 WildChat 分片。
- 规则语言识别和静态标签存在误差。
- 2,048 token 上限导致长尾删失。
- 请求级时延受 dynamic batching 和硬件影响，不是隔离请求的服务时间。
- 数据不含回答质量、事实性或人工偏好标签。
- 本项目 test split 已用于多轮探索性研究，因此基于该 split 的既有结果应视为探索性结果；独立确认需要另设确认数据。

## 许可与引用

WildChat 的数据库许可为其官方仓库所述的 ODC-By；该许可不自动解决单条内容的独立隐私、人格或版权权利。Qwen3 模型许可和引用以所用模型卡为准。见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) 和 [DATA_LICENSE.md](DATA_LICENSE.md)。

本项目代码采用 MIT 许可。CSV 数据按 ODC-By 来源义务发布，具体边界见 `DATA_LICENSE.md`。作者：Hanzhi Zhang，联系邮箱：zhangjiuxi80@gmail.com。
