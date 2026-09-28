# B192低有效率：生成阶段重复与超配额

2026-09-09只读审计。对象为已完成的
`20260909-bo-segment-protocol-v1-spb2-l6-g2-opening-B192-8gpu-eval4-s0-r1`，
44冻结runtime `71fae36ccc112d4c9cd303ab084c1e0a0c7eebec`，现场Git clean。
模型为原始SpikingBrain-2.0-instruct；本run没有PPO，不能称为训练导致的退化。
没有改生产代码、参数、历史run或启动GPU实验。

## 直接发现

**低有效率的直接来源是模型原始回答已重复、超出整段待办配额，不是解析器把正确
计数误读。** 19条无效中17条，仅统计已写完整且名称合法的动作就已超配额。
15条回算达到192token的回答全部属于这17条；CPU仅删除最后一行后，15条仍全部
无效。因此“只差更多token把末尾补完整”不足以解释或解决这一批问题。

| 实际样本（env / action_id） | 输入与原始输出 | 现场队列证据 |
| --- | --- | --- |
| 0 / 3，02:11 | 待办marine20；回答先`marine x 2`，重复`marine x 1`，完整条目合计31，末尾另有半截`marine x` | console第1473行：旧fallback最后仅3个Marine |
| 2 / 7，05:28 | 待办SupplyDepot7、Barracks1；回答先复制示例的2/1，再接待办的7/1，合计9/2，另重复攻击 | 第3215行：规则队列SupplyDepot、Barracks、Marine、Reactor、Marine、攻击；不是原回答原样执行 |
| 3 / 5，02:48 | 合法`marine x 15` | 第1907行完整15个Marine，没有截回6个 |

现场行号来自run/console.log，完整原回答来自prompt_response.jsonl。
上表“队列”只指Planner交付内容，不等同于单位实际建成。

计数样本2/7原始回答为：

```text
Here are the Build Orders with attack instructions:
- marine x 2
- supplydepot x 2
- barracks x 1
- Launch OFFENSE with 8 marine
- supplydepot x 7
- barracks x 1
- reactor barracks x 2
- scv x 3
- Launch OFFENSE with 8 marine
```

冻结validator逐行累计同名动作，随后检查整段配额。它先报告第一个失败原因，
所以原日志中的`marine x`或格式失败会遮住同一回答中已经发生的超配额。
本审计不是把17条重新标成唯一失败类型，而是补充其并存问题。

## 输入中值得优先核验的契约

Planner同时要求“approximately 15 entries”、允许重复整行或`x N`，并说
`N is a positive integer within its remaining count`。validator实际要求的是
**所有同名行的数量相加后**不超过剩余配额；prompt没有把跨行求和写清楚。
动态示例又来自当前待办，并要求`Follow its shape exactly`，但没有直接说明
“示例不是一份额外任务，不能再与整份待办相加”。27/41回答以完整示例内容开头，
其中11条最终无效；上面的2/7确实发生了示例+待办重叠。

这是有证据的提示歧义/复制倾向，**尚不是已证明的生成根因**。完整示例前缀也
出现在16条合法回答中；待办总数量不足15的5次请求全部合法。因此不能说只要
看到示例或“约15条”就必然失败，也不能把15改回6当作结论。

## 已排除与未排除

| 检查 | 结果 |
| --- | --- |
| 缺可用表、缺one-shot | 41份输入都有；动态示例41/41通过同一冻结validator |
| 输入被4096上限截断 | 实际tokenizer/chat template在CPU重建后844–982token，无触顶；日志也未截短 |
| 正确回答被validator误判 | 同输入重算41/41与原日志一致；19条均有可指出的格式、名称或配额违反，未发现正确回答误判证据 |
| 合法`x N`解析错误、仍截6 | 22条合法回答全部解析计数一致；现场15个Marine完整保留 |
| build_scheduled_sequence复制原始回答 | 该函数产物是规则后处理序列，不是query里的展开列表；重复已经存在于原始llm_response |
| EOS明显配错 | run/model与tokenizer均为151645（im_end），pad151643；未见ID不匹配。原生成token IDs/EOS标志未保存，不能据此证明实际停止行为全无问题 |
| PPO把模型训坏 | 本次0更新、原始权重，排除本run的PPO为原因 |

无效回答仍走用户冻结的整段拒绝→旧6条截断/严格名称解析→规则补兵与重排。
含`x N`的无效回答不会按B合法路径展开，这解释了原始计划与实际队列差异；
是本次保留的既有边界，不在本审计中修改或包装为新发现的实现bug。
3/4系统胜率不能说明这些被拒绝的规划已经正常生效。

## 代码位置与最小下一项

以下位置均为冻结71fae36，不以最新开发树替代：

- `agents/planner.py:27–30,117–125,1178–1189`：段长、数量说明和动态示例。
- `agent_system/environments/strategic_sc2_env_manager_v2.py:895–903,928,944–946`：名称/数量解析、整段累计配额。
- `agents/planner.py:1288–1308,1324–1333`：合法B原样展开；无效回答保持旧fallback。
- `agents/planner.py:725–753`：query候选行与规则scheduled_sequence分开生成。
- `agent_system/multi_turn_rollout/rollout_loop.py:92–103,134–139`：单user消息套真实chat template再tokenize。
- `verl/workers/rollout/hf_rollout.py:100–108,138–158`：EOS、192预算、cache及生成入口。

`agents/`完整前缀为
`agent_system/environments/env_package/sc2_online_restored/SC2ArenaDyStrategy/`。

建议唯一下一项候选：只澄清B prompt的**整段数量预算及结束规则**——同名动作
跨行合计不超余量，示例不是额外任务，选完本段结束而不为约15行凑数。
保留约15条目标、允许重复与x N、合法顺序和无展开上限；不改validator/fallback、
BO来源、战术、预算或模型。不强制同名只能出现一次，以免改变BO排序能力。
这只是待批准的prompt候选，不是已验证修复；当前未实施、未申请或消耗新GPU预算。

## 可复核产物

[CPU审计JSON](budget192-answer-audit.json)保存41条原回答、候选配额、prompt哈希、
逐条validator结果和删末行的离线诊断，不将删末行当作生产修复。
原日志SHA256为`5067499ab9efd79f88bb5c11a333780dea003417bb6d53a89fdc6a7ba8ecdebf`。
脚本实际执行并退出0：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:tests \
  /private/tmp/sc2-air-results.HBI61m/decoder/bin/python \
  experiments/records/20260909-bo-segment-protocol-v1/audit_budget192_answers.py \
  /private/tmp/sc2-b192-readonly-20260909.jsonl
```

tokenizer检查在44禁用CUDA、只读实际run/model的小型配置及词表，使用现有
Transformers依赖；没有加载权重。八卡/四卡差异和仅两个独立seed仍限制因果推断。
