# 独立评测与第二条计划分岔

2026-09-11。只读分析，没有新GPU实验、PPO或生产代码修改。

## 结果

三组统一开启失败反馈；A4是**训练时关闭反馈**的step4，B4是**训练时开启反馈**的step4。
两者均从同一原始SpB训练。不是Qwen对照，也不是旧Level7 step24。

| 权重 | 胜率 | 平均整局累计reward | 合格且实际消费的回答 | Planner决策总数 |
|---|---:|---:|---:|---:|
| 原始SpB | 3/8 | -23.3445 | 57/78（73.1%） | 78 |
| A4 | 5/8 | -5.71575 | 22/23（95.7%） | 23 |
| B4 | 2/8 | -31.94725 | 61/63（96.8%） | 63 |

三组8局均自然终止，没有无Planner决策局；退出码均0、评测PPO更新均0。
完成时间分别为北京时间20:45:44、21:00:01、21:26:49。
不是把inactive占位回答计入分母；reward只累计各局首次done之前及当次的返回值，
重算均值与既有纯评测指标相差小于0.001。`val/sc2_action_valid`的末状态平均
不是此处整局回答合格率，不能混用。

| 同一实际游戏seed | 原始 | A4 | B4 |
|---|---|---|---|
| 42613 | 负 | 负 | 负 |
| 52620 | 负 | 胜 | 胜 |
| 17089 | 负 | 负 | 负 |
| 4018390961 | 负 | 胜 | 负 |
| 3236598124 | 负 | 负 | 负 |
| 3297512220 | 胜 | 胜 | 胜 |
| 773351930 | 胜 | 胜 | 负 |
| 1098341402 | 胜 | 胜 | 负 |

这组结果支持保留A4为候选，不支持宣布B4学会纠错或晋级长训。
每seed每权重仅一局，且包含刻意选出的诊断seed，不是总体随机评测；不据此声称
统计显著、泛化提升或反馈机制普遍有害。三组的逐局reward、调用数和回答见
[机器可读证据](learning-eval-analysis.json)。

## 分岔已发生在第一次失败反馈之前

A4/B4八个第二次query逐字一致。开局都选4 Depot、4 Barracks、24 Marine；
第二次A有7/8局选42 Marine，B有6/8局选4 Marine、2/8局选8 Marine。
相同query不意味着后续随机采样流逐token相同；首答长度不同会改变采样消耗。

重点追踪两个A胜/B负seed，并保留一个两组都胜的反例。对应回放tracker从开局
至第二query所在整数游戏秒**所有事件一致**；不是只凭同seed假定开局相同。

| 证据 | 4018390961：A / B | 1098341402：A / B |
|---|---|---|
| 第二query时间 | 同为04:09 | 同为04:21 |
| 第二答Marine数量 | 42 / 4 | 42 / 8 |
| 全局Marine实际出生数 | 66 / 116 | 49 / 44 |
| 6分钟场上存活Marine | 14 / 12 | 27 / 4 |
| 后续关键生产断档 | A持续补兵；B 05:52–07:27无新Marine | B 05:19–07:02无新Marine，约103秒 |
| 首次失败反馈 | A没有；B 07:09缺气Reactor | A 07:02人口阻塞；B 06:45缺气Reactor |
| 终局 | A 08:42胜 / B 19:33负 | A 07:40胜 / B 09:58负 |

### seed1098341402：最清晰的短窗口

两组首攻均04:30建立、04:49出发，5分钟均有27个存活Marine。
5–6分钟A产出9、损失10个Marine；B产出3、损失26个，其中17个被地面Viking击杀。
B第一、第二攻击波分别在05:24、06:03全灭，6分20秒场上Marine已为0。
**这些恶化早于06:45第一次缺气反馈**；不能说都是反馈重规划之后造成的。
A持续补兵，6分钟尚有27 Marine，同一时段摧毁敌方40个SCV，B为23个。
但5分20秒时A/B累计出生仅33/32，存活已为21/13，说明交战过程已经放大差异；
不能用后来的103秒断档倒推它解释了全部最初损失，也不能将胜负仅归因于Marine数量。

B的第二条原始回答（`prompt_response.jsonl:16`）：

```text
<answer>
- marine x 8
- supplydepot x 2
- barracks x 3
- reactor barracks x 3
- scv x 2
- refinery x 1
</answer>
```

04:21后的执行顺序为先补8 Marine，再建Depot/Barracks，再等待Reactor。
06:45反馈记录`processed_prefix_count=13`、缺气`0/50`、目标命令未提交；
尚未执行的后缀包含`reactor barracks, reactor barracks, scv, scv, refinery`。
后缀退出旧队列、待办释放后交回模型；B随后仍把Reactor放在Refinery前，
07:25、08:05、08:45重复同一缺气阻塞。直到08:45才改成Marine28/Depot3，已晚。
该局Refinery、Reactor实际开工均0；并非“Refinery建成但不采气”。

A也不是正确规划的充分例子：07:02已经人口阻塞，第三答仍先Marine19再Depot8；
最后凭此前军队获胜，整局未新增开局四个以外的Depot。

### seed4018390961与反例

B早期少量Marine后转建建筑，05:52–07:27约95秒没有Marine出生；07:09第一次
缺气反馈后改回Marine32。此后确实继续产兵，整局最终116 Marine，比A的66还多，
但损失115个，后期击杀者还包括Battlecruiser。**不是整局没生产，而是时机与交换很差。**
A前两波也被消灭，之后仍持续生产并在08:42获胜；不能把“有波次全灭”单独判bug。

反例seed52620：A/B分别42/39 Marine出生，两组都胜、reward都20.5。
A曾约167秒没有Marine出生，B发生5次缺气Reactor反馈仍获胜。
因此“存在生产断档/重复缺气”不是跨场景失败的充分条件，不能凭单局再改执行器。
另一个A胜/B负seed773351930本轮只核对结果及回答，未深入解码其交战。

## 实现证据与判断边界

冻结SHA `937ae3cfcbf7340208a8f86c0abf680d733c0484`：

- `players/llm_player.py:449`的`_fail_current_bo`仅在未提交目标命令时生成失败反馈，
  释放尚未提交的待办；`:720`下一帧调用Planner替换执行队列。这是已批准候选的
  行为，不是本次新发现的实现偏差。缺气时Refinery后缀被丢回模型的链条与源码一致。
- 同文件`:1015`的`llm_decision_final_action`在`:1025`调用`run_actions`之前记录。
  其计数不能直接称为引擎成功命令；预走位后的建筑执行还可能走其他记录路径。
  本文实际产出以回放`SUnitBornEvent`为准、建筑开工以`SUnitInitEvent`为准。
  seed1098341402的A/B Marine预提交记录49/45，回放出生49/44；不把两者强行等同。
- 回放使用已有s2protocol Base75689、22.4 game loop/s解码；首攻/集结/全灭来自
  各局日志，时间由冻结4 loop/iteration换算，并与trace游戏秒核验。
- 三组manifest登记的源码哈希与`git show 937ae3c:<file>`逐项相符；运行目录的
  模型恢复、种子、退出状态、console及回放哈希均核验。

**已确认：** 模型权重变化对应了相同输入下不同计划；计划差异确实进入生产，
随后补兵节奏、建筑投入与战斗结果分岔；B仍重复无法执行的缺气计划。
**尚未证实：** 某处实现bug、第二答是唯一胜负原因、反馈训练为何学出短段偏好、
以及修改奖励能否改善。不能仅用非零advantage或高格式合格率代替学习验收。

## 唯一下一项

用户已批准下述四卡45分钟安排；准备和启动状态见[第二答对照](second-plan.md)。

优先seed1098341402，用同一B4权重和相同历史开局，纯评测仅将第二答替换成已记录
A4第二答；之后继续在线B4。对照保留B4原第二答。两组重放前缀/替换回答明确标为
人工干预，不算模型决策。只问“改变这一个规划段，能否改变随后补兵和交战”。
不同时修prompt、反馈、战术或reward，也不续长训。
建议复用四分片checkpoint，四卡一个模型实例、同seed A/B各4局，总上限45分钟、
不做PPO、不复制大权重、不自动重试；资源及预算确认后才准备并启动。
少量同seed重复仅验收执行因果，不是独立泛化或学习效果评测。

## 复核入口

43原始run根在`/root/new-scii-execution-feedback-local-20260911/shared/yangjinyue/new_SCII/experiments/20260911-execution-failure-feedback-v1/runs`，
RUN为`20260911-execution-failure-feedback-v1-spb2-{original|A4|B4}-l6-feedback-on-eval8-r2`。
本地小型快照：`/private/tmp/sc2-feedback-second-plan.1KHcXh`，没有复制checkpoint。

```bash
/private/tmp/sc2-air-results.HBI61m/decoder/bin/python -B \
  experiments/records/20260911-execution-failure-feedback-v1/audit_second_plan.py \
  /private/tmp/sc2-feedback-second-plan.1KHcXh
```

实际退出0，核验24局退出/加载/种子/整局reward、8对第二query、6份回放与trace，
同时保留反例。完整审计JSON SHA256：
`de44dec3b1f16613c1b92290e2cb5b94e36d6226be7d2a718ef3262cb26e372d`。
诊断脚本只读，没有将任何诊断代码加入训练部署。
