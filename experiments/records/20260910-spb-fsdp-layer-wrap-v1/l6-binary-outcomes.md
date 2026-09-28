# Level6 全胜／全败跳变：只读审计

2026-09-10，北京时间21:20之后核验；结论范围固定为已完成step1–28。
未停止或改动43八卡长训，未新增GPU／游戏。局数、胜率均为训练rollout，非独立评测。

## 结论

最有证据的解释是：每批只有一个游戏seed，八局环境高度相关；模型又集中到
相同Marine计划，导致同一场景八局一起赢或输。相同计划失败的具体轨迹显示，
脚本小波次不断损兵、长执行队列阻断重新规划。不是发现了“某步PPO突然坏掉”的证据。
早期step3、8仍有八种不同回答序列，也都0/8；因此输出趋同不是全败的必要条件，
提高采样温度本身不能保证修好这些场景。

## 来源

RUN_ID：`20260910-spb-fsdp-layer-wrap-v1-spb2-l6-g2-opening-quota-rl96-8gpu-s0-r1`。
运行根为`/mnt/data0601/yangjinyue/new_SCII/experiments/20260910-spb-fsdp-layer-wrap-v1/runs`。
冻结SHA：`6a1657f7e045dcf998495a3d0adfb331572bf60d`；43 deployment：
`/root/new-scii-spb-fsdp-20260910/code-6a1657f`。

读取manifest、effective_config、training_seeds、episode_outcomes、prompt_response、
console及已结束游戏日志。模型是原始SpB起训后的当前策略，非路径标签中的Qwen3-30B。
实际data.train_batch_size=1、env.rollout.n=8、temperature0.4、top_p0.9、192token、
history0/off；未更改。训练八局是同seed独立进程，不是八个独立场景。

## 1. 为什么曲线接近0或100%

冻结`agent_system/environments/env_package/sc2_online_restored/envSC2.py:723–724`
先抽env_num个seed，再按group_n重复；此次为1个seed重复8次。
`agent_system/multi_turn_rollout/rollout_loop.py:410–421`为八局分配同一episode组、
不同trajectory ID。同组采样本身是相对优势训练的设计，不将它直接定性为bug。

| Step | seed | 八局两轮回答 | 胜局 |
| --- | --- | --- | --- |
| 18 | 52620 | 均为下面两段 | 8/8 |
| 20 | 17089 | 均为下面两段 | 0/8 |
| 22–27 | 18983/43095/51374/24152/35665/59045 | 每批均为下面两段 | 每批8/8 |
| 28 | 16921 | 前两段仍相同；第三段有6种回答 | 0/8 |

```text
<answer>
- supplydepot x 4
- barracks x 4
- marine x 24
</answer>

<answer>
- marine x 42
- supplydepot x 8
- barracks x 3
- reactor barracks x 3
</answer>
```

以上完整两段文本哈希（JSON序列SHA256前缀）均为`a3b47fa094ed`。
八局内的同轮query也逐字一致。projected_action未篡改这些有效回答，日志中最终
Executor队列分别为32和56个展开动作、顺序相同，不是解析成了空计划。
所有所查全败批次均terminated=true、truncated=false、game_result=Defeat、
termination_reason=game_result；并非Planner call20或进程超时被当失败。

这支持场景差异与既有打法共同决定胜负；训练同时改变权重，不能由跨step比较
证明seed是唯一因果变量，更不能据此排除全部优化器问题。

## 2. 兵确实造出来了，但失败场景不能持续作战

复用既有tracker账本，按[暴雪s2protocol](https://github.com/Blizzard/s2protocol)
build75689解码。以下每批仅一份留存回放，不冒称24局逐局统计。

| 留存回放 | 结果／游戏时长 | Marine产出／死亡 | 关键战损 |
| --- | --- | --- | --- |
| step20 | 败／10:24 | 66／66 | 51个死于VikingAssault；11个工人也被它击杀 |
| step27 | 胜／06:46 | 48／12 | 仅损失1工人，终局仍有36个Marine |
| step28 | 败／13:15 | 85／85 | Banshee杀38、VikingAssault杀31；最终12工人全灭 |

VikingAssault是落地形态，不能把step20误写为“Marine完全无法对空”。三份回放
均无新SCV产出（born的12包含初始工人），没有实质战斗升级；己方战斗单位仅Marine。
step28在9分钟时只剩3Marine，而敌方已有Banshee、Viking和Battlecruiser等。

冻结日志显示首攻约20兵，之后10兵波次。step20五个波次依次全灭；step28前五波
也均全灭。其模型回答没有attack条目，仍由自动战斗推进。冻结
`.../SC2ArenaDyStrategy/players/llm_player.py:101–102`默认首攻20、后续10。
这些是已发生的战损/触发证据，尚非“修改波次数即可获胜”的因果验证。

种子还改变出生位置与敌方具体发展。step20我方(127,125)、敌方(24,125)；
step27我方(24,22)、敌方(24,125)；step28我方(127,22)、敌方(24,125)。
额外浏览29份留存回放发现同一出生组合既有胜也有败，不能归咎于某个坐标。
回放initData内部randomValue不等于传入seed，不用该字段冒充游戏seed核验；
批次映射依据训练种子边界、console的保存路径和对应终局记录。

## 3. 第二段之后，Planner直到局面崩溃才得到新观测

step28八局前两轮query/回答各自完全一致；第三轮输入也逐字相同：

```text
04:13：11 SCV、21 marine、4 barracks、4 supplydepot。
       下发marine42→depot8→barracks3→reactor3。
11:43：11 SCV、0军队、7 barracks、12 supplydepot，1165矿、0气。
       此时才第三次调用Planner。
```

中间相隔7分30秒游戏时间。冻结`players/llm_player.py:677–680`仅在当前BO队列
耗尽时调用renew_bo；没有因前几波全灭就打断本段重新规划。
这不是此次分析新增的行为，也不将“约15列表条目”误写为15个展开动作。

第二段选择了Reactor但没选择待办里的refinery。实际日志明确出现：

```text
Action blocked: **Action Failed**: Insufficient Vespene Gas (0/50).
(3495) bo_skipped_53: {'reason': 'timeout', 'action': 'reactor barracks'}
(3716) bo_skipped_54: {'reason': 'timeout', 'action': 'reactor barracks'}
(3937) bo_skipped_55: {'reason': 'timeout', 'action': 'reactor barracks'}
```

三个Reactor逐项等到超时，再进入下一次规划。对应等待/跳过代码为
`players/llm_player.py:763–786`。这是缺气条件，不是此前兵营占用冲突的复发证据。
“格式/配额合法”并不保证这段计划的资源前置、战术和执行时机合理。

## 4. 为什么训练未必纠正这种失败

同seed、相同回答和相同结局使组内回报差消失，优势中心化后为零或舍入残差。
第22–27步日志advantage均0、entropy约0.002–0.003。参见已完成的
[CPU优势审计](advantage-zero-audit.md)和[调用次数折扣审计](sampling-reward-audit.md)。
这与[GiGPO的组内中心化机制](https://github.com/langfengQ/verl-agent/blob/master/gigpo/core_gigpo.py)
相容，不支持直接修改优势公式、跨seed混组或给零优势编造分数。

最值得先验的单一问题是：在不修改回答内容/战术/奖励的前提下，能否在严重战况
变化或明确失败后及时反馈，而不是长段结束才反馈。具体触发边界和队列如何保留
需要先设计并批准，不能直接自动取消队列或恢复6条限制。现有温度A/B仅验证探索，
不能据此替代执行反馈验证；本次未另行上卡或改代码。

后续[温度A/B完成结果](temperature-ab.md)补充：0.4/1.0各4/8胜，两个种子胜负
完全相同；其中合法小计划也被缺气Reactor阻塞、首波全灭。它提供了“缩短队列
并不自动足够”的反例，并未推翻本节的长队列现象或证明某种反馈修复已经有效。

## 可追溯性与限制

三份回放的SHA与解码输出见[l6-binary-replay-audit.json](l6-binary-replay-audit.json)。
原路径相对`sc2-logs/default_player/Flat128_Harder_Air/Qwen3-30B`分别为：
step20 `2026-09-10_10-55-54`、step27 `2026-09-10_12-36-58`、step28 `2026-09-10_12-47-00`。
同秒日志目录被多个worker共用，回放被覆盖后每批仅存一份；step28的trace.json
还存在Extra data解析错误。本次未修写入方式、未拼接损坏trace；文本日志仅作
存在性证据，不按重复行计事件次数，单局战损以留存回放为准。

本地副本：`/private/tmp/sc2-l6-binary-audit.BhYP03`。实际执行下列离线命令退出0，
三个回放哈希、build和胜负断言通过；未加载Torch或模型：

```bash
/private/tmp/sc2-air-results.HBI61m/decoder/bin/python -B \
  experiments/records/20260910-spb-fsdp-layer-wrap-v1/audit_l6_binary_replays.py \
  /private/tmp/sc2-l6-binary-audit.BhYP03
```
