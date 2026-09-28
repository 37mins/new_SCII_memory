# SpB seed43567：零胜发生在原始模型，也发生在更新后

2026-09-08。只读核对既有日志、冻结源码及两份存留回放；未改runtime，未新增游戏或PPO。
**这次证据不支持“首个global step把模型训坏，因而第二批全输”作为唯一解释。**
当前更明确的限制是：固定开局已发生大量战损，Planner到09:52才接管，接管后仍受
预置进攻阶段、BO菜单及自动战斗约束。尚未因此定位到新的实现bug，也未证明RL无害。

## 对照与结果

共同runtime `e1e25ce1d14362b209e34ec73871ebaeb63538a6`；两份manifest中的源码文件哈希、
模型manifest、g2哈希逐项一致。原始模型为Panyuqi/SpikingBrain-2.0-instruct，非项目SFT。
共同为Level6/Harder、Flat128/Base75689、Terran Air、g2、6BO/96token、history0/off、
dropout0、FP32参数/温度、BF16 autocast、HF/eager/cache、采样temperature .4/top_p .9。

| seed43567，每组8局 | 原始模型新评测 | 完成第1个global step后的历史第2批rollout |
|---|---:|---:|
| 自然胜负 | 0胜8负 | 0胜8负 |
| 逐局累计日志reward均值 | −49.119375 | −48.8685 |
| 实际消费的Planner回答 | 184 | 146 |
| 严格有效回答 | 122/184（66.30%） | 110/146（75.34%） |
| 全六行Marine的回答 | 57/184 | 42/146 |
| 无Planner的游戏 | 0 | 0 |

有效率分母只含`policy_action_applied=True`的真实回答，排除已终局环境的占位行。
因此不同于控制台`val/sc2_action_valid=.625`的聚合口径；不混用两个数值。
累计reward包含每局终局奖励，只求和至该局真实done，包含所有8局。

| env索引 | 原始：调用/有效/累计reward/时长 | 更新后：调用/有效/累计reward/时长 |
|---|---|---|
| 0 | 17 / 13 / −49.500 / 15:55 | 16 / 10 / −50.000 / 16:47 |
| 1 | 32 / 24 / −48.500 / 22:28 | 12 / 11 / −49.500 / 15:53 |
| 2 | 28 / 17 / −49.000 / 20:45 | 12 / 7 / −48.668 / 15:54 |
| 3 | 30 / 22 / −49.500 / 23:32 | 36 / 29 / −48.000 / 23:53 |
| 4 | 18 / 10 / −49.000 / 16:28 | 10 / 9 / −49.000 / 15:43 |
| 5 | 12 / 7 / −49.000 / 15:55 | 19 / 16 / −48.500 / 17:16 |
| 6 | 16 / 8 / −50.000 / 15:57 | 10 / 7 / −48.780 / 19:34 |
| 7 | 31 / 21 / −48.455 / 24:10 | 31 / 21 / −48.500 / 23:16 |

两边都是同一个43567重复8次，不是8个独立游戏种子。新评测四卡，历史训练八卡，
未恢复相同模型采样RNG，env编号仅供定位，不代表逐随机轨迹配对。历史训练第5批
超时退出1，只完成4步；save8尚未触发，checkpoints为空。因此这不是两个已保存
checkpoint的独立配对复评，更不能用本次结果证明参数更新链条已通过验收。

## 已定位到的具体限制

**1. 胜率突降混入了游戏seed变化。** 历史四批为2732、43567、42613、52416，
对应8/8、0/8、0/8、0/8。首批8胜发生在首个global step更新之前。
seed2732首次输入在10:00已有26 Marine、4 Tank、3 Marauder；43567在09:52仅有
12 Marine、2 Tank、2 Marauder。这些差异已出现在模型首次回答之前，不能归因给RL。

**2. 这次同seed两组并未在Planner之前分岔，但共同经历了高损耗开局。**
16局的首次query逐字相同，237项固定BO的开始时间/动作序列也相同；两份保留回放
在首次Planner前的tracker事件完全一致。该阶段已死亡91 Marine、2 Tank、2 Marauder、
1 SCV（MULE到期不计战损）。91 Marine中62个由敌方VikingAssault完成最后击杀，
75个死在x=30–39、y=90–99区域；不是单纯“造兵命令没有产出”。
16局日志均显示接管前9次自动进攻触发，仍使用20首攻/10后续。不能仅凭同期发生
就把全部损失归因于波次阈值；侦察、目标、路径和具体微操也可能参与。

**3. 模型确实能影响执行，但不是自由制定整局战略。** 原始模型env0第8次回答（12:48）：

```text
- marine
- marine
- marine
- marine
- marine
- marine
```

Planner和Executor队列均保留这6条，随后日志记录6次`BARRACKSTRAIN_MARINE`提交。
两组全部严格有效的全Marine回答分别52/52、38/38原样进入队列。提交证据不等于
这6个指定单位全部完成；回放另证实两局确实大量产兵。全Marine回答在原始模型
已占31.0%，更新后28.8%，不是此次更新后才新出现的输出模式。

限制也确实存在：所有330条实际输入的剩余BO里都没有Stim、Combat Shield、
步兵武器升级或Medivac；prompt明确只准选当前待办。starport/Viking已加入备选，
不能再说模型完全不能选对空。无效回答仍走规则补齐；Commander的独立Marine guard
也保留，例如原始env7首次合法计划中一个Marauder被换成Marine。这是冻结规则，
不是本次分析新增修复，也不能把该替换计作模型选择。

**4. 接管后仍是持续生产、持续消耗，升级与资源转化没有解决。** 仅存回放分别为
原始env1和更新后env3（按唯一局长匹配）：全局产出216/221 Marine、28/30 Tank、
21/19 Viking。09:52–12:00又损失27/34 Marine、6/5 Tank；12:00剩7/10 Marine、
0/2 Tank。17:58记录存款9120/9635矿、111/107气，两边均55名存活SCV；之后工人才
被敌军大量杀死。因此这两局不是“开局工人已被杀光”，而是先消耗战斗部队、后经济
失守。两份回放均无实质战斗升级。数千矿未转化的具体瓶颈，不能单靠余额定性。

09:52回放上帝视角可见敌方8 VikingFighter和1 Battlecruiser，模型输入却仅见敌方
建筑、无可见敌军。这是需要区分的观测范围，不据此认定观测实现有bug；不将雾外
信息塞给模型，也不声称当时局面已经不可逆。

## 冻结代码依据与唯一建议

以下行号均来自`git show e1e25ce:<path>`，不是当前工作树推断。路径前缀为
`agent_system/environments/env_package/sc2_online_restored/SC2ArenaDyStrategy/`。

| 位置 | 实际行为 |
|---|---|
| players/llm_player.py:64、660 | 先加载完整固定BO；队列耗尽才调用Commander/Planner |
| players/llm_player.py:83 | 首攻20、后续10 |
| agents/strategist.py:273、344 | Marine/Tank all-in阶段；air开关仅增加starport/Viking备选 |
| agents/planner.py:103、1290 | 当前待办约束；合法回答不补兵重排，无效回答仍走fallback |
| agents/commander.py:132 | 独立opening Marine guard仍能调整首次不含Marine的计划 |

**当前最有依据的下一项是审议“Planner接管边界”的单变量对照，不是继续盲改KL或prompt。**
本轮没有证明某个新代码bug，也没证明某个阈值调整必定涨分。若要让模型更早参与，
必须先明确在哪个有来源的原阶段边界交接、如何保留阶段待办，再报最小配置申请；
不擅自清空237项队列、不编造五份BO转阶段算法、不同时换战术或补升级。

后续已完成[接管时机只读审议](planner-handoff-review.md)：建议在g2的#138前交接，
保留同一末阶段及待办来源；尚未实施或新增GPU实验。

## 复核入口与产物

共享根：`/mnt/data0601/yangjinyue/new_SCII/experiments/20260908-garrison-wave-ownership-v1/runs`。

- 原始评测：`20260908-garrison-wave-ownership-v1-spb2-original-l6-g2-s43567-eval8-r1`，
  44 GPU1–4，编排`721ff1e773ce0ada3fae10d3d52ab245272179b2`，21:30:45退出0。
- 历史训练：`20260908-garrison-wave-ownership-v1-spb2-l6-g2-rl96-8gpu-s0-r6`，
  42 GPU0–7，编排`939b41b698e732345a36c921cb24d91a4fc8a83c`；抽取reset1/
  step_index4–39，不混入首批或第三批。global step不等于optimizer update。
- 原始env0 worker后缀`647254.out`第17231–17343行保存上述6Marine回答及命令；
  原始env7后缀`647259.out`第13480–13481行保存Marine guard替换。
- 新旧每批只存留1个回放，存在按秒目录碰撞；两份回放证据不得泛化为16份回放。
  路径中的`Qwen3-30B`是历史玩家日志标签，不是实际加载的SpB权重。

本地只读下载快照：`/private/tmp/sc2-failed-seed-audit.mIL2YQ`。
在本记录目录执行：

```bash
python3 analyze_failed_seed.py /private/tmp/sc2-failed-seed-audit.mIL2YQ /private/tmp/sc2-failed-seed-audit.mIL2YQ/analysis.json
/private/tmp/sc2-air-results.HBI61m/decoder/bin/python decode_failed_seed_replays.py /private/tmp/sc2-failed-seed-audit.mIL2YQ /private/tmp/sc2-failed-seed-audit.mIL2YQ/replay-analysis.json
```

两条离线分析实际通过。第二条复用现有mpyq/s2protocol环境（Base75689，22.4loop/s），
不安装依赖或启动游戏。完整派生JSON保存逐回答、命令行号、worker哈希、死亡位置、
击杀者和统计时间序列，不将大日志提交Git。解析时按完整回答序列匹配worker，
reset slot只是并发槽，不当成env索引；该分析脚本的修正不计项目bug修复。

关键SHA256（原始 / 历史）：

```text
prompt_response.jsonl
718260042a008e27604d1be49075642078aeadb85760dfdcb2165e0f27ce30b2
50d54eee5286f863bf1ad8ec6ba68a6801b593bc775ef9697386ed80fc84e669
episode_outcomes.jsonl
06022291da7eb403e2db52000beb0a887ade03323914b39ea1d3d532fc1428cb
7216bb432a5a15786edbf69bf618b0f6a2dbb7b8b3ef52968d4eae42cafdb086
replay.SC2Replay
b3cee4e053633fd14f10ec00af6d645b052ace623c6d15f12b9483e50bbc3104
4e079a0fd758d55d6b6c3906bb27eac2fb17788178e36d3623a8756e1dadb569
共同首次query
ab58bbc984096d17ad410b3ea2ff87d191a898ebcbbceba8dc8b4345587995ba
共同首次Planner前tracker序列
ff9d5790ab3fbede1322d6e58b499d9e09abcc90f993929c9785cfe1a699f49e
```
