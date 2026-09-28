# 首攻20/40：八局回放只读归因

2026-09-12。对象为`e3c0695`的A4 SpB / Level6 / seed42613，A/B各4局；
全部自然失败、0次PPO。[原结果](first-attack-threshold.md)不变。
本次只新增CPU审计和记录，未修改部署、prompt、奖励、战斗或训练，不占GPU。

## 结论

**不是首攻开关失灵，也不是驻防重新拉走这两波40兵。** 两局没有首攻的战斗单位
峰值确实只有36；另外两局的40个Marine均可在回放中逐tag确认战死。只把首攻20
提高到40没有解决问题：人口/前置顺序错误拖慢成军，敌方已经发展出混合兵团。
尚不能将这些现象的全部原因归于prompt或RL，也不把它定义为某个实现bug已确认。

## 两波40兵如何损失

用实际集结A-Move选择集确定首波tag，再与tracker死亡、击杀事件配对。死亡时间
和日志“全灭”一致；全部成员在进攻开始前存活，首攻后没有MOVE命令，未发现
驻防重新分配的证据。死亡位置在敌方基地附近，而非我方基地。

| 首波 | 出发 | 实际全灭 | 最后一击归因（不是总伤害贡献） |
| --- | --- | --- | --- |
| A四局，每局20 Marine | 4:30.89 | 5:00.85 | Banshee19、SCV1 |
| B env1，40 Marine | 8:24.82 | 9:03.21 | 地面Viking23、Marine11、Banshee5、Battlecruiser1 |
| B env3，40 Marine | 8:31.79 | 9:11.88 | Banshee15、地面Viking13、Battlecruiser9、Marine3 |

A首攻时敌方场上为约5 Marine、2 Banshee、27 SCV；B成军时已有12 Marine、
3 Banshee、10 Viking及48 SCV。第一艘Battlecruiser在8:44.46产出并参与随后击杀。
这些是全地图回放实况，不是模型当时可见的全部敌情。

40兵并非完全没有造成伤害：env1首波击杀8 Marine、2地面Viking、6 SCV等；
env3击杀14 Marine、8地面Viking、12 SCV等。两波各摧毁1个**尚未建成**的
CommandCenter，不能误称摧毁了成熟分矿。两组己方均只有Marine形成战力，无
战斗升级、Marauder、Tank、Medivac或Viking真正产出。

所以“推迟出发即可避免被打光”没有得到支持。这里只证明实际交战损失及敌方
发展时间差，不证明某个新兵种、升级或微操改法一定能赢。没有逐帧HP、隐身/侦测
状态，不能仅因敌方研究了BansheeCloak就断言所有Marine被隐形单位无伤击杀。

## 为什么另外两局没有首攻

四局B共同到达5:28：**36 Marine＋11 SCV，47/47人口，剩余人口0**。
第二条重放计划是`marine x42 → supplydepot x8 → …`。提交12个Marine后，
下一项被人口阻塞；5:49失败反馈已明确要求先造Depot，剩余Marine30/Depot8
也已返还为可选项。第三轮输入相同，随后才因在线回答不同而分岔。

| 游戏 | 后续实际链条 | 没到40的原因 |
| --- | --- | --- |
| B env0 | 仍选Marine3→Depot3；再次反馈后又选Marine28→Depot8，之后SCV也排在Depot前 | 反复在补给站之前人口阻塞；开局4个Depot之后，直到9:37才有新Depot开工。Marine总产41、同时存活峰值36 |
| B env2 | 一度补出大量Depot，但7:01又选Depot9→Marauder4→…→Tech Lab；后续反复把Marauder放在Tech Lab前 | 7:30已有72空人口，却只有45矿；此后Marauder缺Tech Lab超时，后面的Marine50未获执行机会。Marine总产38、同时存活峰值36 |

这里应区分两个问题：env0没有解除人口前置，env2解除人口后又被资源/科技前置
卡住。不能把两局都写成“只差4个兵”或“首攻阈值判断失效”。

env1/env2/env3第三答都把Marine2放在Depot2之前。6:22.50和6:57.68各死了一个
Marine，随后6:25.18和7:00.71出现两个Marine提交命令，才轮到Depot；首批新Depot
7:24.55完成。这是死亡释放人口后队列推进的实证，不是模型主动纠正了顺序。
env0第三答选择Marine3，第三个仍阻塞，7:40再次超时，整个未提交后缀返回重规划。

真实反馈与回答示例（B env0）：

```text
05:49 feedback: Supply blocked (Available: 0 < Required: 1.0).
Suggestion: Construct a SupplyDepot immediately.
answer: marine x 3 → supplydepot x 3 → …

07:40 feedback: Supply blocked …
answer: marine x 28 → supplydepot x 8 → …
```

**接口提供了纠错信息和选择，但当前A4在这些输入上没有可靠地把前置动作排到前面。**
冻结Planner的格式示例此时仍是`Marine2→Depot2→Barracks2`（`planner.py:1190–1201`
从待办前几项构造），与失败建议存在顺序冲突。这是可核验的可疑因素，尚未做
同输入示例A/B，不能据此宣布prompt是唯一根因。

## 代码边界与下一项

- `script_player.py:2961`取得非工人、可攻击单位；`:4226–4288`排除编队/侦察单位，
  按可用数量比较20/40。无首攻两局连全部存活Marine都不超过36，足以解释未触发。
- `llm_player.py:449–483`在未提交动作失败后返回反馈并释放未提交后缀。当前行为
  没有执行到后缀，不等于后缀被引擎拒绝；它也不会自行把Depot/Tech Lab提前。
- `script_player.py:2621`仍排除已在进攻编队的驻防对象；本轮首波实际MOVE为0，
  不支持把此前驻防归属问题再列为这两波全灭的原因。

下一项应收敛到**失败后重规划的前置顺序**，不继续阈值搜索或直接长训。
已有[seed52620 Depot前移对照](supply-order.md#完成结果1700后只读核验)已证明合理
选择可以执行，但两组都赢、reward相同，未证明模型学会。本轮则提供了A4对
同一失败输入重复错误的具体样本，可复用为同输入纠错对照；是否改示例及新增
模型/GPU预算另行确认。本次未实施该修改或启动任何新评测。

## 可复现证据

[CPU脚本](audit_first_attack_replays.py)复用既有选择解码、出生/死亡及快照方法；
[简要机器记录](first-attack-replay-summary.json)包含各局源文件哈希、峰值、首波
击杀归因、人口和在线回答。完整逐tag/命令审计位于
`/private/tmp/sc2-first-attack-audit.kur2Zs/analysis.json`，原始小型产物在同目录。

```bash
<decoder-python> -B experiments/records/20260911-execution-failure-feedback-v1/audit_first_attack_replays.py \
  /private/tmp/sc2-first-attack-audit.kur2Zs \
  /private/tmp/sc2-first-attack-audit.kur2Zs/analysis.json \
  experiments/records/20260911-execution-failure-feedback-v1/first-attack-replay-summary.json
```

实际CPU运行1.45秒、退出0：八局source/result/replay/console哈希校验通过，第二答
之前tracker事件与排除壁钟同步字段后的命令前缀八局一致，六个首波的全部tag死亡
验证通过；没有把新增审计脚本算作项目bug修复。`git diff e3c0695 -- agent_system
verl gigpo scripts configs`为空，正式模块和冻结部署均未改。
