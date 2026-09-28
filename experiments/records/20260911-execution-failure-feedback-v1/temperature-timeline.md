# 温度对照的首轮计划—首攻—第二轮分岔核对

2026-09-12；只读分析完成，未新增GPU实验或修改运行代码。沿用
[温度A/B](temperature-ab.md)的16局：A4 SpB、Level6、游戏seed42613，
T0.4/T0.8各8局。不是16个独立场景，也不是RL学习效果实验。

**结论：不能归因于“第二轮开始时已经把兵送光”。第二轮前主力完整；明显损失
发生在随后同一套自动首攻中。模型的生产选择能影响后续游戏，但这轮探索没有
改变首攻的触发、规模和首批单位。没有证据宣称局势当时已不可挽回。**

## 同一前缀的15局

A全部8局、B除env4之外7局，首答均为：

```text
- supplydepot x 4
- barracks x 4
- marine x 24
```

逐局回放核验：第二轮之前全部tracker事件相同；game-event差别仅是
`SSetSyncPlayingTimeEvent/SSetSyncLoadingTimeEvent.m_syncTime`墙钟同步值，
移除该字段后命令流完全一致。并非只凭相同prompt推断同一局势。
第二次输入哈希均为`b6ede1fcb6b45e98eaddc9bc5711a6bdd99be33823fc03f25e53b816dd36f5a1`。

| 游戏时间 | 实际证据 |
| --- | --- |
| 00:14–00:47 | 先后开工4个Depot；首个Barracks到01:13才开工，首个Marine于02:21出生 |
| 03:34.96 | 1个我方SCV在敌基地附近(20,122)被Marine击杀；仍没有Marine阵亡 |
| 04:09.29 | 第二轮规划：我方20个已出生Marine、11个SCV、4兵营/4Depot；没有建筑损失 |
| 04:10.71 | 自动成立首攻编队；15局选中同一批20个Marine：19个在第二轮前出生，另1个此时出生 |
| 04:13.21 | 第二答的首条Marine生产命令才提交，晚于上述首攻编队成立 |
| 04:30.89 | 15局首波均以20兵完成集结并进攻；回放实际命令与单位选择已核对 |
| 04:39.46–04:39.55 | 首个Marine阵亡；这里是首次死亡证据，不冒称首次开火/受伤时间 |
| 05:00.85–05:23.04 | 首批20兵全部死亡；按tracker最后击杀者，每局18–20个被Banshee击杀 |

04:09的敌方全图回放状态是26 SCV、4 Marine、2 Banshee、双星港/双科技附件。
**模型当时观测中没有可见Banshee**，只列1个敌SCV及双星港/科技附件等建筑；不能
用回放上帝视角指责模型“明知两架Banshee还不应对”。我方观测的army supply=24
也不能当成24个实际出生Marine，tracker当时是20个。

B-env4首答无效，走既有规则补齐，第二轮提前到01:47；不纳入上述同前缀比较。
该局约04:33以20兵出击、05:00首波全灭，也最终失败，单独保留在JSON中。

## 第二轮不是完全无效，但没有改变共同首攻

按实际出生和死亡统计，而非Executor提交成功数：

| 代表局 | 第二答Marine数 | 04:09–05:00出生/死亡 | 05:00–06:00出生/死亡 | 首波最后一兵死亡 |
| --- | ---: | ---: | ---: | --- |
| A-env0 | 42 | 9 / 17 | 12 / 13 | 05:00.85 |
| A-env1 | 8 | 9 / 17 | 3 / 13 | 05:02.28 |
| B-env0 | 1 | 5 / 12 | 10 / 8 | 05:06.79 |
| B-env2 | 4 | 8 / 10 | 0 / 10 | 05:23.04 |
| B-env7 | 2 | 6 / 10 | 0 / 10 | 05:09.06 |

不同回答还改变了建筑/后续规划，不能把上表当成纯Marine数量因果实验。例如B-env2
在05:00–06:00击杀17个敌SCV和1架Banshee，A-env0同期只击杀1个敌SCV、2个Marine；
因此“都败了”不等于“中间战斗完全相同”。但该局仍败，不能据此宣布学习提升。

A-env0整局产出66个Marine，最终死亡65个；后续以10兵重复出击。
该局首个非开局侦察SCV损失到09:30.67才出现，其他同前缀局相应损失也在更晚阶段。
所以不能用后期基地工人被杀，解释第二轮之前已经不可挽回。开局到04:09没有新增SCV，
11工人对26工人的经济差距则确实已经存在；是否值得改经济路线，本轮没有因果验证。

## 对应冻结代码及控制边界

实际SHA `9ae8668c4c6045f5834d197c72f7495c175c1c73`；以下行号均针对该SHA，
不是最新工作树。43冻结部署仍clean，三文件SHA256与本地`git show`逐一一致。

| 文件（SC2ArenaDyStrategy下） | 代码证据与影响 |
| --- | --- |
| `players/llm_player.py:105–110,654–659` | 首攻阈值20、后续10；每帧传入脚本维护，与模型是否输出进攻条目不是同一控制 |
| `players/script_player.py:4258–4288,4351–4356` | 空闲战斗单位达到阈值即`launch_total_attack`，先自动进攻、再处理显式增援指令 |
| `players/script_player.py:3204–3222,3298–3340` | 编队先去集结点，到齐后对目标执行进攻；日志中的最低5不是实际首攻只有5兵，本轮实际20兵 |
| `agents/strategist.py:246–271` | 启用`marine_pressure`，开局压低经济/科技；第二阶段保持Marine生产优先。实际query待办无Viking或撤退/暂停首攻选项 |

实际配置`SC2_AUTO_TRAIN_MARINES=0`、`SC2_ENABLE_NEWEXEC_MICRO=0`；不能把
这轮生产归给自动补Marine，也不能称这轮在使用新微操。`SC2_MAX_FORCED_ATTACKS=0`
没有关闭上表独立的自动进攻路径。A组首两答都没有OFFENSE，首攻仍然发生；B-env0
虽然第二答有OFFENSE8，也没有改变共同的20兵首攻。

**已确认的限制**是生产计划与首攻控制分离，当前候选只探索到了前者。
**尚未证明**自动阈值20本身是bug、首波全灭只由某一个微操缺陷造成，或关闭自动首攻
就会提升胜率。Banshee是最后击杀者，不等于承担全部伤害，更不能据此声称Marine
天生不能对空。未改任何机制，也未晋级温度短RL。

下一项若做因果实验，优先审议“同输入/回答下的自动首攻触发”这一个控制边界，
不能同时调温度、生产、prompt或战术；具体候选与GPU预算另行确认。本次只完成定位。

用户同意进入下一步后，已准备[首攻20/40最小对照方案](first-attack-threshold.md)
及6项CPU边界测试；具体诊断阈值和新GPU预算待确认，未实施运行hook或启动游戏。

## 复核与产物

[完整逐局证据](temperature-timeline.json)保存16局回放/log/trace哈希、实际出生/死亡、
首波20个单位tag、击杀者、执行命令、时间窗及第二轮状态；
[分析脚本](audit_temperature_timeline.py)复用已有死亡账本及命令选择解码，不安装新依赖。
CPU执行退出0；最后版额外核验15局首波单位tag一致。`git diff --check`通过。
这是分析产物，非项目bug修复或训练验收。

```bash
/private/tmp/sc2-air-results.HBI61m/decoder/bin/python -B \
  experiments/records/20260911-execution-failure-feedback-v1/audit_temperature_timeline.py \
  /private/tmp/sc2-temperature-replays.DAGhO7 \
  experiments/records/20260911-execution-failure-feedback-v1/temperature-timeline.json
```

原始run位于43本地根
`/root/new-scii-execution-feedback-local-20260911/shared/yangjinyue/new_SCII/experiments/20260911-execution-failure-feedback-v1/runs`，
两个RUN_ID见温度A/B文档；本地只复制约25MiB日志/回放，未读取checkpoint内容。
旧部署、全部结果和四卡2小时的未使用短RL预算保持不变。
