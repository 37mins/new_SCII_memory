# 末阶段对空备选：单局系统候选

2026-09-08，状态 completed（10:59:48北京时间自然败局，exit0）。用户批准继续最小候选，另明确批准：冻结无效回答
补齐规则，区分归因后测试系统效果。不把本候选称为实现bug修复或模型能力提升。

对照是已完成的 `20260908-valid-bo-passthrough-v1-spb2-l6-g2-s2621640384-r1`，
冻结源码 `019ca9942a586797621e301db77b988b65dfa7bd`；从其结果记录提交
`8c5049b66fe42930d192d9c2102e48b5cf426e1b` 隔离开发，不改原deployment。
对照自然Defeat，reward -49.5，消费37条回答、27条完整有效，其中18条有效的
全6Marine计划全部原样保留。原始数据、哈希与局限见
[对照记录](../20260908-valid-bo-passthrough-v1/README.md)。本次不重跑对照。

## 唯一机制变化

`agents/strategist.py` 增加默认关闭的 `SC2_ENABLE_AIR_OPTIONS`。启用时只在
现有 `marine_pressure_strategy` 最后一阶段的 `bo todo` 末尾追加
`starport: 1, vikingfighter: 8`；原待办及阶段/攻击指令保持不变。
额度逐字复用同文件既有 `anti_air_strategy` 末阶段，不切换整个anti_air策略，
不声称1/8是最优配兵。它是阶段待办额度，不是全局单位上限或即时执行指令。
假设是扩展待办可以让既有系统生成并执行对空生产链；能否及时建成并改善战斗
尚待实际游戏验证。仅胜负波动不能证明这一链成立。

沿用原前置过滤：无Factory不提供Starport；无自身Starport不提供Viking。
沿用原建筑观测扣减和阶段续期，不修正在建/完成语义。CPU回归调用真实
Strategist、Planner、现有validator及假模型socket；合法6Marine仍是6Marine。

`SC2_PRESERVE_VALID_BO=1` 保留；合法回答不额外补兵或重排。**无效回答的
补齐代码未改，但新待办会影响其结果**：无效 `- siege tank` 可触发规则补
Starport，已有Starport后可补Viking。因此按完整回答有效标记分开统计
「合法模型计划」和「无效回答经规则处理的计划」，后者不能算模型选择。
如进一步报告“规则新增”数量，只统计最终计划相对可识别原始动作的新增部分。
Commander独立开局Marine guard仍保留，单独归因，不宣称整条执行链无兜底。

Executor已有映射：STARPORT为SCV建造、前置FACTORY；VIKINGFIGHTER由STARPORT
训练，不要求TechLab。48只读CPU核验使用实际burnysc2 7.1.3的
`sc2.dicts.unit_train_build_abilities.TRAIN_INFO`，文件SHA256
`4f692d6c8ab51407033b6a2954c2a76ececc35a873e48a625a95149a9922c014`。
旧自动攻击将所有非工人可攻击单位纳入战斗队列，包括Viking；这不是专门的
Viking微操验收。新micro仍关闭，旧侦察/20兵首攻/10兵后续波次原样保留。
没有修改模型、prompt模板/one-shot、解析器、资源检查、micro、奖励或生成预算。
动态query仅因待办/后续观测变化而改变。

## 配置与执行

RUN_ID `20260908-planner-air-options-v1-spb2-l6-g2-s2621640384-r1`。
单卡、同seed一局纯评测，最多1800秒，无PPO、无重试；48 GPU6须重新确认空闲。
预算之外的新增实验另申请。源码、Git bundle与run均用独立新目录，GitHub推送
此前被拒绝，本次不重试推送；服务器保存私有Git bundle及哈希用于恢复。

使用同一原始SpB2 instruct snapshot（无项目SFT/RL），关闭SWA dropout，
HF cache开启、eager、FP32参数及温度归一化、BF16 autocast。Level6/Harder、
Flat128、Terran vs Terran Air、canonical g2、实际seed2621640384；history0/off、
legacy6/6、4096/96、greedy、shaping off、自然终止均同对照。模型及BO由原
manifest验证，新增Strategist文件哈希纳入 `configs/system-v1.sha256`。

```bash
python3 -B -m unittest discover -s tests -p 'test_planner_air_options.py' -v
bash experiments/records/20260908-planner-air-options-v1/run_spb_node48.sh
```

启动前保存effective config并实际解析Hydra，核对对照仅多出worker变量
`SC2_ENABLE_AIR_OPTIONS='1'`（除源码/输出等基础设施路径）。模型加载、SC2
推进及实际seed确认后才记running，不将CPU通过称为在线通过。

交付保留退出码、原始回答、最终BO、执行日志、回放及哈希。分别报告备选是否
进入query、模型/规则各选多少、实际提交、开工和产出；这些事件不能混称。
对空单位无产出时定位未选择、未执行或未完成，不直接继续改micro。报告自然
胜负、整局reward、消费回答数/完整有效率及战斗效果；无Planner局亦保留。
单seed不足以估计总体胜率或架构优势，正式采用与否需后续证据。

CPU实录：Python3.9，`python3 -B -m unittest discover -s tests -p <文件名> -v`
分别运行 `test_planner_air_options.py`（8项）、`test_valid_bo_passthrough.py`
（7项）、`test_r6_planner_interface.py`（5项）、`test_r6_format_reward.py`
（6项），共26项全部通过。新入口 `bash -n`、完整源码manifest的
`shasum -a 256 -c configs/system-v1.sha256`、`git diff --check` 均exit0。
游戏源码相对对照只有Strategist的4行追加，无其他模块改动。

10:41北京时间启动前检查拒绝GPU4：其他任务PID2429013已占用约3GB。
没有创建tmux、正式run或模型/游戏进程，不是一次已运行的候选失败。
保留 `c586cda` 及服务器bundle、CPU预检目录；10:42再次确认GPU6空闲，
只将入口设备从4改6，以新SHA恢复独立源码，仍是一局/一卡/1800秒预算。
原CPU配置审计脚本曾因未保留Hydra键名的`+`而assert失败；修正审计读取后
实际Hydra解析exit0、唯一机制差异为新增候选开关。这不是项目bug修复成果。

## 实际启动

冻结SHA `3adaaf70ac879a7f4d7bdecbfb8be60b75bf4fa6`，clean，源码
`/root/new-scii-planner-air-options-20260908/code-3adaaf7`。服务器Git bundle
SHA256 `d0a8ae0ac67305c5076970fc79c53b4c49e32d4132e7a2116b97dc033ed39358`。
48 GPU6，tmux `sc2_spb_air_options_r1`，10:44:34外层启动，manifest时间
10:44:59。最多30分钟，约11:14:34截止，TERM宽限最多30秒；预计11:05附近
结束，但这不是保证。10:47:30已执行到游戏01:36的固定g2 BO#10，GPU进程
`ray::WorkerDict` PID2432080约40.6GB，SC2Worker PID2431676。

10:48实际config与CPU预检逐项比较，除输出路径归一化后完全一致；预检相对
上一局对照仅多出 `SC2_ENABLE_AIR_OPTIONS='1'`。实际 `trainer.val_only=True`、
resume disable、1GPU；模型 `SSESWAMoBAForCausalLM`、swa_dropout0、
attention_dropout0、use_cache true。模型manifest SHA256
`40d6f1417568d5fda1107cdafd66c7441a679b03fb148851c8f226ecc9cf884c`，
g2 BO SHA256 `304b1f4ff12fe30c5e6d5f7b8eaef8ed6823ca65afed3a3706cae10e3639c494`。
种子文件唯一记录为 `base_seed=1000, reset_index=0, seeds=[2621640384]`。
此快照尚在首次Planner之前，不能宣布对空生产或胜率通过。

离线归因入口（不改变部署或运行日志）：

```bash
python3 -B experiments/records/20260908-planner-air-options-v1/attribute_plans.py /path/to/completed-run
```

它按消费回答逐条对齐 `BO ori / BO`，核对原始文本和完整有效性；有效计划核对
逐项保真，无效回答整份标作rule-processed。单独列出原始bullet未包含但最终
计划新增的对空动作，既不把它们算模型选择，也不把计划数当单位产量。
`test_air_plan_attribution.py` 两项CPU测试通过，含归因与对齐失败拒绝。
整局累计reward对全部日志行求和；实际产出/损失后续复用既有
`audit_r4_reactor_pair.py` 的回放分析，不新增在线事件机制。

## 单局结果

11:13只读核验 `.done`、`result.json`、编排退出码均完成/exit0，自然Defeat，
无截断、无PPO；tmux已退出，GPU6无计算进程（4MiB）。终局后有Ray worker
收到SIGTERM的退出警告，未改变已完成的自然游戏、回放及主进程exit0记录。
没有追加或重试实验。[结果、原始回答归因与产物哈希](r1-result.json)。

| 同seed SpB单局 | 不加对空备选 | 新增对空备选 |
| --- | --- | --- |
| 自然胜负 | 负 | 负 |
| 整局环境reward | -49.500 | -47.599 |
| 完整有效回答 | 27/37（73.0%） | 18/23（78.3%） |
| 游戏时长 | 17:43.0 | 17:40.5 |
| Starport开工/建成 | 0/0 | 1/1 |
| Viking实际出生/死亡 | 0/0 | 12/12 |
| Marine实际出生/死亡 | 236/236 | 173/173 |
| SiegeTank实际出生/死亡 | 20/20 | 19/19 |

23条消费回答与原始日志逐条对齐通过，18条合法生产计划全部保真。Starport的
1条计划由首条无效回答后的规则新增；Viking计划共14条，其中9条来自合法模型
回答，5条来自无效回答规则处理。后5条中3条不在原始bullet里，2条在超长无效
回答里出现过，不能把5条都称为凭空补出，更不能算合法模型选择。
trace记录14条Viking生产命令；回放确认实际出生12架。未把每架出生精确回溯
到某条模型/规则计划，不按9/5直接分摊实际产出。

回放时间：10:35.7星港开工，11:11.4建成，11:43.8首架Viking出生；存活峰值
仅4架。12架最终全部死亡，最后击杀者为8次Battlecruiser、4次敌VikingFighter。
14:36后多次兵力归零，随后单架出生又死亡；例如17:13.3出生、17:18.3死亡。
该现象说明生产链存在，但没有形成持续对空兵力。尚未核对每次自动攻击/侦察
指令与Viking死亡的对应关系，不能就此确定是哪一条微操规则导致。

结论：**新增备选到模型回答、执行、实际产出的链路通过本局验证；胜率改善
没有证据，正式采用仍证据不足。** 星港依靠规则启动，不能当成纯模型能力。
reward略高、有效率略高不替代胜负验收；下一项最有价值的是只读核对这12架
Viking的攻击/侦察指令及交战位置，不直接再改prompt或开长训。

离线核验复用既有 `attribute_plans.py` 与 `audit_r4_reactor_pair.py`；两局
console和replay哈希均与原result吻合。历史临时解码环境已不存在，故只在本地
`/private/tmp/sc2-air-results.HBI61m/decoder` 恢复原记录的
s2protocol5.0.16.97563.0 / mpyq0.2.5，未改服务器依赖。日志快照及完整解码
报告位于 `/private/tmp/sc2-air-results.HBI61m`；源码无新增行为改动。

12架Viking的后续只读核对见 [指令、交战位置与损失](viking-order-audit.md)：
确认第1–3架受到驻防与波次集结指令冲突，第8、9架被派侦察，最后3架在基地
附近出生后5–8秒死亡。不是全部由侦察造成；本次未改游戏行为、未追加实验。

后续批准的 [驻防编队占用修复](../20260908-garrison-wave-ownership-v1/README.md)
单独记录；本页已完成r1的冻结源码与结果不因此改变。
