# 第二阶段可选对空路线：最小候选

2026-09-13 22:26北京时间；状态：**r1两组完成、退出0，均1/4胜，没有胜率改善**。
启动快照保留在[r1-launch.json](r1-launch.json)，结果见[r1-result.json](r1-result.json)。
基线runtime `e8d8de164ed5312c33f9905613176503a9d1832c`；开发父提交
`ffc8fca55815c70b8fc680ce31cfc6f032547d01`仅增加既有审计，两者核心源码无diff。
因为改变阶段策略约束，按约定另建`codex/experiment-phase2-air-options`，
工作树`/private/tmp/sc2-phase2-air-options-20260913`；旧分支、runtime和结果未改。

## 为什么只改这里

[三个败局回放](../20260911-execution-failure-feedback-v1/multiseed-failure-boundary.md)
已经证实Marine实际产出66/66/68，绝大多数战死；相同计划也有胜局。
04:09附近模型已观察到敌方两个Starport，但第二阶段要求推迟科技且不提供对空路线。
本次检验第二阶段是否过度限制选择，不预称已定位实现bug，不保证新增选择一定有用。

## A/B定义与实际修改

A保持原第二阶段。B仅在原`SC2Strategist.generate_strategy`返回第二阶段时，
通过显式实验hook追加既有末阶段的`starport=1, vikingfighter=8`可选额度，并将
该阶段“Marine优先、推迟科技”的instruction改成根据观测在Marine和可用对空路线
间选择。格式prompt、one-shot、attack_instruction、阶段时间和其他阶段不变。
这是**阶段策略约束对照**，包含指令文字和待办联动，不称为只改一个参数。

最小代码为[phase2_air_options.py](phase2_air_options.py)。默认入口不导入它。
候选没有改共享`agent_system`、`scripts`、`configs`、`verl`或`gigpo`，
没有复制整个旧版本，没有改Commander、validator、解析器、Executor或模型算子。
挂载需显式调用`phase2_air_options.install()`，仅接受marine_pressure及已启用air-options；
最初仅测试隔离类挂载和CPU协议链；本次补接后，服务器真实Commander导入一致性
检查也通过，A组已有四局阶段记录。B组实局候选生效仍待它启动后核验。

为避免持久阶段表累计目标从1个Starport变成2个，候选不改模板，复用现有
`_reconcile_completed_macro`按原末阶段目标和当前己方建筑算剩余额度，作用于本次
返回的第二阶段副本。已拥有1个Starport时不再提供额外Starport，末阶段仍按原规则。

**额度边界（已获用户确认）**：Viking原本是各阶段生产配额，不按现存数量扣减。
本候选保留末阶段原8架，因此第二阶段新增的8架是额外可选生产，并非从末阶段搬走。
它不是“全局数量上限不变、只提前时间”的实验。实现跨阶段共享8架上限需要新增
计账语义，本轮没有引入。若不接受额外早期额度，先重新确定方案，不自行上卡。

保持原来的前置过滤，因此新路线不是让模型一次回答跳过科技链：

| 己方观测 | A合法 | B新增合法 | 仍不允许 |
| --- | --- | --- | --- |
| 实际04:09，4兵营、无Factory | 原Marine等；Refinery本来已由规则提供 | Factory | Starport、Viking |
| CPU合成：已有Factory | 原候选 | Starport | Viking |
| CPU合成：已有Factory和Starport | 原候选 | Viking，最多8 | 超过配额、其他非法名称 |

合成已有建筑只验证过滤/解析接口，不代表实局真的已建成。Factory仍需实际气体，
阶段调度不变；若模型先选不可实际开工的Factory、持续选Marine或在长队列内不再
规划，都如实记为结果，不能临时放宽前置条件、强制补科技或再加定时规划。

合法旧计划仍原样影响Executor。**无效回答补齐规则不改**；扩大候选可能改变
补齐输出。这部分必须单独归因，不能把规则产出的科技称为模型学会反制。
本轮实际入口处的三种无效探针仍产生相同6动作fallback；在后续已具备前置的状态
可能不同，不能据开局CPU结果宣称所有fallback输出完全相同。

## 已完成CPU验证

9项新回归，加既有Planner/原始BO段/有效回答直通/配额prompt/执行失败反馈，
共**63项通过，4.284秒，0跳过，退出0**。24项system-v1源码hash全部匹配。
真实A4第二轮四份观测，分别注入历史回答及四种手写探针：共20个输入/回答组合，
两臂40次CPU协议调用；不含新模型生成、SC2命令提交或PPO。

历史Marine42等60动作回答在两臂均合法，最终队列hash相同。
手写`- factory`在A无效、B合法且队列仅Factory；手写Starport/Viking在两臂仍无效。
另外通过合成己方Factory/Starport检验逐级解锁、合法Marine不强制补科技、
后期Starport目标不翻倍、hook不影响其他阶段和默认模板。
摘要[CPU结果](cpu-results.json)只保存来源hash/行号、判定和数量/hash，不含完整原始模型回答。

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:tests python3 -m unittest \
  test_phase2_air_options test_planner_air_options test_original_bo_segment \
  test_valid_bo_passthrough test_quota_prompt test_execution_failure_feedback -v
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:tests python3 -B \
  experiments/records/20260913-phase2-air-options-v1/audit_cpu.py \
  <原A4八种子评测小型快照> <新的结果JSON路径>
shasum -a 256 -c configs/system-v1.sha256
```

## 已批准纯评测与预算

使用同一A4 SpB checkpoint，Level6/Harder、Flat128/Terran/Air，两臂各4局，
相同实际seed依次为42613、17089、3236598124、4018390961（三个败局和同计划胜局参照）。
从开局自然生成，不注入历史答案。计划43 GPU0–3串行A/B，最多45分钟，0次PPO，
无新checkpoint、无大权重复制、不自动重试或续训；四卡匹配现有四分片checkpoint。
不能直接挪用上轮预算，也不能将八局不同种子的总胜负当成四次同场景重复。

沿用A4评测的192token、T0.4/top_p0.9、原15条/xN协议、feedback/history/shaping off、
FP32参数及温度、BF16 autocast、cache on、SWA dropout0、现有g2资产和开局Planner接管。
前置过滤、补齐、微操、自动战斗、reward均冻结。两组初始rank采样RNG须匹配；
干预后输入和轨迹可不同，不要求后续随机数消费位置仍相同。

先核实前两阶段覆盖、实际最终query差异和checkpoint加载，再比较模型有效选择、
执行队列、真实科技/单位产出、损兵及工人死亡、Planner调用、胜负和整局reward。
规则补齐与模型选择分开；单凭有Viking或胜率变化不能宣称模型学会，更不是RL验证。
若选项未进入query、模型不选、选后没建成或建成仍失败，分别记录对应边界，
不临时增加第二项机制。小样本只决定是否值得后续短RL，不直接晋级96步。

用户随后明确批准额外早期额度、以上四卡串行45分钟预算及推送到私有
`CeraveNew/new_SCII`。上轮推送被阻止的记录保留，本轮依据新增授权进行备份。

入口为`run_pair_node43.sh`，复用既有`run_feedback_eval_node47.sh`的A4恢复和
纯评测路径，通过显式`FEEDBACK_PHASE2_AIR=A|B`切换候选。两组均由
`phase2_eval.install`记录每次阶段返回的待办及instruction/template hash，
hook挂载到`agents.commander`实际使用的Strategist类。默认入口不启用候选。
这是实验接线工作，不把hook自身的修正算作项目原有bug修复。

在已核验空闲的43节点、独立clean部署中启动：

```bash
SC2_PHASE2_AIR_AUTHORIZED=1 SOURCE_SHA=<已备份commit> \
 EVAL_CONTROL_DIR=/root/new-scii-execution-feedback-20260911/phase2-air-pair-r1 \
 bash experiments/records/20260913-phase2-air-options-v1/run_pair_node43.sh
```

两臂RUN_ID为`20260913-phase2-air-options-v1-spb2-A4-l6-{A,B}-eval4-4gpu-r1`。
结果保存到43既有local mirror下的同名experiment，不修改历史部署或结果。
启动器核验A4父run `.done`、四分片大小以及实际恢复step4；结束时核验
`val_only`、0更新、每臂4个自然结局及实际种子顺序，不能以目录存在冒充加载成功。
预算由两组共享，失败即停止，不自动重试。后续报告须区分已启动和完成结果。

入口补接后，使用本机已有 bundled Python（含numpy）执行以上六组回归，加
`test_phase2_eval test_feedback_learning_eval`，共77项通过，5.140秒、退出0。
初次系统Python运行有5项因缺numpy失败，未修改或安装依赖；新入口的3项当时即通过。
A/B完整effective override对比只有`SC2_PHASE2_AIR_AB`不同，均为A4/step4、
4局、val_only、0更新、192token；现有默认入口回归通过，24项源码hash仍匹配。

## r1结果

43 GPU0–3串行，22:02:30–22:26:50，共24分20秒。两组实际恢复A4/step4，
种子顺序和初始四rank CPU/CUDA采样hash一致；除了候选开关及run/temp路径，
effective配置相同。两组`.done=result.json`、退出0，均四局自然结局、无PPO。

| 指标 | A原第二阶段 | B开放可选对空 |
| --- | --- | --- |
| 胜率 | 1/4 | 1/4 |
| 整局累计reward均值 | -31.93325 | -31.958 |
| 真实调用中语义有效 | 9/10 | 11/11 |
| 第二答选择Factory | 0/4 | 4/4 |

四个种子的胜负逐一相同。有效率只统计`policy_action_applied`的真实回答，
不把其他局等待期间对已终止环境生成的回答当成额外Planner调用；reward也不重复
累加已终止环境返回的旧值。结束批次的`val/sc2_action_valid=1`不是整轮有效率。

B确实把新增选择传给模型并进入Executor队列，没有由规则替模型强行添加Factory。
但三局仍先列Marine42等60项，把Factory排到第61项；其中两局在第54/49项
就自然终局，没轮到Factory。另一局以及短计划那局的首个Factory明确超时跳过，
下一观测分别为12:23/08:11、均40气。不能把“进入队列”解释成“科技链完成”。

因此仅支持**新增选择能影响合法模型计划**，不支持该候选改善游戏效果或可晋级RL。
后续Factory尝试和开工历史尚未逐项核完；唯一优先只读问题是这两次超时的
资源与执行过程，不在本轮顺手重排、强制补兵或启动新实验。
