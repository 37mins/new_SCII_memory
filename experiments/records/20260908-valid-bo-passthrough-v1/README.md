# 有效 BO 回答保真：单项离线对照

2026-09-08，用户批准单独验证，不预设取消补兵规则会提高胜率。
状态：completed（离线及单局在线接口验收）；游戏效果未见改善。单局结果见文末，
离线部分无模型加载、SC2、PPO 或 GPU。
离线 RUN_ID：`20260908-valid-bo-passthrough-v1-cpu-qwen-spb-110-r2`。

基于 `719e6b6e0d679268cec90393c1c67dbe903b31f6`；其游戏源码与已完成
Level6/g2 配对游戏的 `c15d02b08085ddfcf20cff5517d10ceae9fa2af0` 相同。
原始 deployment、结果、用户主工作树不修改。因涉及回答到执行计划的语义，
候选隔离于 `codex/experiment-valid-bo-passthrough`，不合入正式默认版本。

唯一机制变化位于 `SC2Planner.generate_plan`：显式设置
`SC2_PRESERVE_VALID_BO=1`，且现有严格 legacy 整份回答验证为有效时，
跳过 `constrain_bo_to_todo` 的额外补齐、数量限制与重排，保留解析出的生产动作。
未设置时仍是原行为；无效回答沿用原修复路径，不把有效前缀冒充有效回答。
有效性直接使用已有 manager → socket → `get_action` 传来的
`plan_format_valid` / `self.correct_format`，不新增另一套 validator。
非 legacy 模式不启用此候选。
上卡须显式在原 launcher 的附加 Hydra 参数中传入字符串
`+ray_init.runtime_env.env_vars.SC2_PRESERVE_VALID_BO='1'`，使 Ray worker 接收该变量；
只设置本机 shell 环境变量不能当作远端 worker 已启用的证据。实际部署见文末。

prompt、解析器、候选名单、reward、生成预算、固定 g2、自动战斗、微操、模型和
训练配置不变。攻击 BO 的既有抽取与后置逻辑、Commander 的独立开局 Marine
补全仍保留：本候选不等于整个系统完全由模型控制。合法名称也不保证命令立即成功
或单位真正产出。规则可能减少短回答的产兵量，或其生产组合本来有益，须以游戏
对照检验，不能用离线保真率宣布胜率提升。

## 验证方法

输入为既有两局的 `prompt_response.jsonl`、`console.log`、`result.json`、
`effective_config.json`，不重新生成回答；所有文件记录 SHA256。
两 RUN 共用前缀 `20260906-integrated-fixes-v1-r4-reactor-pair-slots-`，后缀：

- `qwen2507-l6-g2-s2621640384-r2`
- `spb2-l6-g2-s2621640384-r1`

先用关闭候选的原路径逐条复现实际日志里的最终 BO；复现不符就停止，不做候选
结论。再用同一 observation、阶段指令、待办、原始回答，仅打开该开关：
逐字核对 query 不变，现有 validator 与历史有效标记一致；有效回答的生产序列、
数量不再被额外改写，无效回答的最终 BO 与 A 完全相同。
另核对 Commander 保底与攻击后置仍存在，单独披露，不暗中修改。

```bash
python3 -B -m unittest discover -s tests -p 'test_valid_bo_passthrough.py' -v
python3 -B experiments/records/20260908-valid-bo-passthrough-v1/audit.py \
  /path/to/qwen-artifacts /path/to/spb-artifacts
```

这不是新轨迹仿真：候选输出改变后的新观测未知。通过后只能说明局部接口保真，
下一项可申请同 seed、同模型的一局候选纯评测，与已完成 A 组比较，无需重训或重跑
原 A。单 seed 只能提供机制/退化信号，不支持总体胜率结论；GPU 预算另确认。

开发回归实录：`python3 -B -m unittest discover -s tests -p
'test_valid_bo_passthrough.py' -v` 7项通过；原 `test_r6_planner_interface.py`
5项通过。包括6个Marine不再被换成坦克、短回答、数量与顺序、无效回答不变、
攻击后置不变、非legacy不变，以及仍保留的Commander独立Marine保底。

保留失败：CPU r1 / `d33d2de` / exit1，在读取第一条历史 BO 时因本地 Python3.9
`ast.literal_eval` 不接受前导空格而停止，尚未比较候选。只修离线日志解析的
`.strip()`，不改候选机制，r2重新验证。这是诊断脚本兼容性返工，不计项目bug修复。

## r2 结果

冻结 SHA `30b5949100dd963e5b0f4a18289b2827026d0946`，本地 Python3.9、exit0，
完整命令使用上述 audit.py 和 `/private/tmp/sc2-valid-bo-inputs.KEPI42/qwen`、
`/private/tmp/sc2-valid-bo-inputs.KEPI42/spb` 两目录。
[逐条回答、A/B计划和输入哈希](cpu-r2-results.json)保留全部110条，不筛掉无效样本。

| 验收 | Qwen | SpB |
| --- | --- | --- |
| A与历史最终BO完全一致 | 54/54 | 56/56 |
| A/B最终query逐字一致 | 54/54 | 56/56 |
| 有效回答生产序列在B保留数量、顺序 | 49/49 | 43/43 |
| 无效回答处理A/B完全一致 | 5/5 | 13/13 |
| 有效回答在A/B间计划发生变化 | 49/49 | 43/43 |

仅统计这些固定历史输入下的**有效回答所对应生产计划**（不是新游戏产量）：

| 计划动作 | Qwen A→B | SpB A→B |
| --- | --- | --- |
| Marine | 144→52 | 129→219 |
| Siege Tank | 27→31 | 29→10 |
| Marauder | 15→24 | 10→18 |

SpB 14:13（step_index22）的有效原回答6条Marine，A变为3Marine+2Tank+1MULE，
B保持6Marine。相反，Qwen经常只要求少量Marine，取消规则补齐后Marine计划大幅减少。
这支持用户对“可能更差”的担忧有具体机制依据，但不是胜率下降证据。SpB得到更多
Marine也不能宣布更强；新轨迹的决策次数、观测、资源和产出都会变化。

这92条有效候选计划均含Marine，因此现有独立开局Marine guard不会再往这些计划
里加Marine；专门的反例测试仍证实guard对其他合法纯Tank回答会添加Marine。
攻击后置与自动波次照旧。不得把本次通过称为完整Planner/Executor控制权修复。

离线阶段结论：**局部接口保真验收通过，是否采用为正式策略仍证据不足。**
当时建议的SpB同seed单局候选随后获批并执行，结果见下；不合入正式默认版本。

另：原格式奖励/传输回归 `test_r6_format_reward.py` 6项通过，连同新7项和原接口5项
共18项；冻结源码manifest全部校验通过。推送尝试被权限审查拦截，尚未外传；本地
commit和记录保留，待用户确认向私有 `CeraveNew/new_SCII` 候选分支推送的权限。

## SpB 同 seed 单局候选（2026-09-08 获批）

状态 completed（北京时间10:07:38自然败局，exit0，无PPO）。仅补候选，
不重复原 SpB 对照、不运行 Qwen 或 PPO，不自动重试。
RUN_ID `20260908-valid-bo-passthrough-v1-spb2-l6-g2-s2621640384-r1`，
产物根 `/mnt/data0601/yangjinyue/new_SCII/experiments/20260908-valid-bo-passthrough-v1/runs`。
预算单卡最多30分钟。09:42北京时间资源快照：44全卡已有任务，48 GPU4空闲；
使用48已部署的相同版本Python overlay，无新增依赖部署或权重复制。
独立源码/备份根 `/root/new-scii-valid-bo-20260908`，按提交命名、拒绝覆盖。
GitHub推送仍未获新的明确许可，不重试；部署前在服务器保留Git bundle及哈希，
源码由bundle恢复到clean commit，原c15d02b deployment和所有旧结果保持原样。

调用 `bash experiments/records/20260908-valid-bo-passthrough-v1/run_spb_node48.sh`。
该薄入口复用原 `scripts/train_sc2.sh` 和逐字未改的 `r4_pair_seed.install`。
相对已完成SpB A组，唯一游戏机制变化是Ray worker环境中的字符串
`SC2_PRESERVE_VALID_BO='1'`；新节点、依赖路径、输出路径是基础设施差异，
需核对依赖版本及固定BO阶段轨迹，不能据此假定SC2逐帧确定性。
候选游戏源码与 `f8c5f042d8586a1c055f9f3c7fdba26dfc686c53` 相同，
最终启动SHA由manifest和编排记录保留。部署/CPU配置核验不代表游戏已启动。

模型原始 `Panyuqi/SpikingBrain-2.0-instruct`，同一共享snapshot/manifest；
实际 `swa_dropout=0`、attention_dropout=0、eager、HF生成cache开启，
FP32 FSDP参数、FP32温度归一化、BF16 autocast，greedy、4096/96预算。
Level6/Harder、Flat128、Terran vs Terran Air、固定g2、真实seed2621640384，
history0/off、legacy 6/6、shaping off、自然终止、旧微操/20-10自动波次不变。
不加载项目SFT或RL checkpoint，不更改无效回答fallback或Commander独立Marine guard。

先实际解析Hydra配置并与A组逐项核对，再在受控tmux启动一局；只有模型加载和
SC2真实开局后才记running。验收核对实际seed、原始回答→最终BO→执行命令，
特别是有效全Marine回答是否保持原数量，另报告自然胜负、整局累计reward、
决策/有效率、挂件完成、产出/损失和回放。若没有出现全Marine有效回答应明确未覆盖，
不能凭离线回放代替在线证据。胜率变差亦保留；一局不能证明总体效果或架构优势。

### 启动核验

实际冻结SHA `019ca9942a586797621e301db77b988b65dfa7bd`，clean；源码
`/root/new-scii-valid-bo-20260908/code-019ca99`，tmux `sc2_spb_valid_bo_r1`。
Git bundle服务器备份SHA256
`fd3c185e33b7a1a4ef4999bcd075e8f931a8f9549716251609d4be2e7233cc54`。
09:49:21启动受控后台任务，manifest开始时间09:50:08；09:53:09已推进到游戏03:14，
GPU4归属本run的 `ray::WorkerDict` PID2423376，约40.6GB显存。
30分钟墙钟从外层启动计算，最晚约10:19:21停止；不是允许超过30分钟再跑一局。
预计约10:10至10:15出结果，受实际游戏时长影响。

本次重新执行18项轻量回归全部通过，`bash -n` 和源码manifest通过。
服务器 `--cfg job` 实际Hydra解析exit0（不是仅dry-run打印命令）。
实际运行effective config与旧SpB对照对比：去除run/源码/依赖基础路径差异后，
唯一差异为worker环境 `SC2_PRESERVE_VALID_BO` 从缺省变成字符串`'1'`。
实际 `evaluation_seeds.jsonl` 只有
`{"base_seed":1000,"reset_index":0,"seeds":[2621640384]}`。
运行日志确认`swa_dropout:0.0`、`use_cache:true`；模式evaluation，单卡单局，
resume disabled。模型manifest和g2 BO哈希与旧SpB对照一致。
此快照仍在首次Planner前，尚不能宣布全Marine在线保真或游戏效果通过。

### 单局完成结果

本run的 `result.json`、`.done` 与编排 `.done` 均确认正常结束；自然Defeat，
未截断，无PPO更新。10:12:16只读核查tmux已退出、GPU4无计算进程（4MiB）。
没有追加游戏或自动重试。[结果与核验口径](spb-r1-result-summary.json)。

| 同seed SpB | 原后处理A | 保留有效BO B |
| --- | --- | --- |
| 自然胜负 | 0/1胜 | 0/1胜 |
| 整局累计环境reward | -49.5 | -49.5 |
| 实际消费模型回答 | 56 | 37 |
| 完整有效回答 | 43/56（76.8%） | 27/37（73.0%） |

逐条将37条原始回答对齐到实际 `BO ori / BO` 日志：27条有效回答的生产序列
全部保留数量与顺序，0条不符；其中18条是6 Marine，18条最终均保留6 Marine。
例如游戏10:50、step_index3，原回答6行 `- marine`，最终BO仍为6个marine，
没有改成3Marine+2Tank+1MULE。本局没有Commander开局Marine guard触发日志。
这证明本候选的有效回答后处理保真在真实游戏生效，不等于全部命令/单位都已完成。

游戏日志最后采样为17:40，旧A回放时长22:19.3；候选精确回放时长和产出/损失
尚未解码，不把日志末采样冒充回放终点。两局首个完整query哈希相同，但不同节点
及同seed不构成全过程逐帧确定性的证明。单局更早失败的信号不能归结为唯一机制
必然有害；也不能因保真通过而宣称游戏能力提高。

结论：**在线接口保真通过；本seed胜负/reward未改善，正式采用仍证据不足。**
保留候选、原版与完整回放，不改其他模块、不新增训练。下一步如需判断兵力变化，
先分析已有回放的生产/损失，无需为此再开GPU局。
