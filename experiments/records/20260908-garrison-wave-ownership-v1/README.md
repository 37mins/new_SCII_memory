# 驻防不重新分配进攻编队成员

2026-09-08，CPU回归通过；同seed单局已自然获胜，但评测收尾失败（exit1）。
驻防编队冲突在本局消失，不等于整条评测或RL链路验收通过。
只修驻防选兵，不同时调整侦察、集结点或波次数量。
后续[纯评测收尾r2](evaluation-only-finish.md)已完成同seed单局验收，exit0；
此处r1的exit1和失败记录不变，不将r2称为RL或学习效果通过。

## 来源与唯一改动

基于已完成的SpB对空备选，冻结runtime为
3adaaf70ac879a7f4d7bdecbfb8be60b75bf4fa6，修改前记录提交为
bce863893a07628eb75c097112baac38bce65523，两者游戏源码相同。
直接证据见 [12架Viking审计](../20260908-planner-air-options-v1/viking-order-audit.md)：
第1–3架已经进入进攻波次，仍分别被驻防拉回5、2、1次。

沿用已有隔离工作树和分支codex/experiment-planner-air-options，不另建分支；
旧commit、服务器deployment、已完成run及失败记录不变。新增原子commit保存候选，
回滚使用上述修改前SHA及原配置，而不是手工改回。

唯一行为修改位于
agent_system/environments/env_package/sc2_online_restored/SC2ArenaDyStrategy/players/script_player.py
的manage_garrison：从现有total_attack_groups收集unit_tags，
驻防候选增加“单位tag不在此集合中”的条件。新增5行（含注释），不新增开关、
持久状态、框架或候选fallback。复用同文件自动进攻/侦察已有的编队占用概念，
不是新设计一套指挥算法。

GATHERING及ATTACKING等仍在编队登记中的成员，即使当前idle也不参与驻防。
成员从集合移除或编队删除后，下次原定驻防检查自然恢复资格。
无编队的空闲单位继续按原来的基地、半径、配额和目的地分配；
不改变波次状态机或删除/补兵条件。
排除编队成员后，参与原配额计算的候选数会减少，因此其他单位的具体分配可能
随之变化；这是本次选兵过滤的直接结果，不是另外修改配额或目的地算法。

侦察选兵、自动防御、集结点、波次数量、最低人数、检查周期、模型、prompt、
BO、reward、cache、精度和训练配置均保持不变。未顺手处理无编队成员在重叠
基地防区内重复分配的问题，也未新增对侦察单位的驻防排除。
configs/system-v1.sha256只更新该文件的哈希登记，不改变任何配置值。
新文件SHA256：89ed4006cf35af238b0962409bd5a183780d8676af5a49af35422d3077030069。

## 验证

本地Python3.9.6、burnysc2 7.1.3（与服务器相同库版本），调用真实
manage_garrison方法和库中的Units/Point2，SC2观测与命令提交使用测试替身；
没有加载模型或运行SC2。这里只证明函数行为，不冒充整局回放重放。

~~~bash
python -B -m unittest discover -s tests -p 'test_garrison_wave_ownership.py' -v
~~~

同一测试在修改前后均实际运行：
[修改前](cpu-before.txt)8个测试方法产生6条失败记录（包括释放方式的2个子用例），
均因已入编队的单位被下发Move；[修改后](cpu-after.txt)8项全部通过。
覆盖集结中、攻击中、多个编队/兵种、连续检查、成员或编队释放，以及普通驻防、
敌情/工人/忙碌/距离过滤和原侦察处理不变。

相关回归另运行49项：test_pending_addon_placement、test_barracks_production_slots、
test_planner_air_options、test_valid_bo_passthrough、test_r6_planner_interface、
test_r6_format_reward、test_empty_bo_frame，全部通过。入口及冻结源码测试
test_system_release另26项通过。合计83项通过，不代表整个训练测试集。
原始输出及命令见 [相关回归](cpu-related.txt) 和 [入口/冻结校验](cpu-release.txt)。
shasum -a 256 -c configs/system-v1.sha256、git diff --check均通过。
AST对比确认ScriptPlayer除manage_garrison外所有方法与修改前一致；
agent_system/scripts/configs范围只有该方法及其哈希登记变化。

## CPU阶段的结论（在线结果见末节）

仅凭上述CPU结果不能声称Viking存活、集结或Level6胜率改善；侦察损失、单独防御、基地失守
仍是独立问题。用户随后授权用新RUN_ID跑同seed2621640384的SpB
Level6/g2单局，与原对空备选比较，其他条件冻结。优先核对入组后是否还收到
驻防Move、波次是否完成集结、Viking存活/战斗位置，再看自然胜负。
不复用旧run目录，不从当前修改树直接调用含旧RUN_ID的历史入口。

本次只提交本地候选；此前GitHub推送被拒绝，不重试推送，未称已合入main或上线。

## 同seed单局验收

用户要求“跑实验看看修改效果，然后继续排查”，据此只安排一局SpB Level6/g2
纯评测，最多1800秒（终止宽限30秒），无PPO、无自动重试，不重跑已完成对照。
RUN_ID：20260908-garrison-wave-ownership-v1-spb2-l6-g2-s2621640384-r1。
对照为air-options-v1同seed的r1。问题是驻防过滤能否消除波次内Move覆盖，并让
部队实际完成集结、存活；自然胜负和整局reward同时报告，但一局不估计总体胜率。

12:43北京时间只读快照：共享42/43/46/47/48均无整张空卡；44 GPU1无计算进程、
0MiB。44已有原Python依赖，实际导入Torch2.7.1+cu128、Transformers4.55.2、
Ray2.48.0、burnysc2 7.1.3，与原依赖版本匹配。其他任务不触碰，启动前再查GPU。
部署到新目录/root/new-scii-garrison-wave-20260908/code-<SHA>，
保存私有Git bundle；不改旧部署。共享模型/FLA/MoBA/SC2只读复用，不下载模型、
不复制或改共享依赖。44使用现有/usr/include/python3.12编译头。

模型、Level6/Harder、Terran Air对手、Flat128地图、canonical g2、实际游戏种子
2621640384、greedy、原始SpB无项目SFT/RL、SWA dropout0、eager/FP32参数与温度
归一化/BF16 autocast、生成cache、history0/off、6BO/96tokens、reward及旧micro
全部冻结。新增行为只有已提交b24960e中的驻防过滤。节点/GPU及依赖目录位置变化
显式记录，不能称为同节点硬件对照；有效配置排除基础设施路径后必须与对照一致。

~~~bash
bash experiments/records/20260908-garrison-wave-ownership-v1/run_spb_node44.sh
~~~

输出在共享new_SCII/experiments/20260908-garrison-wave-ownership-v1/runs下唯一目录；
Ray临时目录/dev/shm/sc2_garrison_spb_20260908_r1；
后台tmux为sc2_spb_garrison_r1。复核配置、源码哈希、模型实际加载、SC2推进及种子
后才记running。结束保留.done、结果、回答、trace和回放，再离线核对编队指令。
未出现的新问题不凭感觉修改；本局期间后续排查只读。

## 实际运行与结果

冻结SHA `bba87e3a2e6d136d0c9facd031f827f06867faac`，clean，部署
`/root/new-scii-garrison-wave-20260908/code-bba87e3`；44 GPU1。
12:53:31启动，12:59:53结束，约6分22秒，未到30分钟预算，无重试。
服务器私有bundle SHA256为
`a45a695f71ad0b65a39987889a35423a2b5e3bb4c664cec87466f6bab55f39f6`。
模型实际加载为SSESWAMoBAForCausalLM，swa_dropout0、cache true；SC2自然结束。
13:02只读确认本次tmux已退出，未停止其他任务；GPU上已有其他进程，不能称仍空闲。

两局实际effective_config排除源码、输出、依赖目录和Ray目录后完全一致；
model revision、model manifest、g2 BO和依赖版本逐项一致，manifest列出的运行源码
只有ScriptPlayer哈希变化。44/48是不同节点，未宣称全部Python环境逐字相同。
预检曾将种子文件名误写为eval_seeds，CPU分析退出1；改读既有
evaluation_seeds后核验成功，未启动额外游戏或修改运行代码。Hydra CPU解析exit0。
实际种子文件均只有`base_seed=1000, reset_index=0, seeds=[2621640384]`。

| 同seed单局 | 修改前对空备选 | 驻防过滤候选 |
| --- | --- | --- |
| 游戏自然结果 | Defeat | Victory |
| 游戏时长 | 17:40.5 | 09:17.1 |
| 整局环境累计reward | -47.599 | +20.000 |
| 实际消费Planner回答 | 23 | 0 |
| 完整有效回答 | 18/23 | N/A（无决策，不是0%） |
| Marine出生/死亡 | 173/173 | 97/43 |
| Tank出生/死亡 | 19/19 | 3/0 |
| Viking出生/死亡 | 12/12 | 0/0 |
| 进入ATTACKING的波次 | 6/16 | 8/9 |
| 编队内单位被驻防Move覆盖（单位×命令） | 179 | 0 |
| 主进程退出码 | 0 | 1（评测空策略批次异常） |

候选在固定g2耗尽前获胜，没有调用Planner.generate_plan/get_action；终局仍由
外层模型生成了一条未消费回答，policy_action_applied=False，已正确排除。
这条回答开头是`Based on the provided game state, here's a breakdown...`，不是
可执行BO，但并未进入执行器，不能算模型动作失败或规划退化。
本局无PPO，result中training_updates=0。metrics为空导致metrics_finite=false，
不是观察到了NaN梯度。`.done`和result仍保留failed，不美化为流程completed。

自然胜局有回放和episode_outcomes双重记录；候选结束时仍有54 Marine、3 Tank、
3 Marauder、45工人。没有产出Viking，因此本局未覆盖空军存活、侦察损失；
收益发生在更早的地面编队阶段，不能当成SpB规划提升或稳定胜率结论。

## 驻防效果的直接证据

两份回放前203条我方命令（时间、选择单位、能力、目标、序号）完全相同。
05:01.6成立的首波具有相同20个Marine tag和相同集结点(29.323,74.800)。
首个分岔在05:10.9：旧版sequence207把已入首波的Marine82575361 Move至
(38.563,52.872)驻防点；候选没有该命令。旧版首波从未完成集结，候选05:15.2
进入ATTACKING。候选首波后来仍全灭，不能声称取消覆盖就消除了战斗损失。

旧版全局179条覆盖分为Marine145、Tank15、Marauder11、Viking8，与上轮12架
Viking审计中的8条一致。按候选结束的09:17.1统一截断，旧版仍有130条、
候选0条。候选普通驻防仍有113条单位命令，并非把驻防整体关闭。
波次生命周期按实际成立/删除日志恢复，解散后驻防不算冲突；第9波在胜局结束
前3.4秒才建立，不能把它未进入ATTACKING直接定性失败。

[结果与产物哈希](r1-result.json)、[两局命令与编队证据](r1-wave-orders.json)。
回放SHA256：`5b173118a7d2b0cc1a567854e048a511a859de4ff3e345737ac19c945534037b`；
console SHA256：`1456c3154b47d61c68ce8460c16235c28f2d2881f6172b788d8686084197527c`。
原始产物在上文RUN_ID对应共享目录，本地回放副本在
`/private/tmp/sc2-garrison-results-20260908/sc2-logs/default_player/Flat128_Harder_Air/Qwen3-30B/2026-09-08_04-55-35/replay.SC2Replay`。
Qwen3-30B是历史日志目录标签，不是实际模型。

离线分析复用原s2protocol/SelectionTracker命令解码，不运行模型或SC2：

~~~bash
python -B experiments/records/20260908-garrison-wave-ownership-v1/audit_wave_orders.py <旧run> <新run>
python -B experiments/records/20260906-integrated-fixes-v1/audit_r4_reactor_pair.py <新run>
~~~

解码器按真实67帧周期和驻防目的地几何匹配，再核对编队有效时间；不是看到Move
就称驻防。新分析脚本处理了缺失最后击杀者、burnysc2 rounded为floor、同帧维护
重复给编队子集下令三处分析适配，这些不算项目运行bug修复。原对照独立恢复出
12架Viking、14条生产命令、8条Viking编队驻防覆盖；全部生产命令与trace对齐。
旧版1条未匹配目的地的周期Move没有计为驻防；候选未匹配数0。

## r1完成时的只读排查与下一项

已确认评测边界问题，不是这5行游戏修改新增：
`ray_trainer.py:753`以is_train=False调用multi_turn_loop；后者仍无条件进入
`rollout_loop.py:670`的gather_rollout_data，最终第372行对空effective_batch抛
`No consumed policy actions in this batch; refusing a PPO update`。
该检查来自2026-09-05提交7fdab020；训练拒绝伪决策是正确保护，但没有区分
“本次纯评测已经自然结束”。本次真实日志已复现，既有CPU终局归因测试10项也
全部通过并明确复现这个全空批次异常，见[cpu记录](cpu-empty-policy-audit.txt)。
因此不是为了得胜强行删除检查，而是下次应单独处理评测结果与策略样本的边界。

唯一优先建议：纯评测允许无策略样本，正常汇总自然胜负和整局reward；训练端
继续拒绝空策略批次，不制造回答、不重复计权、不自动重采样。先做CPU覆盖
全无决策胜/负、混合有无决策、正常评测、训练空批次。未实施、未追加GPU预算。

另见26个“无编队单位同帧被两个基地分配不同驻防点”的单位-帧记录，符合本次
明确保留的重叠防区逻辑。例如loop4016同一Marine66322433收到sequence55/56
两个Move。它早于首次入组，不违反本次修复边界；尚未证明其损害，不同时修改。
侦察、对空集结和长期RL/KL可信性仍未通过本局验收。结论是保留这个有局内证据
的单点修复，再解决评测收尾；不是宣布系统彻底修好，也不据一胜启动96-step。

上述收尾建议随后经用户批准实施并单局验证，见[完整记录](evaluation-only-finish.md)
及[r2结果](r2-eval-result.json)。没有追加侦察、驻防配额或训练修改。
