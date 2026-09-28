# 参考五月动态分支的开局 Planner A/B

2026-09-09 用户要求参考五月十难度版本做 A/B。状态：两组已完成，各4局自然终止、
退出0、PPO更新0；12:22恢复回读并核验结果。早前SSH中断未阻止本次评测完成。
本次仍为原始 SpB、Level6/Harder、Terran Air、Flat128 的纯评测，各四卡四局，
同节点八卡并行、从实际启动起最多50分钟（含退出宽限），不做PPO、不续训。

## 检验什么

A 保留当前完整237条固定g2，队列耗尽后调用Planner。B初始化空执行队列，
首次游戏回调沿现有Commander→Strategist→Planner链生成BO，再经现有解析/执行。
唯一游戏参数差为 `SC2_FIXED_BO_PREFIX_LIMIT=0/-1`。旧138候选保留，不改默认0。
B仍核验原g2资产哈希，但不执行其中任何固定动作，不把237条转成待办或记作完成。
`-1`只影响初始队列，后续模型计划不切片。

优先观察真实首次调用时间、阶段策略/待办来源、原始回答到实际执行的作用，
再按env_index与游戏seed比较自然胜负、整局累计reward、产出/损失及回答退化。
计入无Planner局。增加调用数或有效率不等于效果提升，更不能证明RL有效。

## 五月依据与不可恢复边界

本地来源 `/Users/yangjinyue/Desktop/codex/星际争霸`：

- `run_spb2_difficulty_recover_0065_Harder_clean_8gpu_96ep.sh` 指定
  `/data1/xrp/verl_agent/verl-agent/planner_only_missing_bo.json`，调用Planner runner。
- `code/SC2_online.zip` SHA256
  `e40fb7dad98f4c7d336031f3b2047ee3e15113ce14925830cfb6540aaa5888ab`，
  包内 `SC2_online/SC2ArenaDyStrategy/players/llm_player.py:64,68` 注释固定加载、
  初始化 `bo_data=[]`，队列为空调用 `commander.renew_bo`。
- `experiments/records/20260813-may-spb2-bo-branch-audit/MAY_SPB2_BO_BRANCH_AUDIT.md`
  支持五月使用动态Planner分支，但明确完整五月runner/runtime未恢复。

只参考该可见调用行为，不复制包内其他源码；原包未核实独立许可证，不重新分发。
包内动态文件是未提交工作树状态，不能用嵌套Git HEAD代表它们。
本候选不是五月全量复现，不恢复旧SFT权重、旧prompt、旧策略实现或旧reward。

## 待办和冻结边界

当前 `marine_pressure` 首阶段仍是 `Fast Marine Pressure Opening (00:00–03:30)`，
原配额SCV2/supplydepot4/barracks4/marine24。Commander调用现有时间阶段选择器，
按观测宏观扣减，然后由原Planner前置过滤生成query。12 SCV的合成开局观测会
使原SCV配额被扣为0；这一原有规则本轮不调整，不宣称它就是五月规则。
无g2尾部转换、强制阶段或新配额。阶段推进和选择即扣待办的原语义均冻结。

现有canonical prompt、6BO/96token、history0/off、air选项、无效回答补齐、
Marine guard、有效回答直通、自动20/10波次、驻防归属、微操、终止/reward、
FP32参数和温度归一化、BF16 autocast、HF/eager/cache、SWA dropout0全部保留。
规则补出的动作须与模型原始选择区分。不能把不同接管时间产生的query变化
说成另改prompt；也不能忽略现有规则可能覆盖模型计划的作用。

## 配对、部署和保留

复用已实际完成r3的 `handoff_eval.py` / `run_eval.sh` / `run_timed_arm.sh`，
只新增B参数选项和47薄启动器 `launch_may_node47.sh`。游戏seed逐环境为
`[2732,2732,43567,43567]`，只有两个独立游戏seed；生成RNG沿原每rank首次seed规则，
温度0.4/top_p0.9/top_k0。交接后请求数不同，不保证逐token采样随机数配对。
两组均原始SpB快照，模型文件由 `configs/spikingbrain2.sha256` 验证；
SC2 Base75689/map/g2/data与源码哈希均由原入口验证并写manifest/effective config。

47在10:49核验八卡0MiB、无compute进程、主机available约954664MiB。
复用 `/dev/shm/spb-unitid-20260906/python` 和已部署Python头文件，
Torch2.7.1+cu128/Transformers4.55.2/Ray2.48.0/NumPy1.26.4，不装新依赖。
冻结目录 `/root/new-scii-g2-planner-handoff-20260909/code-<SHA>`；
独立 `pair-may-r1` 记录共同截止和退出码；启动前再次查空卡，失败不自动重试。

RUN_ID为 `20260908-g2-planner-handoff-v1-spb2-l6-may-opening-{A,B}-eval4-s0-r1`，
输出根沿用 `/mnt/data0601/yangjinyue/new_SCII/experiments/20260908-g2-planner-handoff-v1/runs`。
初始化读取的是源模型，不加载任何RL checkpoint；val_only、save_freq=-1、resume disable。
额外GPU占用或存储错误停止并保留原失败记录，不用失败代替0胜。

父SHA `b8c5388da2134af73dcf08f5892f1fc9f7cf86fb`，沿用隔离候选分支。
改前完整bundle `/private/tmp/sc2-before-may-opening-b8c5388.bundle` 已验证，SHA256
`c5e8dab1925162adc7ef489b6c50bfd71672f9209a206abd350563c58f9f9e74`。
旧r3两组分别10:33:41/10:30:12退出0；不覆盖其源码、结果、控制日志。
本次从新clean SHA启动，启动后另记模型/SC2确实运行的证据。

## CPU验证

实际命令（工作树 `/private/tmp/sc2-planner-handoff-20260908`）：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:tests /private/tmp/sc2-air-results.HBI61m/decoder/bin/python -m unittest test_g2_planner_handoff test_handoff_paired_eval test_empty_bo_frame test_valid_bo_passthrough test_system_release.LauncherTest test_r6_planner_interface test_planner_air_options test_air_plan_attribution test_r6_format_reward test_r6_natural_termination test_sc2_evaluation_without_policy test_garrison_wave_ownership test_barracks_production_slots test_pending_addon_placement -v
```

115/115通过、退出0、无跳过，见 [原始输出](may-cpu-tests.log)。新增覆盖初始空队列、
第一次回调先Planner后BO提交、后续队列不被清空、阶段0配额来源、合法回答改变
队列和待办扣减，以及实际A/B dry-run除接管/GPU/产物路径外配置相同。
[完整合成接口证据](may-cpu-interface.json)包含原阶段、扣减后待办、query、手写回答
和Executor队列；没有假称模型生成或SC2真实接受。原无策略评测收尾和训练端
禁止空批次/伪决策回归仍通过。两份shell语法、全部21项源码manifest及diff检查通过。
仅llm_player的已登记源码hash改变，其余20项（含Planner、Strategist、reward、训练器）未改。

## 实际启动

runtime `e3ffac9e2da577679b9b980d66678e4cc72b06aa`，47独立冻结目录
`/root/new-scii-g2-planner-handoff-20260909/code-e3ffac9`，clean。
完整Git bundle在本地和47校验一致：
`c345f3b1a30f216b15d27fad6265ce55c8b0f197f1fa7b7f923d88cf2d37166e`。
GitHub推送被安全检查拒绝，未执行；47现有gh也无该仓库访问权限，不读取/迁移凭据。
本次源码已提交并在用户计算节点独立备份，GitHub同步须重新确认授权，不能称已推送。

47固定依赖下18项构造器/调用链/实际启动器CPU测试通过，耗时57.522秒、退出0。
八卡再次确认空闲后，于北京时间11:03:40在tmux `g2_may_opening_pair_r1`启动，
launcher PID675863；A GPU0–3，B GPU4–7，共同截止11:53:40（epoch1788926020）。
没有改变局数或进行PPO。两份实际manifest均为evaluation、clean/formal、world_size4，
原模型与BO哈希一致，配置确认val_only/resume disable/save_freq=-1与prefix0/-1。

11:05:30最后成功回读显示两组actor开始加载11个shard、日志use_cache=true、
SC2进程数0，尚无实际seed派发/采样文件或胜负。之后新SSH连接在banner exchange
超时，连44读取共享日志也同样失败；跳板机11:08:54仍可连。
这只是连接可观测性中断，不是已确认NFS故障、训练错误或评测结束。
不自动重试实验，不延长50分钟预算；恢复后先查现有tmux、A/B.done、result、
SC2归属和实际种子/首次Planner证据，再更新状态。
[启动审计快照](may-startup.json)保存已确认字段；目前无A/B效果结论。

## 完成结果（9月9日12:22核验）

两组runtime仍为clean `e3ffac9e2da577679b9b980d66678e4cc72b06aa`。
A于11:34:50、B于11:18:34结束，均在50分钟预算内；A/B.done、pair.done、
各run的result/.done一致为退出0。12:17核验47八卡0MiB、无tmux和SC2进程。
原始模型、真实游戏seed与首次Planner回答已核对，见[结果与artifact哈希](may-completed-results.json)。

| 指标 | A：完整固定g2后接管 | B：开局Planner |
|---|---:|---:|
| 自然胜负 | 2胜2负 | 3胜1负 |
| 整局累计reward均值 | -13.992 | 3.012 |
| 首次Planner游戏时间 | 09:52–10:00 | 四局均00:00 |
| 有效回答/可判定回答 | 34/53（64.2%） | 39/45（86.7%） |
| 无模型决策游戏 / 人工截断 | 0 / 0 | 0 / 0 |

按env_index配对，实际游戏seed均为[2732,2732,43567,43567]。
前两局双方均胜；第三局A负/B胜；第四局双方均负。
各局累计reward为A[20.5,20.5,-48.467,-48.5]、B[20.5,20.5,20,-48.95]。
计算仅累加至每个环境首次done，排除已终止环境的重复占位日志；不把最终单步
`val/sc2_action_valid`的1.0/0.75当全程有效率。

B第一局00:00原始回答为：

```text
- supplydepot
- barracks
- marine
```

该回答被接受并消耗policy_action_id=1；四局均真实调用Planner。A第一局仍有
`- siege tank`拼写错误，B仍出现剩余数量超限、超过6条和进攻格式错误。

结论：本次开局接管没有导致系统崩溃，小样本结果优于固定g2对照；只支持继续
验证此方向，不证明五月完整恢复、RL能学习或稳定提升。每组只有两个独立游戏seed，
同seed重复局不能按四个独立样本计算置信区间。不同接管时刻也改变了实际阶段输入
（B从当前Marine pressure开局），不能把收益全部归给模型本身。尚未逐单位审计
执行/战斗归因；不据此改默认或启动长训。部署和旧记录均未改动。
