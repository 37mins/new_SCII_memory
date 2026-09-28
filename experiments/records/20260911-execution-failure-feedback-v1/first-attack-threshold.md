# 自动首攻阈值：最小因果对照方案

2026-09-12；状态：**两组完成、退出0；均0/4胜，不支持将40阈值晋级为修复**。
本次是单独批准的新实验，没有把此前未使用的温度短RL预算转作新实验。
继续沿用本分支/record；正式基线、旧deployment和历史结果不变。

## 为什么测这里

[16局时序审计](temperature-timeline.md)确认15局第二轮前仍有20个Marine，之后
同一批20兵自动出击并全灭。第二轮选择能影响后续生产，但没改变共同首攻。
这支持测试自动首攻时机，不证明20阈值必然错误，也不保证推迟就能获胜。

冻结`9ae8668`的`script_player.py:4258–4288`用`has_launched_first_attack`
区分首次阈值和后续10兵阈值。直接阻止首攻并不设置该标志，会同时阻断后续分支；
伪置标志又会立刻按10兵重发进攻，不能当成“只取消一波”。因此拟先测试已有参数，
不另造“等多久再放行”或“模型同意才进攻”的新状态机。

## A/B 唯一干预

| 项目 | A | B |
| --- | --- | --- |
| 自动首攻的可用战斗单位阈值 | 20，原值 | 40，已批准诊断值 |
| 后续自动波次阈值 | 10 | 10 |
| 前两答 | 同一份A4历史回答 | 同一份A4历史回答 |
| 第三答起 | 在线A4，原采样配置 | 在线A4，原采样配置 |

40是预先声明的一个加倍对照值，不是公开工作证明的最优值，不是根因修复，
不新增阈值搜索。若B始终攒不到40，记录未触发，不降低阈值、补兵或自动重试。
本实验检测系统对首攻阈值的敏感性，**不是恢复模型对进攻的控制，也不是RL验证**。

两组同用原A4 SpB完整step4、Level6/Harder、Flat128/TvT/Air、实际游戏seed42613
各4局，只有一个独立场景。g2资产、开局Planner、marine_pressure阶段待办、
失败反馈开、约15条/xN、192token、T0.4/top_p0.9、history off、shaping off、
FP32参数/温度归一化、BF16 autocast、cache开、SWA dropout0及微操均冻结。
不使用高温0.8，也不换A8/B4或原始权重。

[输入样本](first-attack-inputs.json)直接提取温度A-env0的前两答及完整query，保存
源文件SHA256和逐条prompt SHA256。第一答为Depot4/Barracks4/Marine24；第二答为
Marine42/Depot8/Barracks3/Reactor3/SCV3/Refinery1，均无OFFENSE。
沿用`second_plan_eval`的历史回答重放边界，两组前两答都明确标记为人工重放、
不算在线模型选择；同时保存未采用的原始模型生成。每次重放前校验输入哈希。
之后游戏状态分岔，不能要求后续输入或token随机数流继续逐字相同。

最小实现：`first_attack_eval.py`仅在`LLMPlayer.__init__`结束后覆写
`strategy_params['first_attack_threshold']`并记录原/新值；不修改
`execute_auto_attack`、生产方法或模型回答。复用既有评测入口/受控后台控制器，
不整包替换；候选默认不安装。既有`run_feedback_eval_node47.sh`只在显式设置
`FEEDBACK_FIRST_ATTACK_THRESHOLD=20/40`时选择此hook、四局种子与审计声明；
拒绝同时启用温度或第二答替换候选。`run_first_attack_pair_node43.sh`负责串行两组，
共同2700秒截止、各自确认退出码/真实加载/种子/阈值记录，失败停止不重试。
外层后台会话用2685秒timeout＋15秒kill-after落实总时限。下文保留启动与结果记录。

拟定RUN_ID：
`20260911-execution-failure-feedback-v1-spb2-A4-l6-firstattack{20|40}-eval4-s42613-r1`。
前后两次模型加载均需返回step4、val_only、0次PPO，不能只检查checkpoint目录。

## 预算与验收

已批准43 GPU0–3串行A后B，各4局，总上限45分钟，最多3 GPU小时；不做PPO、
不保存checkpoint、不复制大权重、不自动重试或续长训。09:50:20北京时间快照：
43八卡空闲，无GPU进程/tmux，可用主机RAM449GiB，本地盘94GiB；16个旧SC2孤儿
进程未处理。用四卡匹配现有checkpoint，串行避免此前同节点双模型主机内存不足。
空卡快照不等于资源预留，启动前重新核验归属。

先验收前两次输入/回答及第二轮前回放前缀一致、A真实20阈值/B真实40阈值、后续
阈值10不变。若不一致，报告实际偏差，不将它算作有效单变量结果。
记录首波形成/出发时间、实际单位tag及数量、首波战损/击杀、后续实际出生单位、
SCV损失、下一次Planner调用、整局累计reward和自然胜负。未触发首攻或无后续
Planner的局仍计入整局结果，单独标明。

B若降低首波损失并改善整局表现，只支持进一步验证首攻时机；不能宣布训练学会或
阈值40可跨种子推广。若只是延迟失败、没有改善、未触发或前缀校验失败，应分别
记为无提升、触发不足或无效对照，不扩大预算/范围。之后是否另做RL由结果决定。

## 已完成的CPU工作

```bash
/private/tmp/sc2-air-results.HBI61m/decoder/bin/python -B \
  experiments/records/20260911-execution-failure-feedback-v1/test_first_attack_boundary.py -v
```

6/6通过，0.040秒，退出0。测试从`git show 9ae8668:…/script_player.py`提取实际
`execute_auto_attack`，核验整文件哈希，再用小型单位集合测试：无模型OFFENSE仍
会自动首攻；侦察/忙碌单位排除；20/40边界；首攻后仍按10；伪置已首攻不能取消
当前20兵出击；历史回答合法且哈希一致。它没有模拟SC2交战，也不验证40的胜率。
第一次CPU运行因测试替身缺`harassment_wave_index`及导入路径报3项错误；补齐
测试设置后通过。这是测试自身返工，不是项目新发现或修复的bug。

正式运行源码、prompt、reward、Executor、微操和训练配置均未改；只有实验专用hook
改变该首攻参数。09:58:13重查43仍八卡空闲、449GiB可用RAM、94GiB本地盘。

实施后本地验证：`PYTHONPATH=.:tests <decoder-python> -B -m unittest
test_first_attack_eval test_second_plan_eval test_feedback_temperature_eval -v`，13项中11通过、
2项因缺Torch跳过，1.186秒；边界6/6再通过，0.036秒。A20/B40 effective config
归一化输出路径后只差`SC2_FIRST_ATTACK_THRESHOLD_AB`，默认A4入口仍为原八局独评。
setup-hook不探测CUDA的两项需在服务器固定依赖上补验；不把本地跳过描述为通过。
`git diff --check`、shell语法通过。没有进行模型或游戏CPU替代评测。

## 2026-09-12 正式启动

冻结源码`e3c0695adf0a2e36c97afba838ec3b508f000e34`，43独立目录
`/root/new-scii-execution-feedback-20260911/code-e3c0695`，clean detached HEAD。
增量bundle已备份到该节点的`first-attack-runtime-r1.bundle`，SHA256
`fc1143890ec7a01840f1f22dfd1e540435034aa6ca000722ab9e989240262846`。
未修改旧deployment，也未把新候选合入System-v1核心代码。

服务器固定依赖下同一组13项CPU回归全部通过，14.706秒、退出0，无跳过；
包括本地缺Torch跳过的两项setup-hook无CUDA探测测试。日志位于该节点
`/root/new-scii-execution-feedback-20260911/first-attack-cpu-e3c0695.log`。

10:08:19北京时间重新检查43八卡空闲、无GPU计算进程/tmux、可用主机RAM449GiB、
本地盘94GiB。16个历史孤儿SC2进程未动。10:09:00启动受控tmux
`sc2_firstattack_r1`，仅用GPU0–3，A20后B40，共用10:54:00的45分钟预算截止。
控制记录：`/root/new-scii-execution-feedback-20260911/first-attack-pair-r1`；
外层日志为同路径加`.log`。两组使用上列唯一RUN_ID，run根仍为43本地盘：
`/root/new-scii-execution-feedback-local-20260911/shared/yangjinyue/new_SCII/experiments/20260911-execution-failure-feedback-v1/runs`。

10:11:55核验A组已真正进入游戏：`checkpoint_evaluation.json`记录
`loaded_global_step=4`、`checkpoint_load_returned=true`、`val_only=true`、
`optimizer_updates=0`，完整路径指向A-r2的step4，并非初始模型或B4。
`evaluation_seeds.jsonl`实际四局均为42613；四份`game-*-first-attack.json`
均为首攻20、后续10，其他策略参数未变。SC2日志10:11:02建局、10:11:07首次Planner
调用，Flat128/Harder/Terran/Air；GPU0–3各约12.6GiB，GPU4–7未用，未发现异常堆栈。
日志里的旧`default_player_Qwen3-30B`是沿用的logger名，实际模型以恢复证明及
`SSESWAMoBAForCausalLM`日志为准，不能据logger名把这次认成Qwen。

当时状态是A运行中、B待接，尚无完整胜负结论。B组只在A组完成并通过加载/种子/阈值/自然终止核验后进入；
任何失败均保留并停止，不自动重试，不启动PPO或温度0.8短训。

## 完整结果（9月12日15:22起只读核验）

控制器10:09:00–10:30:52北京时间运行，共21分52秒；A于10:16:41完成，B随即
启动、10:30:51完成。控制器`done=0`、`stage=completed`；两组`.done`/`result.json`
均completed/exit0、4局自然Defeat、0截断、0次PPO。15:22:08核验43八卡均0MiB、
无GPU计算进程和tmux。本轮无新checkpoint、无重试或续训。

两组manifest均为clean `e3c0695`，均真实恢复同一A4 step4、T0.4、实际seed42613×4。
扣除RUN_ID及隔离临时路径后，effective config唯一差异是首攻阈值；四局参数证明
A20/B40、后续10未变。八组前两次完整输入哈希和转发回答相同，重放边界全通过。
这是输入/配置核验，尚未逐事件重新比较本轮回放前缀。

| 实测 | A首攻20 | B首攻40 |
| --- | --- | --- |
| 自然胜负 | 0/4 | 0/4 |
| 平均整局累计reward（result口径） | -49.382 | -49.553 |
| 每局消耗回答数（含2条历史重放） | 2、2、2、2 | 7、4、7、5 |
| 每局在线回答数（排除重放） | 0、0、0、0 | 5、2、5、3 |
| 在线语义有效回答 | 无在线样本 | 13/15 |
| 日志记录首攻的游戏 | 4/4 | 2/4 |

不能把A的8/8有效回答当成在线模型准确率：它们全部是指定历史重放。
B总有效21/23也含8条重放，真实在线为13/15；`val/sc2_action_valid=0.5`并非这个
全回答分母。原始`prompt_response.jsonl`还含已终止环境的重复占位行，不能简单将
全部行reward累加为整局回报；此处使用结果文件既有整局聚合，不更改统计代码。

实际攻击日志（游戏时钟；iteration×4/22.4）表明：A四局均4:10.71建首波、
4:30.89记录20兵出发、5:00.89记录首波全灭，随后继续10兵波次。
B env1于8:03.39建首波、8:24.82记录40兵出发、9:03.21首波全灭；
B env3于8:11.43建首波、8:31.79记录40兵出发、9:11.96首波全灭。
B env0/2没有首波创建或出发记录，原因尚未结合回放核验，不能直接说始终没有40兵。
两局B在首波全灭前仍按原规则派出后续10兵。首波全灭是现有编队日志结论，尚未
逐tag审计死亡/重分配，不能把它替代本轮完整击杀和死亡账本。

B两个无效在线回答均有裸`techlab x 1`，失败原文为
`action outside current canonical BO candidates: techlab`。其余13个在线回答有效，
但有效并未带来胜利。本轮未调整prompt或过滤。

结论：40阈值实际延迟了两局首攻，并让四局进入后续在线规划，但没有改善胜负或
平均回报；**拒绝将40作为已验收修复或据此晋级长训**。只有一个诊断种子，不作
跨种子优劣或模型学习结论。下一项优先复用这8份回放，核对两波40兵的组成、
战损/击杀和另两局没有首攻的原因；不继续搜索阈值、不新增GPU实验。

原始结果SHA256（完整回放哈希分别保存在各组result.json）：

- A result：`9aa4497eaab10a06db3f28542bfacf619c52210eaebbd13f68897a8e333ef3d4`
- B result：`692d2e4fb470d15cfbd49a56c860abe227539f7c743c8a70004991adbe688b4a`
- A interventions：`18b4f7856cd9334801397031beb9583d83d7c5b739f3d6e56c4e48ec3951def9`
- B interventions：`d2fd2f1570520618a5f943ede4ebaad1343dd48561eb6302f8382b1c17635822`
