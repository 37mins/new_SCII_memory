# 冻结 r3 系统的 SpB Level6 四步 RL

2026-09-08，用户要求立即开始真实RL，不以单局失败或星港停工代替学习验证。
当前状态：42八卡Level6运行已失败（19:23:07），完成4/96步，无checkpoint；
旧初评已完成，启动情况见文末。
沿用本experiment/branch，冻结runtime
`e1e25ce1d14362b209e34ec73871ebaeb63538a6`，不加入星港接班、early-stop或其他修复。
私有bundle备份编排；原r6、r3、旧deployment及结果不动，不推GitHub或合并main。

## 问题与预算

验证同一系统中SpB能否完成连续参数更新，以及训练后的回答和游戏表现是否变化。
initial-eval八局 → train4四个global step（每步8条rollout）→ step4-eval相同八局。
一次global step包含若干mini-batch optimizer更新，二者不混称。最多48局，不
追加种子、重复初评、自动重试或接长训。初评/末评各60分钟、训练240分钟的墙钟
上限，总计最多六小时加终止宽限。预计2–4小时，首步实际耗时后更新估计。

15:37–15:39北京时间只读检查：42/43/46/47/48无整张空GPU，44仅1、7空闲，
均80GB；其他卡属于外部任务，不终止、不共享。15:47启动前GPU1被他人占用，
正式run未创建。15:48确认44的GPU6、7空闲，编排仅将物理选择1,7换为6,7；
新SHA部署，之前f9ca613编排与CPU产物保留，各阶段仍相同两卡。
全局mini8、micro1不变，现有FSDP将每卡mini正规化为4，累积4个micro再更新。
这不是与八卡旧曲线的严格单变量对照；硬件并行布局显式记录。共享盘7.5TB可用、
/dev/shm491GB可用，依赖已存在。不部署新环境、下载权重或跨节点拼卡。

## 冻结内容

原始`Panyuqi/SpikingBrain-2.0-instruct`／SSESWAMoBAForCausalLM，5.17B，无项目SFT
或RL续训。资产manifest SHA256 `40d6f1417568d5fda1107cdafd66c7441a679b03fb148851c8f226ecc9cf884c`，
内容revision `sha256:de52e8ef993afc763f9d4fb498e8d8302ef0bc13bd7dfd53c6ae5f77f43698b5`。
复用原prepare_spb_model的已登记训练kernel补丁及权重软链接；SWA dropout0，
FSDP参数FP32、温度归一化FP32、BF16 autocast、HF/eager、生成cache均保持。

Level6/Harder、Flat128、Terran/Air，canonical g2、Python游戏随机数固定。
g2 SHA256 `304b1f4ff12fe30c5e6d5f7b8eaef8ed6823ca65afed3a3706cae10e3639c494`。
6BO/6bullet、4096/96tokens、history0/off、原prompt/解析器、合法BO不补兵重排、
对空备选及无效回答规则、原微操/20首攻10后续/驻防不抢进攻队、自然终止、
shaping off和reward全冻。SC2_RESET_RETRIES=1，不以重采样补空策略批次。

训练复用原GiGPO mean_norm/gamma .95，PPO epoch1、AdamW lr1e-6/weight_decay .01、
grad_clip1、entropy .001、reference KL .01、原loss裁剪和复制补齐口径，不增机制。
训练temperature .4/top_p .9；两次独立评测均greedy/top_p1。4个global step，
SAVE_FREQ=1，无checkpoint数量清理，完整保存model/optim/extra/data及模型配置。
不启用额外整批概率诊断；使用已有训练指标、原始回答、outcome和checkpoint。

初始/step4的八个实际种子必须逐项一致：
`2807211443, 882774615, 494016583, 2621640384, 4081499387, 44124510, 2071061852, 1549697793`。
来源为冻结环境validation `RandomState(1000)`首批，非按胜负挑选；训练seed序列
另记training_seeds.jsonl。原始模型与step4各自新进程评测；step4通过原resume_path
恢复本轮两卡checkpoint，不误用初始权重。纯评测计入无模型决策局，不产生PPO。

## 启动与验收

入口 `run_spb_rl4_node44.sh suite` 顺序执行三段，任何阶段失败就停止后续；每段
重新查GPU空闲，不删除检查、不杀别人的任务。runtime只读复用44已有
`/root/new-scii-garrison-wave-20260908/code-e1e25ce`，新编排放独立`orchestration-<SHA>`。
RUN前缀 `20260908-garrison-wave-ownership-v1-spb2-l6-g2-rl4-2gpu-s0-r4`，后缀为
`initial-eval`、`train4`、`step4-eval`，写入本experiment共享runs；Ray路径独立。
完整CLI/effective config/manifest/模型准备哈希/.done及编排副本分别保存，不覆盖旧目录。

按游戏seed比较胜负、累计reward、Planner消费次数、完整回答有效率、重复/空答/截断，
区分初始评测、训练rollout、末评。无Planner局保留在游戏分母中，动作有效率为N/A。
回放若遇到既有按秒目录碰撞，明确缺失份数，不声称8份齐全；不在本轮改日志机制。
训练核对4步确有更新、所有指标有限、每步7个必需两卡checkpoint文件及配置齐全。
NaN/Inf、归因错误、空策略batch、未更新或checkpoint缺失按现有失败路径停下，
保留产物，不伪造样本或追加自动重试。单局负、四步未涨胜率或单个KL高不单独判失败。
固定BO开局不能由Planner的RL直接优化；接管时局势与接管后表现分开解释。
此短测只能判断连续性及明显退化迹象，不能证明小幅学习提升或架构优势。

启动前CPU记录：纯评测无策略边界12项、系统入口/冻结26项回归通过；bash语法及
diff检查通过。44上三个实际入口dry-run、Hydra解析、真实训练器_validate_config
均通过，确认env rollout8/world2/mini8/micro1；train4为非val、4epoch/4step/save1，
初末评均8局/greedy、末评resume至本轮global_step_4。未加载模型或开始游戏，
不将这些结果记作训练跑上。旧f9ca613私有bundle已在44校验完整历史，SHA256
`1e0c0fe9ae2a18cac67172a967d4b52913eb988d76044910698eeaaa94d5c2ad`。

## 实际启动与用户要求迁移43

15:51:21北京时间，确认44 GPU6、7无计算进程及新run不存在后启动受控tmux
`sc2_spb_l6_rl4_r4`。编排冻结 `b6797eed5350ec08e6a054450239aa1c321ed867`，
私有完整bundle已复制校验：`/root/new-scii-garrison-wave-20260908/rl4-b6797ee.bundle`，
SHA256 `b275d5d07359124a955d06c7c9648539e89a0f56059555f18e520851a72d639f`。
15:51:47初评manifest记录formal/clean、runtime e1e25ce、SpB原资产、两卡八局。
15:54确认模型已加载、swa_dropout0/use_cachetrue、SC2进入02:52；实际八个种子
与上表逐项相同。此时PPO尚未开始，不能把初评称作训练更新。一次外部VLLM进程
曾短暂在模型加载期间出现，15:54已退出；未终止它，未发现本任务OOM。

用户随后明确要求停止43任务并使用43。15:56只读定位为旧Qwen3-30B-SC2-Merged
TP8/port8001服务，主PID2391524、引擎2391700、GPU worker2391927–2391934。
重新核对model/TP/port后只向主PID发送SIGTERM；15:59:32确认八卡0MiB、服务
进程均退出。没有删日志、模型、环境或停止其他服务。

为避免44自动接训与43重复，15:58仅向本套件父编排PID444866发送SIGSTOP；
初评子进程444871继续，冻结脚本未改。15:59:33确认父进程T、初评继续，train4
目录尚未创建；已有游戏和回答保留。不得直接恢复该父编排又另启43训练。

43的系统Python也是3.12.3，但所查已验证Python覆盖目录和3.12编译头均不存在；
已询问是否允许将44约8.1GB依赖及编译头复制到43独立新目录，尚未执行该复制。
当前状态是44初评running、训练迁移待依赖部署授权，**不是43八卡训练已启动**。
后续初末评测须记录两卡到八卡的布局变化或复用既有导出入口保持评测布局；不
静默称所有配置完全相同，不因迁移自行增加初评游戏数。四步及原游戏预算不变。

随后新增[43八卡SpB Level5 96步](spb-level5-96.md)时，agent误将其当作替代Level6，
未恢复本记录train4。该解释不是用户明确取消Level6；保留此调度错误记录。

## 初评完成与42八卡96步授权（2026-09-08）

44初评16:22:50完成，退出码0、training_updates0，八局均自然结束、4胜4负。
四胜均未调用Planner；四负分别消费7/31/34/31条回答，共103条，75条有效。
`val/text/test_score=-14.096`；此为日志聚合指标，不用它替代未经逐局核对的累计reward。
回放记录只有一个文件，存在按秒目录碰撞，不能声称八份回放齐全。初评及原run不覆盖。
16:28核验train4/step4-eval目录均不存在，旧父编排仍暂停，不直接恢复以免重复占卡。

用户明确授权在42跑Level6 96个global step。沿用原始SpB、冻结runtime
`e1e25ce1d14362b209e34ec73871ebaeb63538a6`，无SFT、无旧RL续训、dropout0、cachetrue。
复用刚部署验证的八卡96步入口，只选择Harder和42路径；prompt/BO/微操/reward/
精度/lr/loss不变，不加early-stop或空批次重采样。相对原两卡train4，资源布局变为
八卡、目标96步、SAVE_FREQ8；不是与两卡初评的完全相同硬件配置。
直接训练，不重复初评；每步8条rollout，96步最多768条。每8步完整保存model/
optimizer/extra/data，不清理checkpoint，首份在step8、共12份约0.8TB。72小时墙钟
上限；实际耗时首步后估计。若遇空策略批次、NaN/Inf、归因或运行错误，冻结入口
退出，保留失败记录，不自动重试或伪造训练决策。单步胜负或KL高不单独判bug。

16:30只读确认42 GPU0–7均0MiB、无compute进程；系统盘20GB、共享内存52GB、
共享盘7.5TB可用。独立部署`/root/new-scii-spb-level6-96-20260908`，复用同一份44
依赖归档8.1GB及编译头；新目录解包，不修改旧环境，不下载或复制权重。CPATH同时
包含本目录`include/python3.12`与`include`。固定Ray2.48默认按剩余/dev/shm的95%
封顶object store（`ray/_private/resource_spec.py`），本轮不增加内存配置override。

入口`run_spb_level6_96_node42.sh`复用`run_spb_level5_96_node43.sh`，共同逻辑只开放
部署路径、难度和记录文件三个已用配置。源码仍从独立clean `code-e1e25ce`运行。
RUN_ID `20260908-garrison-wave-ownership-v1-spb2-l6-g2-rl96-8gpu-s0-r6`，写入本
experiment共享runs；Ray `/dev/shm/sc2_spb_l6_96_0908_r6`，tmux `sc2_spb_l6_96_r6`。
正式启动须再次确认空卡、RUN未使用、资产哈希及配置；准备不计作PPO更新。

### 42 启动核验

编排SHA `939b41b698e732345a36c921cb24d91a4fc8a83c`，runtime仍为`e1e25ce`。
私有完整备份`/root/new-scii-spb-level6-96-20260908/source-939b41b.bundle`，SHA256
`7ef67220279ab3203f1697e99b371932195a765e86b8867f023c2c8a48c32d8e`。
依赖/头文件归档哈希与43登记逐项一致；解包后系统盘约12GB可用，不清理旧文件。
旧`spb_l5_s1_r10_a10042`队列仍等待300GiB空闲/dev/shm，目前52GB、不满足条件；
该队列及其他tmux未终止。新任务启动前再次确认八卡无compute进程。

本地入口语法、`git diff --check`及`test_system_release.py`26项通过。
42实际导入Torch2.7.1+cu128/Transformers4.55.2/Ray2.48.0/Gymnasium1.2.3、FLA/
MoBA/flash_attn通过；dry-run `/dev/shm/spb_l6_96_dry_939b41b`的实际命令经
`--cfg job`和`RayPPOTrainer._validate_config`通过。与43配置相比，非路径差异
只有SC2_DIFFICULTY和SC2_EXPECTED_DIFFICULTY同时Hard→Harder；此前CPU比较
断言漏列后者而失败，是检查脚本错误，补齐核对后通过，未更改训练配置。
一次SSH中断发生在训练启动前；重连确认未产生正式run，未重复启动GPU任务。

16:42:14新tmux `sc2_spb_l6_96_r6`启动，pane PID1934299。控制台
`/root/new-scii-spb-level6-96-20260908/rl96-939b41b-console.log`。
16:42:38权重/BO/SC2资产哈希通过；此时尚不能声称PPO已更新。

16:42:49.806579实际manifest生成：clean/formal、runtime e1e25ce、mode=training、
world8/GPU0–7、96epoch/96step/save8、val_before_train=False、val_only=False，
模型manifest及g2哈希与前轮相同。八卡WorkerDict PID为1938248、1938451、1938453、
1938454、1938455、1938456、1938458、1938459；权重加载完成，actor/ref日志均为
SSESWAMoBAForCausalLM 5.17B、swa_dropout0.0、use_cachetrue。
16:45:42八局SC2实际创建：Flat128、Computer Harder(Terran,Air)。16:45:58游戏
已推进至00:35，首批seed记录为base0/reset0的八个2732（同组rollout，不是八个
独立seed）。没有Traceback、.done或已完成actor更新指标；当前是训练首批rollout，
不是又一次初评，也不是96步或学习效果已通过。记录保留首步后估时及后续checkpoint验收。

### 结束与同seed控制

19:23:07最终exit1：第五批SC2等待900秒后无数据，前四步32局自然终局8胜24负，
未发生人工call20截断。日志指标有限，不代表概率链条已验收；save8尚未触发，
20:32只读确认checkpoints目录为空。console SHA256
`9d5f53ff5152dda90670954967ec4bca196f66e44a338f6a12c6e16d83985e6a`。
用户随后批准[原始模型同失败seed控制](failed-seed-control.md)，不重训或替换历史权重。

21:30:45原始SpB的seed43567控制完成，同样0/8。两组首次Planner输入全同，
均在09:52接管；不能再把第二批零胜全部归因于第一次参数更新。
逐局结果、固定BO损耗与受限接管证据见[同seed只读分析](failed-seed-analysis.md)。
