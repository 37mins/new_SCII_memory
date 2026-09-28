# Qwen KL 统计与更新定位

2026-09-06夜间最新方向：停止新增机制和数值诊断，early-stop只保留备用；
用户授权冻结现有配置进入[Level6真实RL，96步目标、8步验收](../20260906-integrated-fixes-v1/real-rl-stage8.md)。
r33及单更新参数链重放均不启动。下文失败、候选和旧计划均保留为历史证据。

已有产物的[首次更新CPU定位](first-update-r32.json)：最差为rank0/sample63
第二个token ` marine`，logp −0.142677→−8.125298；它不是首次8回答之一。
八回答都唯一且来自env0；最差样本来自env7。直接logprob梯度与冻结计权公式
最大差1.56e-10，分片范数重建0.193066411，与全卡0.193066418一致；未保存
参数梯度张量、Adam moments或参数差，故未证实根因。`audit_first_update.py`
仅CPU读取旧r32、r7及tokenizer，已执行，不启动模型；未据此改任何训练机制。

最新结果（09-06 15:57北京）：44 GPU0–3的r7已于15:36完成1步/8局，退出0，
checkpoint完整。零更新eval/train一致，但生成端尾部差异、更新后EOS漂移
尚未解释，并确认跨rank指标只保留rank0，不能放行长训。见末尾结果与
[完整诊断归档](result-node44-r7.json)。整合四步随后于16:08完成，1/32胜，
见[整合完成记录](../20260906-integrated-fixes-v1/README.md)。未启动r6/r8或96步。

后续状态：47的八卡诊断r5在进入PPO前退出137，0/1更新，不能算GPU验证通过。
[失败证据](https://github.com/CeraveNew/new_SCII/blob/334ce26207b9c44885389b63cc464f4b3cfb31c3/experiments/records/20260906-qwen-kl-metric-v1/failure-node47-r5.json)
与诊断代码保留在独立分支；Integrated Fixes v1只纳入已通过CPU等价验证的
KL汇总一行，不纳入诊断forward，真实KL漂移仍未定位。

状态：开发验证中，长训 held。复用同一 KL experiment，旧尝试的完整记录
保留于 [c16384c](https://github.com/CeraveNew/new_SCII/tree/c16384c6e2f4c56675e9bab573a83a6f5dd2d2fb/experiments/records/20260906-qwen-kl-metric-v1)。
48 的旧 r2/r3/r4 均未完成 PPO：分别是 Python.h 缺失、外部显存竞争；
不是证明 KL 修复有效或无效的实验。本轮不重试那些 RUN_ID。

## 基线与问题

当前基点 fa76c3a，训练代码与 44 正在运行的 44a6e9d 一致。该八卡 Level 6
四步运行的第一步 PPO KL=0.271、clipfrac=0.005、grad_norm=15.338。
166 个真实决策经既有 copy 补成168条，以 global mini-batch=8 做21次更新。
KL=0.000 是各 worker 最后一个 micro-batch 的 reference KL，不是全批次均值。
PPO KL 已按列表收集，不受这项覆盖 bug 影响；也不是更新前/更新后的统一测量。

修改前完整 Git bundle：`/private/tmp/new-scii-before-kl-diagnostics-20260906-fa76c3a.bundle`，
SHA-256 `6d28b2263a616b40905e9188e2985d26cd83af5d80f2744c75232245469eb8d1`。
复用现有开发分支；所有旧 deployment、run 和 checkpoint 保持原样。

## 分开验证

第一项只把 reference KL 的覆盖赋值换成现有 append_to_dict；公式、系数、
梯度和优化器不变。移植原 c16384c 测试，CPU A/B 比较真实 update_policy
每次 KL loss、梯度和最终权重，必须完全相同，只允许汇总指标不同。
技术依据是固定 VERL 中相邻指标的同一收集路径，见 [指标定义](../../../docs/reference/kl-metric.md)。

第二项仅增加可关闭的观测，用更新前的 train-mode 重算和逐次更新记录，
区分起点 log-prob 不一致与更新累积漂移。不把 K1 单个数值当精确分布 KL，
也不把 clipfrac 当所有 ratio 越界比例。先验证诊断不改变 RNG、loss、梯度、
参数，再做真实模型短测；未得到证据前不调整 LR、KL 系数或 PPO batch。

单行修复 CPU 验证：47 独立 `staging/20260906-qwen-kl-diagnostics/metric-source`，
`CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=<候选>:/dev/shm/spb-unitid-20260906/python python3 -B -m unittest discover -s tests -v`
72/72通过，无跳过，7.284秒；包含真实 update_policy A/B 的逐次loss、梯度、
最终参数全同检查。Mac仅两项静态KL测试通过，真实Torch测试因无依赖跳过；
Mac全套还缺gymnasium，不能将它写成全套通过。没有上传数据或模型，CPU副本
复用47已有d3e7073源码，只同步actor、两份测试和源码哈希四个已核验文件。

47 已经部署的同版依赖可复用，原始 Qwen3-4B-Instruct-2507 cdbee75 权重
只读。计划 Level 6、8 rollout、1 global step、八卡、保存 step1。其他沿用
44a6e9d：HF/eager/cache 开启，4096/96 tokens，6 BO，canonical g2，seed0，
Harder/Air、legacy reward、GiGPO mean_norm、lr1e-6、grad_clip1、KL系数0.01。
GPU短测需用户批准和启动前空卡/配置/资产核验，不启动96步。

先验判别：若零次 optimizer 更新时新旧 log-prob 已明显不一致，优先查前向
路径与数值；若起点一致而在后续更新增长，再定位梯度/优势/异常 token。
不凭单步胜负或0.271直接宣称学习率错误。最终结论必须分开报告已修统计、
一致性是否通过、真实漂移是否定位；不能把新增日志当作第二项已解决。

## 历史八卡专项短测

用户已批准47八卡、Level6、8 rollout、1 global step；不启动长训。入口
`run_node47.sh`，新RUN_ID `20260906-qwen-kl-metric-v1-qwen-instruct2507-l6-8gpu-trace-rl1-s0-r5`。
源码部署在共享盘 `new_SCII/deployments/qwen-kl-diagnostics-20260906/code-<SHA>`；
复用47已批准的 `/dev/shm/spb-unitid-20260906/python` 与头文件，模型不下载。
独立Ray目录 `/dev/shm/qkl_l6_trace_r5`，screen `qwen_kl_l6_trace_r5`。
输出为本experiment共享runs目录。预计35–60分钟含checkpoint，两小时运维上限。

诊断定义与精确边界见 [PPO诊断](../../../docs/reference/ppo-diagnostics.md)。
CPU回归74/74通过，无跳过，8.590秒；包含带随机扰动模型的真实update_policy
A/B：RNG、loss、梯度、最终参数和原指标全同。源码中loss及optimizer语句
不变，增加的整批forward只观测、不反传。

dry-run与当前44a6e9d的128项override逐项对照，非路径差异只有训练4→1步
和新增 `actor.ppo_diagnostics_dir`。预检effective SHA-256为
`ccb4ff7aa6eac214a9f59dc047b5cc65511dc16044d445a5d42e92c1c8325c17`。
预检不是正式运行；启动前仍需核验clean/pushed SHA、资产与空闲GPU。

11:51已正式启动，[启动核验](launch-node47-r5.json)确认clean/pushed
`1e38a6ab97f03fa76ffe4b5abc96c5e72f990e69`，模型为原始Instruct-2507，
无SFT/RL恢复。最终冻结部署再次74/74 CPU通过；全部源码/模型/SC2哈希通过。
截至11:59仍在第一批游戏，无done，尚不能给出更新前差异或真实KL修复结论。
用 `python3 summarize_trace.py <run>/ppo_diagnostics` 只读汇总三阶段及逐次更新。
首个手工资产检查曾误写manifest文件名，校正为仓库实际文件后验证通过；
未修改启动器、模型或实验配置。原44四步第二步PPO KL已升到1.986，长训不放行。

另做了一次小张量CUDA核对，FP32/BF16、logit scale1/8，共4组，使用实际
Flash-Attn交叉熵、现有PPO/KL和entropy组合。inplace backward开关前后loss
和梯度逐位相同，全部有限，tensor显存峰值26,749,952 bytes；没有加载模型或
更新实验权重。该结果不支持把当前漂移直接归因于此开关，故没有修改该实现。

## r5 最终失败记录（09-06 12:08 北京时间）

[失败审计](failure-node47-r5.json)：退出137，0/1更新，无checkpoint，
`ppo_diagnostics` 未创建，`metrics.json` 为空。8局仅3局自然结束（1胜2负），
不能用这3局的胜率代表完整实验。原始result中的 `metrics_finite=true`
来自对空指标集合的检查，不构成数值健康证据。

控制台显示训练driver被Killed、Ray连接关闭及ActorUnavailableError。
未定位SIGKILL的发起者。内核未发现对应时段OOM；原进程组及两级父cgroup
的oom/oom_kill均为0，进程组memory.peak为160,861,110,272字节，memory.max
不限。systemd-oomd未运行。同期NFS有lost-lock告警，但这些现象不足以断言
根因；47随后读取共享结果超时，同一结果通过44成功读取并核验哈希。

12:24仍有5个已确认归属于本RUN_ID的孤立SC2进程，未发送终止信号，未清理
文件。47另一任务的8个Python进程于12:14启动，12:23已占满八卡；它发生在
本次失败之后，不证明终止原因。不重用r5或原位修改部署，也不自动重跑。

结论：统计覆盖bug已修且CPU验证不改变更新；更新前一致性与后续漂移仍无
真实GPU数据，不能判定学习率、dropout或loss为根因。下一次需获准使用
空闲资源、同一冻结源码和配置、新RUN_ID重跑；两种模型的长训继续held。

## r6：整合版四卡专项（取消启动，改排r7）

基线为当前main `7306c5a`（运行代码`1f2d8b5`）；已有诊断分支合入main，
合并前bundle `/private/tmp/new-scii-before-kl-integrated-20260906.bundle` SHA-256
`2ca566a68dfac8e8adbf29d330868f462bcec05bcacf060d585eea9448ece18d`。
不改prompt、reward、loss、优化器、学习率或cache。相对整合版仅启用现有PPO
观测，并用HF官方接口补齐生成端概率记录。诊断默认关闭，不先合入主线。

RUN_ID `20260906-qwen-kl-metric-v1-qwen-instruct2507-l6-4gpu-integrated-trace-rl1-s0-r6`。
44 GPU4–7，原始Instruct-2507 cdbee75，无SFT/RL续训；Level6 Harder/Air、
Flat128/Base75689、canonical g2、seed0、1步8局，保存step1。保留整合版
off shaping、HF/eager/cache、4096/96 tokens、6 BO、GiGPO mean_norm、lr1e-6、
mini8/micro1/epoch1、KL系数.01、entropy.001。四卡checkpoint应为13个非空文件。

新部署 `/root/new-scii-kl-integrated-20260906/code-<SHA>`，复用44固定Python及
共享权重。独立tmux `qkl_int_l6_r6`、Ray `/dev/shm/qkl_int_l6_r6`，共享本experiment
runs目录。不重用r5。预计45–90分钟；3小时仅运维上限。先CPU原行为等价测试、
资产/有效配置预检，然后在整合四步成功完成并确认空卡后接续；若前序失败，
记录原因并报告，不擅自追加采样、重启或放行长训。

验收分别报告：生成/重算未截断概率差异、top-p后概率差异、零更新eval/train
差异及逐次更新漂移；必须覆盖4个rank的全部有效训练token。统计非有限值、
最差token、完整trace和checkpoint，不使用单个KL阈值自动宣布健康。整合
训练本身的低胜率与前向数值问题分开，不用一次胜负判定KL实现正确。

首次候选`bf54fe5`的CPU全套为77/78通过；新测试把TensorDict当普通dict迭代，
在比较原字段时触发TypeError。修正测试为显式`.keys()`后重跑，不修改生产
实现、不放宽概率或RNG等价断言；旧CPU日志保留在独立部署目录。

最终冻结候选`1fc9f33`在44固定依赖下78/78通过、无跳过，7.499秒。执行命令
为 `CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=<source>:/root/new-scii-spb2-level5-control/python /usr/bin/python3 -B -m unittest discover -s tests -v`。
全部源码/模型/SC2/地图哈希通过；130项有效override排除路径后与当前四步
仅差4→1步和两个观测开关，未修改其他参数。见 [接续准备证据](ready-node44-r6.json)。
13:46北京前序仍未结束，首批8局1胜，step1正在保存；专项尚未占卡启动。

## r7：同节点并行单步（completed；一致性未通过）

用户批准把同一专项移至44新释放的GPU0–3，GPU4–7整合四步继续运行。
14:15二次只读检查0–3无compute进程，主机available约872GiB；并行共享
CPU/NFS可能增加耗时，因此不把这次吞吐与独占节点结果直接比较。
原串行自动接续已暂停并取消启动r6；r6的冻结代码、预检和记录不删除。

RUN_ID `20260906-qwen-kl-metric-v1-qwen-instruct2507-l6-4gpu-integrated-trace-rl1-s0-r7`，
tmux `qkl_int_l6_r7`、Ray `/dev/shm/qkl_int_l6_r7`、外层日志
`/root/new-scii-kl-integrated-20260906/launch-r7.log`。新commit独立部署，
输出沿用本experiment共享runs下的新目录。1步、8局、保存step1，预计45–90分钟。

相对已验证78项CPU测试的`1fc9f33`，只改实验启动器的GPU、唯一运行路径及
去除串行等待条件；训练/诊断Python与全部算法配置不变。启动前做shell语法、
CPU回归和dry-run有效配置对照，确认新Ray使用local独立集群。确认模型原始
cdbee75、shaping off、cache开启和两个诊断开关；不用当前整合checkpoint续训。
验收沿用上面的r6定义，旧计划取消不等于旧实验失败。无额外重试或长训授权。

修改前完整备份 `/private/tmp/new-scii-before-kl-parallel-r7-20260906.bundle`，
SHA-256 `dae217835ab239854a347515427b2b48803066a1ddab89403e389fc70d20885d`。

14:24北京正式启动，源码`c49642b`已提交推送私有诊断分支，部署clean。
本地22项启动器测试、shell语法通过；44固定依赖78/78 CPU通过、无跳过，
全部资产哈希通过。130项override排除运行路径后与r6预检完全相同，实跑
又与r7预检完全相同；实际GPU0–3、四卡、1步8局，模型无SFT/RL恢复。
[启动证据](launch-node44-r7.json)记录SHA和哈希。14:24:42仍在Ray/模型
初始化，尚无PPO或策略一致性结果；原整合实验的GPU4–7仍在保存checkpoint。
14:26:44复核四个新Worker分别占用GPU0–3，8个SC2环境已进入首批对局；
原GPU4–7四个Worker PID保持不变。自动任务只跟踪r7与整合四步，不启动r6。

14:33发现[外部显存竞争](resource-contention-node44-r7.json)：另一组cjy_mob
Python于14:30:11–12启动，在KL已占用的GPU0–3各新增约14GiB显存。
KL仍在首批对局、0次更新，尚无trace或OOM证据；进入actor更新后存在OOM
风险，不能把后续资源失败归因为模型/KL。已通知用户协调，没有停止任何进程。
整合四步已完成2/4，前两步checkpoint各13文件完整，16局1胜、无截断。

## r8：48 独立两卡单步（held before launch）

用户随后要求先启用history_length=5并提供0134历史结果，因此此无历史候选
尚未提交部署或启动GPU。23项本地启动器/记录测试通过（1.167秒），不代表
两卡GPU通过。当前仅保留薄启动器准备，待历史输入语义确认，不自动启动。
历史核验见 [history-window审计](../../../docs/reference/history-window-audit-20260906.md)。

09-06 15:03北京用户明确要求不停止44任务，另在48启动。48的GPU0–3曾于
14:59释放，但15:03新外部任务占用0/3等卡；15:04只确认1、2为空，故按用户
提出的两卡方案使用这两张卡，不与其他任务共用GPU。启动时再次检查空卡。
44的r7及整合四步继续，均不停止、不原位修改。

入口 `run_node48.sh`，RUN_ID为
`20260906-qwen-kl-metric-v1-qwen-instruct2507-l6-2gpu-integrated-trace-rl1-s0-r8`。
沿用r7模型、Level6/g2/seed0、8 rollout、1步、每步保存及全部训练/诊断参数。
唯一非路径有效配置变化是world size 4→2；全局mini8、micro1不变，已有worker
归一化使每rank mini2→4、梯度累积2→4。FSDP分片、生成分组、随机数消费及
旧batch-copy凑整会随rank数改变，不声称与四卡逐位等价或用于吞吐对比。
四卡实际allocated峰值41.23GiB、reserved71.92GiB不是两卡显存保证；两卡
若OOM仅记录资源/可行性失败，不自动缩短prompt、关cache或改变loss后重试。

复用48已批准复制的 `/root/new-scii-qwen-kl-metric-20260906/python` 及匹配头文件，
不安装依赖或复制/下载权重。新部署位于 `/root/new-scii-kl-integrated-20260906`
的commit目录，tmux `qkl_int_l6_r8`、Ray `/dev/shm/qkl_int_l6_r8`，输出仍为
本experiment共享runs下新目录。预计60–120分钟，3小时运维上限；需两个rank
完整诊断、实际更新和7个非空checkpoint文件，`summarize_trace.py --ranks 2`。
本轮不启动长训、不引入新方法；参数来源及验收逻辑均复用现有FSDP/诊断实现。

修改前bundle `/private/tmp/new-scii-before-node48-kl-r8-20260906.bundle`，SHA-256
`d65757c23272835cdd30276dab21a6739e6b6a6a104ce939965b18f8cde1ef90`。

## r7 完成审计（09-06 15:57北京）

15:36:40北京完成，退出0、1/1 global step，step1的13项必需checkpoint
均非空（另有9项HF配置/tokenizer文件）。8局全部自然Defeat，无截断；
其中1局没有模型决策。真实应用274条回答，现有copy补齐到276条PPO样本，
不是276条唯一决策。四rank各69条、三阶段完整，共3370个有效token位置。
原始四份JSONL、四份张量及哈希、模型分片哈希均见结果JSON；未下载权重。

训练日志reward=-46.592，有效率257/274=93.8%，无response截断；
17条无效回答含10条`- marine x 2`不在canonical候选中。这不是新增协议修复。
日志PPO KL=4.866、reference KL=0.586、clipfrac=0.022、grad_norm=0.620；
这些actor指标只保留rank0，不能再称为全卡均值。

| 观测位置 | 本次真实证据 | 结论边界 |
| --- | --- | --- |
| 零更新eval与train重算 | 276条、3370个有效token的绝对差均值和最大值均为0；第一次实际update也为0 | 本批次未发现起点train/eval偏移 |
| generate未截断概率与eval | 平均差0.00630；p99=0.03195；8个token差值>1；最大3.04131 | 小均值不能掩盖尾部，尚未通过 |
| top-p后概率与eval | 平均差0.00638；最大3.05911 | 含正常重归一化，不要求与eval全同 |
| 后续真实参数更新 | 一个global step有35次optimizer更新；第28次更新后观测到有效EOS差119.24889 | 明显漂移发生在更新阶段，尚未锁定具体loss/数值原因 |
| 全部更新后重测同一批 | 绝对差token加权均值3.68221、最大110.99816 | 不能仅凭梯度有限或clipfrac低判健康 |

最大生成差异来自`- marine x 2<|im_end|>`的结束符：generate raw log-prob
为-0.017803，eval为-3.059113；top-p后为0。更新阶段最差样本为
`- marine<|im_end|>`，有效结束符从-0.001114变为-119.25，advantage为
+0.015833。它不在padding区域。暂不归因为BF16、学习率或cache某一项。

**新定位的跨卡统计遗漏**：`collect_dp_compute_data_proto`走
`DataProto.concat`，后者返回`meta_info=data[0].meta_info`，随后trainer
只归约该字典。四rank的PPO K1分别为4.866162、5.327567、7.488033、6.291207，
同口径全卡等micro均值为5.993242，reference K3为0.776749；日志4.866/0.586
恰好匹配rank0。此前`append_to_dict`修的是卡内覆盖，未修跨卡丢弃。
这仅解释汇总口径，不能解释保存的真实log-prob漂移；本轮没有修改训练代码。

只读分析命令为`python3 summarize_trace.py <r7>/ppo_diagnostics --ranks 4`，
并用保存的CPU张量、有效mask与原tokenizer核对最差token。全部828条诊断
micro记录均有限。训练执行完成与策略一致性通过分开记录：后者仍未通过，
不启动长训、不重试、不改变loss/cache/history/LR；整合四步继续。

## r7 后续：跨卡修复与固定 token 重放

跨卡统计修复 commit `0d79d2d`：仅合并 worker metrics，不改 PPO 更新。
真实 r7 四卡各 69 个 micro 指标经 DP collector 共保留 276 个，PPO K1
5.993242、reference K3 0.776749；旧日志不覆盖。新 Git 部署下完整 CPU
回归 83/83 通过（7.788秒）；首次 tar 暂存因缺 `.git` 失败的12项已重跑。
详见 `metrics-collection-fix.json`。

`replay_fixed_tokens.py` 是独立诊断工具，不接入默认训练。使用原始
Instruct-2507 与 r7 保存的四 rank batch；不启动 SC2、不重新采样回答。
用 HF `LogitsProcessor` 固定历史回答，保留 `use_cache=True`，读取未经过
processor 的 raw logits。依次比较正常 actor 重算、FSDP整段前向+FP32
归一化、summon-full-params整段前向和cached generate，记录实际
embedding参数/激活dtype，以区分归一化、参数精度与cached路径差异。
强制选词的 scores 不是策略概率，不用于 PPO。

只有前8条/rank的原始 log-prob 重现（绝对误差≤1e-5，纯诊断准入条件）
才调用原 `actor.update_policy` 复现至多4次优化器更新。保留原 mini=8、
micro=1、lr=1e-6、entropy=.001、KL=.01、原 old/ref/advantage、mask、
temperature=.4、eager与cache；不重算reward或advantage、不启用history。
trace保存前/中/后概率；不输出可续训checkpoint，不作胜率或学习结论。
若起点不一致则报错停止，不能据不匹配重放声称复现了r7的更新。

固定回答工具已在微型 Qwen CPU 上验证raw概率未被强制选词污染、cached
与整段前向一致（1/1通过，4.170秒）；配置已通过CPU解析。4B/FSDP验证
尚待授权和执行，不把工具测试当作策略一致性通过。

公开接口依据：[Transformers 4.55.2 generate](https://huggingface.co/docs/transformers/v4.55.2/en/main_classes/text_generation)、
[PyTorch 2.7 FSDP mixed precision](https://docs.pytorch.org/docs/2.7/fsdp.html#torch.distributed.fsdp.MixedPrecision)。
这不是新的训练算法或loss改动。拟用44四卡、30分钟上限，批准后使用新
RUN_ID、clean已推送SHA、独立目录；不排96step。

启动器为 `run_fixed_tokens_node44.sh`，拟用 GPU0–3，
RUN_ID=`20260906-qwen-kl-metric-v1-qwen-instruct2507-fixedtokens-4gpu-opt4-s0-r9`。
启动前检查clean Git、GPU无进程、模型/源码哈希；依赖复用44已验证目录。
CPU配置检查确认原始权重、eager、lr=1e-6、top_p=.9、temperature=.4；
原FSDP worker导入成功，不初始化模型。授权尚待答复，尚未启动。

09-06 后续用户已批准两组四卡并行，各30分钟，不启动SC2/长训：
`bash run_fixed_tokens_node44.sh paths` 使用GPU0–3，仅做概率路径对照，
RUN_ID后缀 `fixedtokens-4gpu-paths-opt0-r9`；
`bash run_fixed_tokens_node44.sh updates` 使用GPU4–7，仅复现前4次优化器更新，
后缀 `fixedtokens-4gpu-updates-opt4-r10`。上述未启动的合并式r9方案由此替代。
两组仍各自保持r7的四卡拓扑、每卡mini=2、micro=1，不改变生产配置。
先前 `b721fd7` 独立Git快照完整CPU回归84/84通过（7.926秒）；本次仅拆分
诊断入口和产物目录，启动前再次测试。结果不自动放行任何长训。

r9/r10 于19:16北京在资产预检通过后、worker启动前失败，退出码2。
`torchrun` 将脚本的 `--run` 当作自身 `--run-path` 的缩写，报 ambiguous
option；没有rank日志、没有模型加载或参数更新。这是诊断启动器缺陷，
不是模型或训练失败。补上标准参数终止符 `--`，并用真实torchrun parser
回归原错误与修复。只改参数传递和RUN_ID，重启使用r11/r12，保留旧产物。

`33a2e81` 独立Git快照CPU回归85/85通过（7.900秒）。r11/r12于19:24:46
北京并行启动，19:25实测八卡均已加载模型。r11于19:25:39完成、退出0；
r12起点四rank概率与历史完全一致（最大差0），但在pre-update观测、任何
optimizer更新之前失败：诊断脚本直接调用actor时没有复现原worker的
`data.to(current_device())`，导致CE收到CPU labels。原训练入口已有该搬运，
不是生产训练新bug。仅为诊断prefix补回相同搬运，新编号r13，不重跑已完成r11。

### 两项诊断结论（19:33 北京核验）

[生成路径 r11](fixedtokens-result-r11.json) 完成8个指定回答、135个有效token，
无参数更新。Actor重算与历史old概率最大差0。相同完整输入、FP32归一化下，
FSDP与HF前向最大差3.019；实测embedding权重/激活分别为BF16与FP32。
例：`- marine x 2<|im_end|>` 的EOS在Actor、FSDP+FP32归一化、HF完整前向、
HF cached generate中分别为-3.059、-3.177、-0.157、-0.061。
这定位了明显的前向精度路径差异，而不是证明关闭cache才正确。指定样本包含
历史异常点，不代表整体发生率；cached重放与历史生成仍有最大1.073的残差。

[更新重放 r13](fixedtokens-result-r13.json) 于19:30:33完成，退出0，四rank均
完成4次optimizer更新。源 `7cee3f2` 完整CPU回归86/86通过（7.935秒）。
使用原始模型及r7前32个rank-local样本（每rank8个，共413有效token），起点
eval/train均精确复现old概率，全部更新forward的有效token也与r7逐值相同。
不是重新采样，也不是4个完整rollout step。

| 观测 | 结果 |
| --- | --- |
| 已更新0/1/2/3次时、对应下一mini-batch最大log-prob偏移 | 0 / 0.0797 / 0.0040 / 5.5337 |
| 更新4次后，同一32样本的最大偏移 | 19.0 |
| 更新4次后，有效token加权平均绝对偏移 | 0.10177 |
| 训练forward全rank、micro等权平均PPO K1 / ref capped K3 | 0.00683 / 0.01213 |
| 训练objective clipfrac | 0 |
| 四次裁剪前grad norm | 1.7997 / 0.1332 / 0.0000304 / 0.2004 |

最大偏移位于rank1样本3的空白/换行token `2303`，old=0、new=-19、
advantage=+0.01562。所有有效概率有限，但不能因此称为健康；均值和clipfrac
掩盖了尾部变化。mini-batch行之间输入不同，不将表中四个最大值画成同一
固定样本的逐更新曲线。已在无SC2、无需重跑游戏的条件下精确复现早期漂移，
但尚未证明漂移起因，更未修复。生产loss/LR/prompt/reward/cache均未修改。
两组结束后19:31实测八卡已释放；不自动启动96步。

### 参数精度零更新 A/B（completed）

仅验证现有FSDP `actor.fsdp_config.mixed_precision.param_dtype`：A显式bf16，
B显式fp32。源自 `2526f88` 已完成诊断，沿用本分支/record，以新commit冻结。
不修改生产源码、主线默认值、reference、temperature归一化、loss、LR、
prompt、reward、cache、attention实现或历史窗口。参考
[PyTorch 2.7 MixedPrecision](https://docs.pytorch.org/docs/2.7/fsdp.html#torch.distributed.fsdp.MixedPrecision)
中param_dtype与summon_full_params的全精度语义，不自行改写FSDP。

两组从同一原始Qwen3-4B-Instruct-2507快照启动，读取r7相同四份batch，
每rank两份指定回答、共135有效token，和r11完全相同；固定token只是数值
诊断，不是新的采样/RL数据。各4卡并行：44的0–3=A、4–7=B，19:42北京
只读核验八卡空闲。各最多30分钟；历史同规模r11耗时53秒，预计初始化后
数分钟内完成。若OOM、数值非有限或原始token无法复现则停止，不扩量。

入口 `bash experiments/records/20260906-qwen-kl-metric-v1/run_fixed_tokens_node44.sh precision-bf16`
及 `precision-fp32`；RUN_ID后缀分别 `param-bf16-opt0-r14`、`param-fp32-opt0-r15`。
完整worker配置按rank写入新run，manifest记录SHA、模型/输入来源、GPU与预算，
继续核验已有source/model SHA。CLI限制param-dtype候选必须max-updates=0。

主要判断：相同完整输入、相同FP32归一化时，FSDP/HF前向差异能否从r11的
最大3.019降至1e-5以内。另核对A与r11的Actor输出重现、A/B实际有效配置仅有
param_dtype不同、两边原始模型和输入哈希相同。报告所有135有效token的均值、
最大值、原始回答、embedding dtype、CUDA峰值显存和运行耗时；不从单点
统计推导整体发生率。cached/full与原temperature归一化的残差分开报告，
主指标通过不等于生成/重算全路径通过，更不等于更新漂移已经修复。
这次不运行PPO、不保存训练checkpoint、不启动SC2或96步。

19:48北京两组均完成、退出码0，四rank分别都有completed记录。
源commit `a6ab6e05dad47282a42a12f49960602e3fd264e2`，完整CPU回归87/87
通过（8.301秒）。[完整结果、真实输出与哈希](param-precision-ab-r14-r15.json)。

| 135个有效token上的差异 | A：BF16参数 | B：FP32参数 |
| --- | ---: | ---: |
| FSDP/HF完整前向、同一FP32归一化：最大绝对差 | 3.019297 | **0** |
| cached生成/Actor重算：最大绝对差 | 2.998304 | 0.808783 |
| cached生成/Actor重算：token平均绝对差 | 0.055296 | 0.008560 |
| Actor原归一化/FP32归一化：最大绝对差 | 0.291807 | 0.149077 |
| HF cached/完整前向：最大绝对差 | 0.808782 | 0.808782 |

四rank实际配置排除产物目录后仅param_dtype不同；模型与输入来源一致，
指定回答一致，A精确复现r11的Actor值。HF完整前向与cached固定token概率
在A/B间逐值相同，变化确实发生在FSDP重算一侧。B的embedding权重/激活
实测均FP32，而BF16 autocast与生成cache继续保留。

例：回答 `- marine x 2<|im_end|>` 的EOS，Actor从-3.059113变为-0.142937；
HF完整前向仍为-0.157486，cached仍为-0.060808。这是参数精度不一致的
有因果对照的局部修正；主指标通过，但剩余归一化与cached/full差异未解决，
不能称为整个策略一致性通过，更没有验证更新后的19级偏移是否消失。

A/B各耗时56.31/53.98秒。两组最大CUDA allocated峰值均33.72GiB，包含
初始化；reserved分别42.56/35.41GiB。这只是一次冷启动短诊断，不代表FP32
训练更快或更省显存。没有OOM或非有限值，生产默认设置与旧结果未改。
下一项候选应单独验证response logits温度归一化精度；此次到此停止，
不修改cache、不新增游戏、不自动运行参数更新或长训。

### 温度归一化精度 A/B（completed，用户追加授权）

在已验证的FP32参数配置上，仅切换上述response logits归一化开关。
A `norm-legacy-opt0-r16` 使用原BF16温度除法；B `norm-fp32-opt0-r17` 先转
FP32再除同一个0.4。44的0–3和4–7各四卡并行，各最多30分钟，仍为r15
相同8回答/135有效token、零更新；模型、输入、loss、cache、LR、游戏配置
均不变。新源码提交/推送后冻结部署，旧r6/r7/r15源码与产物不修改。

主要判断是Actor/同一完整前向FP32归一化最大差是否由0.149077降到1e-5内，
并核对两组有效配置只差该开关，参数路径差仍为零。单独报告cached/full和
cached/历史generate残差，不能把归一化通过说成完整生成一致性通过。
若通过，再准备同原始模型、同固定token的最多4次更新诊断：old/ref必须按
各自起点重算，不能把历史BF16概率混入新FP32策略；不重跑游戏、不续训历史
checkpoint、不自动启动长训。更新诊断以独立RUN_ID和有效配置记录。

20:07北京r16/r17均完成、四rank completed、退出0。源 `1f62491`，完整CPU
88/88通过（8.266秒）。归一化最大差0.149077→0.000002804，token加权均值
0.00132635→0.0000001279；达到预定1e-5门槛。FSDP/HF完整前向仍逐值一致。
cached/full最大差仍0.808782，cached/历史generate仍1.073228；二者没有被
温度修复消除，不能称完整策略一致性通过。
四rank有效配置核验仅该开关不同，模型输入哈希相同，HF full/cached概率
在A/B间逐值不变。[完整结果及日志哈希](temperature-ab-r16-r17.json)。

### 同一起点更新漂移 A/B（completed）

沿用FP32参数与上述单一温度变量，入口 `drift-legacy` / `drift-fp32`，
RUN_ID后缀 `norm-legacy-opt4-r18` / `norm-fp32-opt4-r19`。44同两组各4卡，
各最多30分钟，重读原模型，使用r13同32个回答/413有效token/advantage，
最多4次optimizer更新。学习率、AdamW、loss及所有游戏配置保持原样。

两组分别在更新前重算old/ref，ref是原始Actor权重按同一路径的概率快照，
不是另一个reference worker验证；固定token不是从候选重新采样，不作为
on-policy训练或胜率结论。禁止修改历史batch，新的old/ref随新run保存。
更新前必须train/eval有效token误差≤1e-5且有限，再检查实际pre-update trace；
报告更新后同一批token的最大/加权平均偏移、尾部token、梯度与clipfrac，
与r13的19级漂移对照。未解释残差不放行长训，不调整学习率或关闭cache。

20:14北京两组完成、退出0；源 `97495f4`，CPU89/89通过（8.289秒）。
两组实际配置四rank均仅归一化开关不同，初始eval/train和实际pre-update
均差0。完整4次更新，413有效token全部有限。
[原始指标、最差token与哈希](temperature-drift-ab-r18-r19.json)。

| 同一32回答、更新4次后 | 原温度除法 | FP32温度除法 |
| --- | ---: | ---: |
| 最大log-prob偏移 | 13.495918 | 0.228683 |
| token加权平均绝对偏移 | 0.033813 | 0.000999 |
| 训练forward平均PPO K1 | -0.0000167 | 0.001010 |
| 训练forward平均ref capped K3 | 0.00000114 | 0.000132 |
| objective clipfrac | 0 | 0 |
| 最大裁剪前grad norm | 0.234385 | 0.605244 |

原组最差rank0样本3的token28591由-0.00408变成-13.5，优势仍为正。
新组本短前缀不再出现十几级漂移；但训练forward平均KL更高，说明不能仅
凭KL均值判健康。该A/B支持温度精度对本短段尾部漂移的修复价值，不保证
完整35次更新或长训稳定；未验证新reference worker或重新采样的策略一致性。

### cached/full 残差分解（completed）

r18/r19释放GPU后，用 `residual-bf16` / `residual-fp32`（r20/r21）各四卡
零更新对照。两组Actor均FP32参数+FP32温度归一化，仍8指定回答/135token。
唯一变化是诊断脚本内HF完整前向和cached生成是否启用BF16 autocast；B用
FP32运算。FSDP actor、生产HF rollout、cache=True、模型/输入均不修改。
各最多30分钟；不是推荐或启用生产全FP32训练。先记录TF32设置和实际logits
dtype，再判断cached/full最大差能否由0.808782回落至1e-5。若不能，继续将
模型逻辑、输入position/mask和数值误差区分，不能把残差全归咎于BF16。
HF full/FSDP的比较在B中故意不同运算精度，不作为这组的通过门槛。

20:22北京r20/r21均完成退出0，源 `4fafa37`，CPU89/89通过（8.198秒）。
两组TF32均False、matmul precision=highest。cached/full最大差
0.808782→0.000231683，token均值0.00764169→0.00000262144；大幅下降，
但仍未达到1e-5预设门槛，不标通过。生产cache和autocast均未改。此结果
支持残差主要与前向运算精度有关，不能证明每个模型层的逻辑都正确。
浮点批量/切片运算并不保证逐位相同，参考
[PyTorch 2.7数值精度说明](https://docs.pytorch.org/docs/2.7/notes/numerical_accuracy.html)。
[完整数值、dtype与哈希](cached-precision-ab-r20-r21.json)。B的HF/Actor最大
差反而为2.554，因为只提升了诊断HF一侧；绝不能把这种单侧改动作为生产修复。

下一组 `head-control` / `head-observe`（r22/r23）仍各4卡、最多30分钟、零
更新、同135token，只在B的诊断hook中额外计算FP32输出投影概率，不替换模型
输出、不改变隐藏层或生成路径。先核验A/B原有全部概率逐值相同，再比较
相同hidden state经过FP32 head后的cached/full差异，以定位是否只需处理
末端投影精度。不是生产lm_head修改，也不自动启用全FP32训练。

r22于20:30完成，但r23诊断报错退出1，只有54/135有效token观测，不能作
完整对照。HF的FSDP同步生成会在某rank已EOS后继续forward；hook收到27次
而该rank真实回答仅3token，观测拼接报长度错误。只将shadow观测按真实
回答长度取有效前缀，和已有generate概率/mask相同；不改变模型生成过程。
加入真实赋值语句的3/27长度回归，新编号r24/r25，旧失败不覆盖。
[失败与部分产物哈希](head-observation-r22-r23.json)。

20:36北京r24/r25均completed、退出0，源 `946ebc2`，全CPU91/91通过
（8.266秒）。两组完整135token的所有原始概率逐值相同，观测无扰动。
shadow FP32 head将cached/full最大差0.808782降为0.505378，平均绝对差
0.00764169降为0.00497428；仍有明显残差，不能只改lm_head宣称解决。
[完整结果与哈希](head-observation-r24-r25.json)。

本轮已验证的最小行为修复是response logits温度归一化，保留为显式候选，
配合前一轮已验证的Actor FP32参数路径；不合并未验收的生产精度默认值。
短更新从最大13.50降到0.229，但不是完整rollout/PPO更新验收。生成残差仍
需定位主干隐藏状态的逐层数值差异；新reference worker尚未对齐验证，
完整35次optimizer更新、重新采样与真实游戏训练尚未验证。不启动96步。
全FP32 HF与shadow head均只存在于固定token诊断，不是生产变更；没有
关闭生成cache、调整LR/loss/reward/prompt、改history或覆盖旧checkpoint。

### 继续解决缓存残差：逐层观测及GEMM累加对照（completed）

用户明确要求继续修复，沿用r25的原模型、FP32参数/温度、BF16 autocast、
cache=True和同8回答135token。44在20:52北京实测八卡空闲。A `layers-control`
用0–3卡、r26；B `layers-reduction` 用4–7卡、r27，各最多30分钟、零更新。
唯一行为变量是PyTorch现成 `allow_bf16_reduced_precision_reduction=False`，
不更换attention backend，不调整loss/LR，不编辑模型资产或共享依赖。
依据：[PyTorch 2.7 BF16 GEMM累加精度说明](https://docs.pytorch.org/docs/2.7/notes/numerical_accuracy.html#reduced-precision-reduction-for-fp16-and-bf16-gemms)。
这只是待证伪候选，不能预先认定低精度累加是根因。

两组相同hook记录embedding、36层attention/MLP/残差输出、最终norm及第0层
Q/K/V投影和Q/K norm；严格对齐预测同一个token的位置，并排除EOS后空转。
保存逐层最大/平均/RMS及逐token最大差，不改变任何输出。先核对A与r25原概率
逐值一致，再比较A/B最终概率与首个出现差异的算子。仍以1e-5为原门槛，不
因试验失败放宽门槛；数值非有限、OOM或无法复现输入则停止并保留失败。

21:00北京r26/r27完成退出0，源 `32c6a7c`，CPU92/92通过（8.282秒）。
部署曾因增量bundle缺少4b9fb95前置提交失败，补齐链后才启动；未产生GPU
失败run。BF16低精度累加False使全体最大log-prob差反而由0.8088增为1.7656，
不采用该设置。仅看rank2样本44的0.8088→0.0493会得出错误的乐观结论。
[完整结果与哈希](layers-reduction-ab-r26-r27.json)。

该样本embedding精确一致；第一层input norm最大差2.98e-8，q_proj差
0.0009766，q_norm差0.03394；第0/8/18/35层残差输出最大差分别
0.0078125/0.08984/0.21484/4.35352。说明差异在进入cache更新之前的
归一化/投影已经出现，不能只归咎KV存取逻辑；也不能据此排除其他数值来源。

### 逐行RMSNorm最小候选（completed, rejected）

使用已有FlashAttention 2.6.3（BSD-3-Clause）的rms_norm_fn，来源
`flash_attn/ops/triton/layer_norm.py`，安装文件SHA256
`33de94d251e1825c02dad999d32fcb39f01584cd17753a28f132bfbdbd612590`，
[公开实现](https://github.com/Dao-AILab/flash-attention/blob/v2.6.3/flash_attn/ops/triton/layer_norm.py)。
薄适配器先用单位weight求FP32归一化，再cast回原输入dtype，最后乘Qwen的
真实weight，保持HF 4.55.2原公式和转换次序。不开启attention fused backend，
不编辑第三方安装或权重；只在诊断中显式调用安装函数，生产默认不变。

r28 `rms-control` 44卡0–3，r29 `rms-rowwise` 卡4–7；恢复PyTorch默认累加
设置，不夹带r27失败选项。单一变量是RMSNorm归约后端，同样135token/零更新。
B先做180秒内实际CUDA kernel输出/输入梯度/weight梯度对照与整批/单行一致
检查；FP32相对L2≤1e-5、BF16≤0.02且全有限才继续，BF16容差只用于独立
算子回归，绝不放宽最终策略一致性门槛。后续probe最多1600秒，总GPU预算
不超30分钟。所有层相同hook，比较首个分歧及最终cached/full，不能只选收益
最大的样本。若通过再用新起点做短更新，当前不启动新游戏或长训。

21:20北京两组完成退出0，源c783729，CPU93/93（8.350秒）。真实CUDA核的
输出/输入梯度/weight梯度及整批/单行检查通过，四rank各替换145个RMSNorm。
但全体cached/full最大差0.808782→0.819685、均差0.00764169→0.00714165，
不能采用为残差修复。rank2样本44第一层input norm从2.98e-8降至0，但q_proj
仍差0.0009766；最终该样本改善被另一个样本恶化抵消。
[完整结果](rms-ab-r28-r29.json)。候选保持诊断opt-in，生产默认未改变。

### 统一模型前向精度与短更新（completed, rejected）

r30 `model-control` 用44卡0–3；r31 `model-fp32` 用4–7。两组回到原生RMSNorm，
保留已测FP32参数/温度、相同输入，先测8回答135token，再从原权重重算old/ref
做4次optimizer更新（不是4个RL步）。A与r19相同设置，B唯一变量是在共享模型
forward上应用PyTorch `autocast(enabled=False)`，覆盖Actor、full和cached路径，
梯度checkpoint重新计算应继承该精度。生成cache=True、权重不变，不改第三方
文件。与r21仅提高HF精度不同，本轮不让Actor仍用另一个计算精度。

每组最多30分钟；更新前train/eval误差≤1e-5且全有限才进入短更新。cached/full
仍按原1e-5门槛报告；即使减少到全FP32数值量级但未过门槛，也不得标验收通过。
这次更新只诊断稳定性，不是采样自新策略的在线RL；reference仍是原Actor快照，
不冒充独立ref worker。记录全路径dtype、残差、更新漂移和耗时，不预先将全FP32
作为生产方案，不启动新游戏、长训或修改lr/cache/history。

21:31北京两组全部完成、退出0；源83ac72e，CPU94/94（8.592秒）。四rank的
worker配置除诊断输出目录外相同，B的唯一运行变量为model forward禁用AMP。
A的四rank训练trace与前一轮r19逐字节一致，排除了新增路径观测改变对照更新。

| 指标 | BF16模型前向 r30 | 统一FP32模型前向 r31 |
|---|---:|---:|
| cached/full最大log-prob差（135token） | 0.808782 | 0.000231683 |
| cached/full平均绝对差 | 0.00764169 | 0.00000262144 |
| FSDP/HF整段计算差 | 0 | 0 |
| 新起点train/eval最大差 | 0 | 0 |
| 4次更新后最大old/new差（413token） | 0.228683 | 112.026398 |
| 4次更新后平均绝对old/new差 | 0.000999488 | 1.738758 |
| 训练microbatch平均PPO k1 | 0.00100953 | 3.960609 |
| 训练microbatch平均reference capped-k3 | 0.000132106 | 1.308578 |
| 最大裁剪前梯度范数 | 0.605244 | 0.497134 |
| 带观测的更新阶段耗时 | 35秒 | 132秒 |
| 整次运行耗时 | 102秒 | 250秒 |

这些数据分别是路径误差、参数更新漂移和训练期统计，不能互相替代。两组全有限，
但B明显不稳定，不采用为生产修复；没有关闭生成cache，也没有把B权重续入训练。
B缓存路径残差缩小约3491倍，却仍未达原1e-5门槛；更新变坏也独立说明，零更新
数值更一致不保证训练更稳定。B首个optimizer更新之后，rank0 sample3第1个
response token的log-prob从-0.005394变为-73.030792；四次更新后变为-112.031792。
更新诊断中的log-prob来自整段forward，不使用生成cache，不能把这段漂移再归因
给KV cache。其具体梯度/参数原因尚未证实，不归咎lr、模型能力或SC2。

[完整结果、配置核对及哈希](model-precision-ab-r30-r31.json)。PyTorch明确说明
[批量与切片计算不保证逐位一致](https://docs.pytorch.org/docs/2.7/notes/numerical_accuracy.html#batched-computations-or-slice-computations)，
这支持区分数值差异与缓存逻辑错误，但并不自动证明当前残差可接受。统一精度
采用[官方autocast局部禁用接口](https://docs.pytorch.org/docs/2.7/amp.html)，仅诊断。

本轮结论：温度FP32局部候选与Actor FP32参数路径仍是已有有限验证的组合；
低精度累加开关、逐行RMSNorm、全模型FP32均未成为可用新增修复。缓存路径尚未
验收，完整更新/独立reference/新rollout亦未验证；不合并这些候选到生产默认，
不启动96步。下一次修改前应先定位首个optimizer更新的参数变化与梯度证据，
不能通过继续叠加算子替换、放宽门槛或调整lr来宣称修好。

### 夜间优先任务：完整35次更新（r32 failed/incomplete，用户新指令取代前节下一步建议）

2026-09-06晚用户明确要求不再追加零更新候选，下一项必须是完整历史batch。
r32 `full-updates` 使用44卡0–3，保留历史四rank分配；每rank69条，本地mini2，
共35次optimizer更新，最后1条不额外复制、不丢弃，保持历史grad accumulation=2。
原r7已有274真实样本补成276条的2条copy也原样保留；本次是数值重放，不能
把它说成274条唯一样本的干净在线RL计权验收。未来真实RL必须单独审计此边界。

配置为原始Qwen3-4B-Instruct-2507、FSDP参数FP32、FP32温度归一化、BF16 autocast，
生成cache保留；不加载RMSNorm或全模型FP32候选，不改变lr/loss/优化器和SC2系统。
在原权重以此配置重新算完整old/ref；此处ref是原Actor快照，独立ref worker仍待验。
`--skip-paths` 跳过重复零更新生成诊断，直接准备完整更新。最多30分钟；预计带
每步全batch观测约5–15分钟，不是35个global step、不启动SC2、不保存RL checkpoint。

现有PPO观测记录每个训练microbatch，并在每次optimizer更新后重新观测完整69条/
rank，保存到 `post-each-rank-XX.jsonl`。新增仅诊断observer记录实际backward的
有效token d(loss)/d(log_prob)、每rank参数梯度shard范数/最大值和原始全局裁剪前
grad_norm。token梯度包含PPO+reference KL，不含独立entropy分支；参数梯度包含
完整实际loss，二者不能混称。无额外backward，不换loss，观测不改变RNG、梯度
或参数；CPU测试执行真实update_policy比对开关前后逐值相同。

原1e-5不再作为cached/full唯一成功标准；这不等于擅自放宽验收。完整更新后
报告全有效token分布、极端值、ratio和裁剪影响，严重未解释漂移不放行；新起点
train/eval仍对齐，若失败记录事实，不退回零更新精度搜索。35步完成后才决定
Level6 4个global step（8 rollout/step）→8-step→同种子独立评测→条件式96-step。
更改冻结模块/资源预算/验收标准需先询问用户，失败记录保留。夜间接续任务sc2
每15分钟在本任务核对进展，上午09:00 Asia/Shanghai交付总结后删除；无变化安静。

21:47北京r32已启动，源b671bb4，部署`code-full-r32`，CPU94/94（8.410秒）。
完整新起点四rank train/eval均为0。逐次全batch观测实测使每次更新间隔约58秒，
总时长估计37分钟，超过自己设置的30分钟上限。21:54已向用户请求允许45分钟
新RUN_ID重跑；未收到批准前不停止r32、不修改其deployment或timeout。
`full-updates-45`/r33仅提前准备，等待用户预算批准后才启动；没有增加新候选。
该耗时低估与测试适配返工是诊断工作问题，不算项目原有bug修复成果。

离线分布入口：`python3 summarize_full_updates.py <RUN_DIR>`，从run内保存的新
old概率和每次更新后的同批完整观测计算p50/p90/p99/max，排除padding、读取有效
config中的clip界限，单独标记尚未完成四rank各69条的观测。输出不是在线训练
胜率或独立评测，也不自动作出通过判定。

22:17:35北京r32触发冻结的1800秒timeout，退出124：四rank均完成28/35次
optimizer更新，前27次各有完整276回答/3370有效token后验观测；第28次只观测
到每rank54条，不能混入完整batch对比。没有SC2运行或可续训checkpoint。
这是诊断运行预算低估导致的未完成，不是模型/游戏崩溃；四卡已释放。
原部署、输入与失败产物全部保留。r33的2700秒重跑入口已准备、未启动，
仍待用户预算批准，不能自行延长冻结run或跳过35次门槛进入真实RL。

完整batch的后验概率变化如下（都相对本配置重算的old，不是cached/full误差）：

| optimizer更新 | 平均绝对Δlogp | P99 | 最大值 | 超出ratio上下界 | 目标clipfrac |
|---|---:|---:|---:|---:|---:|
| 4 | 0.008959 | 0.035868 | 10.380890 | 0.504% | 0.326% |
| 8 | 0.042708 | 0.412493 | 14.123047 | 1.157% | 0.534% |
| 14 | 0.202558 | 10.000046 | 34.332001 | 2.315% | 0.979% |
| 27 | 3.231027 | 94.157559 | 112.031029 | 5.430% | 1.662% |

此前r19/r30的4更新最大0.229只覆盖32回答/413token，不能代表此处完整batch。
r32前4次训练trace及相同32回答观测与r19逐值一致，未发现新observer改变更新。
第27次3370token中151个已超出原ratio指数截断范围；最差是rank3/sample30的
EOS，old=-0.000221、new=-112.03125，advantage为正0.012839。
正advantage样本也被严重压低，因此不能简单解释成正常抑制负advantage动作。
不过这些后验观测仍不是独立评测，不能据此量化胜率变化。

28次裁剪前global grad_norm全有限、四卡逐值一致，最大20.474855（第25次）；
第14次14.992576，各rank实际参数shard范数为3.823/9.664/10.416/2.876。
冻结FSDP是单flat parameter分片，这些证据不能定位到具体Transformer层。
实际训练的224条microbatch平均PPO k1为2.361083、reference capped-k3为
0.528505、目标clipfrac为0.632%；这是训练时不同参数状态的统计，不与上表
固定同一参数状态的全batch后验统计混称。

`python3 audit_loss_gradients.py <RUN_DIR>`在CPU按冻结PPO/KL实现重算实际
d(loss)/d(selected log_prob)，224条记录最大误差4.66e-10。部分严重偏离token
已触及既有ratio指数截断和KL上限，实际两项梯度均为0；这解释了该区域纠正力
消失，但尚未解释最初漂移来源。该核对不含独立entropy分支，也未验证模型
Jacobian、Adam状态或证明修改loss有效；未调整loss、LR、cache或模型算子。
审计脚本曾把global mini8误当成本地mini8，现按world_size4得到本地mini2，
两项CPU回归通过（3.319秒）；这是诊断脚本纠正，不是项目原有bug修复成果。

[完整配置、逐次全卡分布与产物哈希](full-updates-r32.json)、
[逐microbatch实际梯度核对](full-updates-r32-loss-gradient-audit.json)。
结论：FP32参数/温度组合仍仅局部有效，未通过完整更新/连续RL验收；存在已
观测但未解释的严重更新后漂移。没有启动Level6 4/8/96-step或宣称修复完成。
