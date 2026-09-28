# 反馈短 RL 后的同条件独立评测

2026-09-11，状态：r2三组独立评测全部完成，24局自然结束，三组退出码0、0次PPO。
原始模型/A4/B4分别3/8、5/8、2/8胜；完整结果和只读分岔分析见
[评测分析](learning-eval-analysis.md)。43 GPU0–3于21:34核验已释放。
此前r1在Ray初始化失败，0局/0 PPO；失败记录保留，r2经过用户重新确认后启动。
本轮不改生产模块或奖励，不晋级长训。继续沿用本 record 和分支，旧部署保持只读。

## 要回答的问题

反馈接口接通不等于训练学会纠错。比较原始 SpB、反馈关闭 A 的 step4、反馈开启 B
的 step4，在**相同且开启反馈的评测接口**下能否纠正失败、减少重复阻塞，以及游戏
结果有何变化。A/B 训练差异仍只有反馈；评测差异只有权重来源，不再混入接口差异。

训练来源是 runtime `361cb7ae7fd86fe8811d6fcb6946d0241bf93e11` 的 r2：
`20260911-execution-failure-feedback-v1-spb2-original-l6-g2-{A|B}-rl4-4gpu-s0-r2`。
只使用 `checkpoints/global_step_4`；原始组使用相同原始 SpB 资产。
不得替换成 r1 不完整分片、Level7 step24 或其他历史权重。
先核验 result/.done、SHA、world4 和完整分片；这不是实际加载通过。
评测 hook 在原分布式 loader 返回后核对 global step0/4，保存加载证据；纯评测返回，
不执行第五步。沿用 loader 会恢复 optimizer，但不进行更新，也不续接 SC2 状态。

## 配对条件与预算

已批准43 GPU0–3串行三组，各8局，合计24局；总上限2小时、最多8 GPU小时。
等当前B完成、checkpoint完整且卡空闲后启动，不占用其他任务。
两份step4已经是四rank；不改变checkpoint布局，不再把三组模型并发塞入主机RAM。
不复制/生成大权重，新增仅评测日志和回放。不自动重试或续长训。
这份预算已单独确认，不使用此前短RL的3小时授权。

按 env index 固定实际种子：
`[42613, 52620, 17089, 4018390961, 3236598124, 3297512220, 773351930, 1098341402]`。
前三个分别是短RL失败场景、人口阻塞对照、既有执行顺序诊断；后五个来自
`numpy.random.RandomState(20260911).randint(65536, 2**32-1, 5)`，在评测前声明。
后五个不与本轮四个训练seed重合，但未穷举全部旧日志，不冒称全项目从未见过。
已知诊断seed和新声明seed分开汇总；每seed每权重仅一局，不宣称统计显著。
每rank首次生成前统一 Torch seed=rank；之后不逐call重置。模型权重不同可能导致
后续query和采样消耗分岔，不宣称三组后续输入或token逐字相同。

其余冻结：Level6/Harder、Flat128、Terran/Air、g2资产、Planner从开局接管，
原阶段待办、约15列表条目/数量展开/累计配额说明、过滤/补齐/微操/自动战斗，
history0/off、step shaping off、FP32参数/温度归一化、BF16 autocast、cache开启、
eager/HF、分层FSDP、swa_dropout0，T0.4/top_p0.9/top_k0、4096/192 token。
仅复用既有只读每局观测/命令采集；没有手写回答、强制Depot顺序或新战术干预。

## 入口和产物

独立冻结源码入口 `run_learning_eval_node43.sh` 复用既有 `run_feedback_eval_node47.sh`。
文件中的47是旧命名，新控制器显式设置43的本地资产/依赖/四卡目录。
当前r2启动命令只在重试获批、source已提交和私有备份后使用：

```bash
SOURCE_SHA=<新冻结SHA> SC2_FEEDBACK_LEARNING_EVAL_AUTHORIZED=1 \
EVAL_CONTROL_DIR=<未使用的控制目录> \
timeout --signal=INT --kill-after=15s 7125 \
bash experiments/records/20260911-execution-failure-feedback-v1/run_learning_eval_node43.sh
```

三组r2 RUN_ID为 `20260911-execution-failure-feedback-v1-spb2-{original|A4|B4}-l6-feedback-on-eval8-r2`。
输出沿用43本地新项目 runs，Ray/cache按model label隔离，不关闭生成cache。
旧run存在即拒绝；结果、退出码、种子、checkpoint加载证明、完整query/原始回答、
游戏PID/队列/单位观测和回放分别保存。控制器任一失败即停止后续组，不重复尝试。

若B仍在运行，可由同一冻结目录的 `queue_learning_eval_node43.sh` 在受控tmux等候。
额外设置 `EVAL_QUEUE_DIR=<新排队状态目录>`；它只读r2控制完成标记，等待最多到
原3小时截止后60秒。r2退出非零、未完成或GPU未释放则不启动评测；不杀进程。
评测自己的7125秒timeout加15秒退出窗口只在接续时开始，不把排队报为已经评测。

按同seed报告胜负、整局累计reward、真实调用/有效率、反馈次数、重复失败、
反馈后计划变化、实际提交与产出。没有遇到失败的局不纳入“纠错成功率”分母；
没有模型决策的局保留在整局统计。奖励按既有纯评测episode ledger读取，不能
将终局占位日志再加一次。回答变化、正确顺序、命令提交、实际建成分别标记。

若B没有改善，不立刻改prompt/优化器。后续用户已将下一项限定为既有日志的
“第二条计划→生产→首次阻塞→损失”只读对齐；原先过程奖励想法不实施。
若还需因果对照，仅另报第二条回答替换的单变量配置和预算，不自动上卡。
有限梯度、有效率或少量胜负不能代替独立学习验收。

## CPU 验证

`test_feedback_learning_eval test_feedback_eval test_feedback_rl_launch test_execution_failure_feedback`：
36项，35通过，1项因本地缺Torch跳过，2.577秒、退出0；包含三组effective config仅模型
选择/恢复路径不同、四卡八局显式覆盖、固定种子只用一次、step0/4/统一反馈选择、
缺失checkpoint拒绝、终局奖励不伪造决策、控制器串行/预算边界。
旧step24入口的默认种子/四局及A/B协议不改。Torch setup hook验证未执行，
部署前须用43固定依赖且隐藏CUDA补验；CPU结果不是模型实际加载成功。

## r1启动失败与单点准备修正

冻结SHA `9ae43eae689b9d69eb5120ffb77249ffc2d81d34`，43独立目录
`/root/new-scii-execution-feedback-20260911/code-9ae43ea`。私有完整bundle已两端校验：
`45104a8c517cf6d0c7282a9b95bc77a88d362163586f301287fb28b7c53b7153`。
43固定依赖的26项CPU回归全部通过，4.672秒；另外核对三组setup hook均未提前
初始化CUDA或导入trainer，23个核心源码哈希通过。没有更改旧运行目录。

20:07:53北京时间启动受控tmux `sc2_feedback_learning_eval_r1`；此前r2短训已结束，
GPU0–7均0MiB/0%，GPU查询退出0。A/B的完整step4小文件/分片长度检查通过。
20:08:24.712原始组在Ray初始化退出1；没加载模型、没启动SC2、没有回答或PPO。
直接报错：

```text
OSError: validate_socket_filename failed: AF_UNIX path length cannot exceed 107 bytes:
/dev/shm/sc2_feedback_learning_original_0911_r1/session_2026-09-11_12-08-23_497678_3823173/sockets/plasma_store
```

控制器停止，A4/B4未启动。失败原始run、控制目录`learning-eval-r1`及
`learning-eval-queue-r1`保留；run console SHA256为
`401701bdfbd1afaf882ee78114618aa90d9547e948f285e11d9fa0605c92280c`。
这是本次诊断入口命名返工，不是项目原有训练bug的修复成果。

r2仅缩短Ray路径为`/dev/shm/fl_{original|A4|B4}_0911_r2`，更换唯一RUN；
扣除首次约32秒，将重试上限降到119分钟，不重置为新增两小时。
新增回归用最大10位PID/session名验证三组socket路径均不超过107字节，且旧路径失败。
本地37项回归：36通过、1缺Torch跳过，2.532秒。生产模块、模型/游戏/评测配置未改。

r2候选已冻结为 `937ae3cfcbf7340208a8f86c0abf680d733c0484`，43独立目录
`/root/new-scii-execution-feedback-20260911/code-937ae3c`保持clean。
私有完整bundle本地/43哈希一致：
`44ba6d890ce4ee824fff2d70b232bb01a74a9a665db108a5658354727a172889`。
43隐藏CUDA执行 `test_feedback_learning_eval test_execution_failure_feedback
test_feedback_eval.FeedbackEvaluationTest.test_setup_hook_defers_trainer_and_does_not_probe_cuda`：
27项全部通过，4.642秒。重试仍待批准；没有新GPU评测、模型加载或游戏。

## r2启动

用户确认后从上述937ae3c部署直接启动 `sc2_feedback_learning_eval_r2`，未再修改源码。
控制目录`/root/new-scii-execution-feedback-20260911/learning-eval-r2`；20:17:07开始，
全部三组共同硬截止22:16:07北京时间。启动前GPU0–7均0MiB/0%，没有其他GPU任务；
旧SC2遗留进程未擅自清理。本次只用已授权GPU0–3，不在其他四卡并发恢复模型。

20:18:59已创建Harder/Terran/Air实局，四rank原始11分片加载完成，开局Planner调用
可见。`checkpoint_evaluation.json`记录global_step0/val_only，不加载历史RL权重。
八个实际game seed与声明逐一相同。A4/B4完整分片预检通过，仍待原loader实际恢复；
任何一组异常停止后续，不自动重试。运行与配置哈希见[启动记录](learning-eval-r2-launch.json)。

20:29复核：原始组已有32条prompt_response记录、SC2继续执行，尚无整组结果；
A4/B4目录仍未创建，不能把排队或分片预检描述成已恢复这两组权重。
