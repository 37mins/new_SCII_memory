# SpB HF/FSDP分层包裹候选

状态（2026-09-11核验）：`1-step completed；96-step failed after63 steps`。
八卡Level6父实验于03:43:36北京时间因SC2子进程900秒未返回退出1，
完成504局、390胜；并非96步完成或独立学习验收通过。旧产物保留，未重启。
已完成[gamma-only A/B](gamma-ab.md)：两组各新增8步、各39/64胜，未证明胜率改善。
[Level6后续核验](l6-training-signal-followup.md)复核63步中的41个近零优势批次，
并记录失败状态覆盖不足；未确认新的PPO实现bug，不将gamma1合入。
一步于2026-09-10
11:37:27北京时间正常退出；该批反向传播不再OOM，不代表已验证长程学习。
用户2026-09-10批准44 GPU4–7、8条rollout、1个global step、最多1小时；不自动重试或续96步。
原基线为`6e40a5cc47dd5004a3dbb9a433c520cc6f497b54`，失败run与产物见
[原短训记录](../20260909-bo-segment-protocol-v1/quota-rl4.md)。
独立分支`codex/fix-spb-hf-fsdp-wrap`从`1ae98a8`建立，后者相对运行SHA只有记录变更。
原deployment和模型资产只读，不续用没有checkpoint的失败run。

## 证据与范围

原run在`dp_actor.py:476 loss.backward()`申请19.25GiB失败，当时仅余17.30GiB。
冻结`fsdp_workers.py:307–311`先按模型`_no_split_modules`获得包裹策略，再因
HF rollout强制改为None。原日志450行为reference的分层策略，551行为Actor的None，
不是FSDP完全没有分片，而是Actor没有按Transformer层拆分计算单元。

既有SpB文件头的FakeTensorMode只读核验：815个保存条目、5,555,171,328个BF16
元素。模型源码与config确认embedding/lm_head绑定，扣除重复head后共有
5,166,215,168个参数，完整FP32块为19.245651GiB。36层按原
`SSESWAMoBABlock`分组后，根部绑定embedding/norm为1.448984GiB，单block最大
144,921,088个参数。报错大小与完整参数块吻合，是强证据，但没有CUDA分配栈，
仍不能声称已证明具体分配来自梯度拼接、累加或哪个内部算子。

PPT《规划层多轮RL优化_副本》第7–8页讲的是整轨迹拼成长序列。当前collector已
按轮形成样本，history0/off、每卡micro-batch1，因此不重复修改轨迹拆分。
四卡时每卡梯度累积2次，八卡时1次；增加卡数不会让完整展开块本身减半。

唯一生产修改是给既有HF兼容分支增加默认关闭开关：

```text
+actor_rollout_ref.rollout.fsdp_layer_wrap=True
```

开启时保留`get_fsdp_wrap_policy`从模型原生`_no_split_modules`构造的策略。
不增加新的分组算法，不改FSDP版本、参数/温度精度、loss、学习率、reward、
prompt、BO、微操、采样预算、cache或dropout。未设置/False保持旧行为。
当前适用验证对象仅为SpB SSESWAMoBA + Torch2.7.1 + Transformers4.55.2。

HF生成仍使用原`recurse=False`上下文，未改成递归全模型展开。现有HF4.55.2
会为FSDP管理的模型自动启用`synced_gpus`，FLA的FLAGenerationMixin不重写
generate；仍需实际验证不同rank的EOS长度不会使collective卡住。

## 公开依据

- [PyTorch2.7 FSDP](https://docs.pytorch.org/docs/2.7/fsdp.html)：auto_wrap_policy
  控制子模块包裹，None只包根；使用原生Transformer分组策略。
- [VERL v0.4.0的HF兼容分支](https://github.com/verl-project/verl/blob/v0.4.0/verl/workers/fsdp_workers.py)：
  同样因Gemma/HF挂起而关闭包裹。当前runtime自报0.3.1.dev并有项目补丁，
  不冒称整个runtime等于该tag；这里只对照相同兼容分支。
- [PyTorch #100069](https://github.com/pytorch/pytorch/issues/100069)：保留
  HF生成时只summon根参数的既有处理，而非取消cache。
- [Transformers4.55.2生成源码](https://github.com/huggingface/transformers/blob/v4.55.2/src/transformers/generation/utils.py)：
  FSDP模型的synced_gpus默认值用于避免rank提前结束造成不同步。

## 验证与晋级

CPU入口：`PYTHONPATH=.:tests python -m unittest test_hf_fsdp_layer_wrap -v`。
检查默认不变、显式启用、非HF/reference不变；小型Qwen替身以两个CPU/Gloo
rank验证实际嵌套FSDP、两次梯度累积、裁剪与Adam参数一致、cache保留和不等EOS。
小模型CPU通过不能证明SpB内核、BF16路径或真实GPU峰值通过。

已批准44 GPU4–7、原始SpB、Level6、8条rollout、1个global step，最多1小时，
save_freq=1，无自动重试/96步。沿用原失败run的四卡配置，仅开启包裹开关并缩短
验收步数。这是更新/显存验收，不是学习效果A/B；生成重新采样，不冒称原batch
逐token重放。原失败run没有保存完整PPO张量、原advantages和参数梯度，不能
将人为生成的优势值冒充原更新。只读尺寸匹配也不替代CUDA分配栈。

### GPU验收入口

`SOURCE_SHA=<冻结SHA> SC2_FSDP_RL_AUTHORIZED=1 bash experiments/records/20260910-spb-fsdp-layer-wrap-v1/run_train1_node44.sh`

RUN_ID为`20260910-spb-fsdp-layer-wrap-v1-spb2-l6-g2-opening-quota-rl1-4gpu-s0-r1`，
输出到共享新项目同名experiment的`runs/`，受控tmux运行。入口拒绝dirty源码和
4–7已有GPU任务，保留失败结果。依赖、模型/地图/BO哈希由原资产校验与manifest保存。
`trainer.total_training_steps=96`仅保留原学习率调度预算；`total_epochs=1`限定本次一步，
没有后续96步命令。原始SpB模型、game env seed0、Terran/Air、Flat128、g2资产、
Planner开局接管均不变；实际游戏seed另从training_seeds记录，不冒称8个独立种子。

验收依次确认：Actor日志保留原生block包裹、缓存生成不挂起、8局真实rollout、
反向传播不再OOM、完成1个global step并保存完整checkpoint。单次通过仅支持该配置
显存/更新可运行，不证明长程稳定或学习效果。CPU入口对照测试逐项比较完整effective
config，只允许分层开关及验收步数变化；输出路径不计作方法变量。
启动入口CPU回归：`test_fsdp_rl_launch test_hf_fsdp_layer_wrap test_quota_rl_launch
test_system_release`共33项，32通过、1因本地无Torch跳过，4.718秒；实际44的
FSDP四项此前已全部通过。完整effective config差异检查通过，不仅检查少数手选字段。
本轮控制日志/起止时间/退出码写到44的`/root/new-scii-spb-fsdp-20260910/rl1-r1`，
模型与游戏初始化通过前不得称为已开始真实RL。

## 已执行结果

候选SHA `08fa2e3c1091eb15f31b893018c8af3cd3959202`，唯一生产改动为
fsdp_workers.py的条件与注释；HF生成、loss、模型源码和其它冻结文件未变。
CPU部署`sc2-a100-44:/root/new-scii-spb-fsdp-20260910/code-08fa2e3`，依赖只读复用
`/root/new-scii-spb2-level5-control/python`，未部署或下载新环境。完整Git bundle
本地和该节点均为`f751a5d14565c221febbb4193ebd89e8d46d01f0fa746e48c4e8b037860514e2`。
未推送GitHub，保留先前推送未授权边界；服务器bundle提供远端私有备份。

本地`test_hf_fsdp_layer_wrap test_quota_rl_launch test_system_release`共32项，
31通过、1因本地无Torch跳过，4.853秒。44实际Python下专用4项全部通过，
6.605秒、退出0，日志`../cpu-r1.log`与`../cpu-r1.done`保留。运行显式
`CUDA_VISIBLE_DEVICES=`；两个子进程均断言CUDA未初始化。真实CPU/Gloo结果：

```text
rank0: nested_fsdp=3, backward/Adam/cache/EOS passed, response_tokens=1
rank1: nested_fsdp=3, backward/Adam/cache/EOS passed, response_tokens=3
```

这里是随机小Qwen替身验证FSDP与HF API，**不是原SpB GPU/内核验收**。
没有修改生成cache、BF16 autocast或更新精度来让真实实验过关，也没有声称
19.25→1.45GiB是实测总峰值；它只是静态最大参数组大小，激活/临时缓冲仍另占内存。

第一次直接meta构造SpB的CPU检查因FLA导入时Triton要求GPU driver而失败，未分配
GPU、未读取权重数据；改为FakeTensorMode只读权重元数据后完成尺寸核验。
这是诊断方法限制，不是项目原有bug的修复成果。

## r1实际启动核验

运行冻结SHA为`7e5b1f9c58558ea2585c3b6fa58692851a0a1a9c`，44独立源码目录
`/root/new-scii-spb-fsdp-20260910/code-7e5b1f9`。相对08fa2e3，只有入口、CPU入口
测试和文档变化；部署保持clean。受控会话`sc2_spb_fsdp_rl1_r1`于北京时间
10:58:32开始，最迟11:58:32结束；不在原失败run重试、不自动续训。

本地及44完整Git bundle SHA256均为
`bf96ecab54755ebf7a3a7815e828ecca996d7c507f2a89cbb74d5c82af74edfe`，
44位置`/root/new-scii-spb-fsdp-20260910/source-7e5b1f9.bundle`。原08fa2e3 CPU
部署、6e40a5c失败部署与全部产物未动。GitHub未推送，不绕过已有授权边界。
44真实Python下`test_fsdp_rl_launch`1项通过、0.319秒；随后才启动GPU。

实际manifest确认formal=true、dirty=false、training、GPU4–7、world_size4、
target_steps1、resume禁用；原始SpB模型manifest仍为
`40d6f1417568d5fda1107cdafd66c7441a679b03fb148851c8f226ecc9cf884c`。
源码/模型/SC2资产核验通过；Torch2.7.1+cu128、Transformers4.55.2、Ray2.48.0、
Gymnasium1.2.3。实际effective config确认fsdp_layer_wrap=True、8rollout、save_freq1、
FP32参数/归一化、swa_dropout0、192回答预算，未加诊断hook或修改cache。

10:59:47日志已见11/11权重分片加载与5.17B模型；reference及Actor两次构造均打印
`transformer_layer_cls={SSESWAMoBABlock}`，不再是Actor wrap_policy=None。
11:00:11 SC2实际开始Flat128/Harder/Air，固定BO保留0条、Planner开局接管。
实际第一批seed为8个2732，与失败run相同；不是8个独立种子，也不承诺模型逐token
相同采样。11:01:12游戏已推进到约01:06、四张卡worker存活，无已报告OOM。
**此时尚未完成反向传播、global step1或checkpoint，不能宣布GPU验收通过。**

11:28–11:30只读进展：8局全部Victory，44条真实Planner决策，环境整局reward
`[21,21,20.5,21,21,21,20.5,20]`，均值20.75。这是更新前rollout，不是学习提升。
已进入global_step_1保存，原反向传播OOM未再出现；四个worker均在
`actor_rollout_save_checkpoint`，rank0/2进程处于D态的`wait_on_commit`/
`folio_wait_bit_common`。共享根stat成功、具体optimizer文件stat超时。
尚无退出码/最终指标，不能宣布checkpoint完整、长程稳定或概率验收通过。

## 已批准的96步后继任务（等待前序完整验收）

用户批准：43八卡、原始SpB、Level6、96个global step、最多48小时，每4步完整
保存，不删旧产物、不自动重试。入口`run_train96_node43.sh`，正式执行须设置
准确`SOURCE_SHA`及`SC2_FSDP_LONG_RL_AUTHORIZED=1`。RUN_ID为
`20260910-spb-fsdp-layer-wrap-v1-spb2-l6-g2-opening-quota-rl96-8gpu-s0-r1`。
与四卡验收相比只扩量、换八卡、保存频率1→4和节点依赖路径；不是单变量修复A/B。
每卡micro1、global mini8不变，因此累积2→1。生产模块不再改。

从原始模型新开完整96步，不加载step1 checkpoint；保留短测权重。这样避开现有
断点恢复会重置SC2独立training_rng/reset_index且归档只验收1..target的已知覆盖
缺口，不伪装为不中断续训，也不在本轮顺手修复这些边界。

前序门槛由`check_rl1_acceptance.py`只读核对：准确7e5b1f9来源、formal/clean、
完成1步、有限指标、退出0、result/.done一致、四卡model/optim/extra与data.pt
文件大小符合前序记录。无新KL阈值或胜率门槛。未通过则不得启动后继。
启动还须八卡无compute任务、依赖版本匹配、原资产哈希通过、共享盘至少2TiB空闲
并通过有超时的1MiB write/fsync检查；保存探针作为记录，不删除任何旧文件。
24份checkpoint估计约1.4TiB，是由参数与Adam状态大小推算、不是已测占用。

43在11:31–11:32八卡全部空闲；旧依赖目录
`/root/new-scii-spb-level5-96-20260908/python`及include可复用，CPU实际核验仍为
Torch2.7.1+cu128/Transformers4.55.2/Ray2.48.0/Gymnasium1.2.3，CUDA未初始化。
后继采用独立`/root/new-scii-spb-fsdp-20260910/code-<SHA>`、控制目录
`rl96-node43-r1`及Ray`/dev/shm/sc2_spb_fsdp_rl96_0910_r1`，不改43旧deployment。
日志/权重仍在本experiment共享runs。资源是快照，启动前重新检查，不预占GPU。

前4步重点查看连续更新、回答和游戏趋势，但不中途插入额外诊断实验；0胜或单个
KL阈值不自动判错。明确运行错误、非有限数、归因错误或明显回答退化需报告/处理，
不以有限梯度、8/8首批胜率代替学习有效性。48小时上限含运行和退出，不自动延长。

## 1-step最终结果及后继准备

前序控制退出0，结束于11:37:27；result状态completed、completed_steps1、
metrics_finite=true、checkpoint_complete=true，result/.done一致。保存约982.439秒，
更新约174.219秒；整段墙钟38分55秒。存储等待是本次耗时主项，不是新增GPU OOM。
13份核心checkpoint文件均存在，Actor文件合计63,552,433,724字节（约59.19GiB）；
四卡model SHA已保存在result，另有data.pt。未加载全部权重重算，无覆盖或删除。

| 指标 | 1-step结果 |
| --- | --- |
| 自然游戏/胜利 | 8/8，全部Victory，无截断/空策略批次 |
| 整局环境reward均值 | 20.75 |
| actor/grad_norm（裁剪前汇总） | 10.393 |
| actor/kl_loss（reference惩罚） | 0.003 |
| actor/ppo_kl（日志显示精度） | 0.000，不代表逐token完全一致 |
| actor/pg_clipfrac | 0.014 |

这些KL均为现有训练统计，不是完整词表精确KL或全批更新后重放结果；未增加诊断。
8/8是第一次更新前的rollout，不能证明RL提升。结论限于实际SpB分层FSDP缓存生成、
该批完整更新和保存通过；下一步在批准的96步内检查连续性。

后继冻结SHA`6a1657f7e045dcf998495a3d0adfb331572bf60d`，生产模块与7e5b1f9
完全一致；只有新入口/门槛/文档和测试。43独立部署
`/root/new-scii-spb-fsdp-20260910/code-6a1657f`，完整Git bundle
`/root/new-scii-spb-fsdp-rl96-6a1657f.bundle` SHA256为
`9dcefbe6a49a0bca7e6ae224d099f935332c5025b9fca5f45e1779a5d13cf1d1`。
本地35项回归34通过、1因无Torch跳过，5.481秒；43真实Python下入口/门槛3项
全部通过、0.536秒。CPU门槛单元测试使用模拟文件，不能代替真实前序检查；实际
启动前另读取7e5b1f9的已完成result并核验13份文件元数据。

自动跟进`sc2-96`已改为本次批准条件并恢复ACTIVE；每5分钟核对前序/后继，
未变化保持安静，只在完成、失败、启动或需要决策时报告。它是应用内条件跟进，
不是预占GPU的集群调度器；实际长训须独立tmux并确认模型和SC2已开始。

## 96-step实际启动

北京时间2026-09-10 11:43:16在43启动`sc2_spb_fsdp_rl96_r1`，48小时硬上限至
9月12日11:43:16。前序真实门槛通过，启动前八卡无compute进程；共享盘1MiB
write/fsync通过，可用6,963,297,517,568字节。新run及控制/Ray目录此前均不存在。
没有占用44、停止他人任务或修改前序运行；本次是已批准的后继，不是自动重试。

实际manifest始于11:43:48，formal=true、dirty=false、source_commit=6a1657f、
world_size8、GPU0–7、target_steps96、mode=training，模型manifest保持不变。
effective config确认epochs96、save_freq4、resume_mode=disable、rollout8、
fsdp_layer_wrap=True、response192、swa_dropout0。11:45:06前后完成11/11权重分片
加载；Actor和reference均打印原生SSESWAMoBABlock包裹。后续日志确认8局SC2
实际创建且Planner已生成开局计划，不只是CPU准备或排队。
首批实际游戏seed仍为8个2732，不称八个不同seed；此时尚未完成long-run step1。

后继从原始模型开始，新run中的step1不冒充短测step2；旧短测checkpoint完整保留。
前序8/8和有限指标仅支持进入连续运行，不能预先宣布96步学习成功。

## Level7 四卡短训（completed）

2026-09-10 用户另行批准44 GPU4–7、原始SpB、Level7（VeryHard）、4个global
step，每步8局、最多4小时，step4保存完整checkpoint（约60GiB）。不自动重试、
不扩96步、不加载Level6训练中权重；43的Level6八卡任务不动。本次复用原record
和branch，只增加RUN_ID与薄入口，不更改生产模块或默认训练配置。

入口为 `run_train4_level7_node44.sh`，设置 `SOURCE_SHA=<冻结SHA>` 和
`SC2_FSDP_LEVEL7_RL_AUTHORIZED=1` 后运行。RUN_ID：
`20260910-spb-fsdp-layer-wrap-v1-spb2-l7-g2-opening-quota-rl4-4gpu-s0-r1`。
输出仍在本experiment的共享runs；44独立 `code-<SHA>`、`l7-rl4-r1` 控制目录、
`sc2_spb_fsdp_l7_rl4_r1` tmux和 `/dev/shm/sc2_spb_fsdp_l7_rl4_0910_r1` Ray目录。
复用已验收的 `/root/new-scii-spb2-level5-control/python` 和系统Python3.12头文件，
不复制模型或部署新环境。旧deployment和已有run保持只读。

目的：验证同一系统在更高难度能否连续完成采样、PPO更新和保存，并观察游戏表现。
基准生产源码为 `6a1657f7e045dcf998495a3d0adfb331572bf60d`；模型manifest仍为
`40d6f1417568d5fda1107cdafd66c7441a679b03fb148851c8f226ecc9cf884c`，无项目SFT/RL。
Flat128、Terran/Air、g2资产、Planner开局接管、原BO段和quota说明、192回答预算、
history0/off、FP32参数/归一化、BF16 autocast、cache、swa_dropout0、lr1e-6和
GiGPO/PPO loss均保持。四卡使每卡梯度累积1→2，全局mini8/micro1不变；不是
相对43八卡的纯难度单变量对照。`total_training_steps=96`保留调度口径，但
`total_epochs=4`明确限制本轮四步，无第5步命令或长训晋级。

记录每步自然胜负、整局reward、真实决策有效率、回答截断、KL、clip、梯度、
游戏种子与step4完整保存。环境seed0驱动既有种子序列；每批八局共用游戏seed，
不能当作八个独立场景。四步训练胜率不作为独立学习效果或架构证据，不设新的
KL/胜率停止阈值；明确运行错误、非有限数、空策略批次和归因错误按现有边界退出。
入口拒绝dirty来源/占用GPU，真实1MiB写入探针与2TiB共享存储余量检查保护既有
长训余量；探针保留，不清理任何旧产物。共享IO竞争仍须观察，不保证完全无干扰。

本地实际执行 `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:tests
/private/tmp/sc2-air-results.HBI61m/decoder/bin/python -m unittest
test_fsdp_rl_launch test_hf_fsdp_layer_wrap test_quota_rl_launch test_system_release -q`：
36项、35通过、1项因本地无Torch跳过，2.516秒。新增CPU对照比较完整effective
config：相对四卡1-step仅Harder→VeryHard（实际与expected）、epochs1→4、
save_freq1→4；原有测试同时约束四卡/八卡入口差异。没有GPU或SC2启动。
`bash -n`与`git diff --check`通过；正式运行另核验44依赖和clean部署。

### Level7 实际启动核验

冻结SHA `cfdd3b468cf9bbe3a1f85ce8dc9a5a9ea530ffbd`，44独立部署
`/root/new-scii-spb-fsdp-20260910/code-cfdd3b4`，formal=true、dirty=false。
相对6a1657f只增加Level7入口、测试和文档；两次实际manifest的生产源码哈希映射
及模型manifest相同。归一化run/source/Ray路径后，实际effective config仅有
Harder→VeryHard的两项难度、epochs96→4、GPU8→4四项变化，save_freq同为4。

44真实Python下`CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1
PYTHONPATH=<deployment>:<deployment>/tests:<python依赖目录> python3 -m unittest
test_fsdp_rl_launch -q`四项通过，0.950秒。Torch2.7.1+cu128、Transformers4.55.2、
Ray2.48.0、Gymnasium1.2.3均一致，CPU核验未初始化CUDA。部署前完整Git bundle
保存在44的`/root/new-scii-spb-fsdp-20260910/source-cfdd3b4.bundle`，本地和远端
SHA256同为`514965da0c9358d1ac75e05a98b612767e8f1430c922e1eac446043ed6613222`。
未推送GitHub，不绕过此前推送授权边界。

2026-09-10北京时间15:06:32启动受控tmux，四小时硬上限到19:06:32；
控制日志为44的`/root/new-scii-spb-fsdp-20260910/l7-rl4-r1/console.log`。
启动前GPU4–7无compute任务，共享盘约6322GiB可用且write/fsync探针通过。
GPU0属于其他任务，未终止；43长训和既有deployment均未修改。

11/11权重分片加载完成，Actor/reference均打印原生SSESWAMoBABlock包裹；
15:08:24 SC2创建VeryHard/Terran/Air对局，15:08:29开始Planner开局调用。
15:12–15:14只读核验：四张卡worker分别为1685172/1685479/1685480/1685481，
真实决策已覆盖8个env；第一批实际游戏seed为8个2732。第一条模型原始回答为：

```text
<answer>
Here are the Build Orders with attack instructions:
- supplydepot x 2
- barracks x 2
- marine x 2
</answer>
```

该条semantic_action_valid=true、policy_action_applied=true。已有16条真实决策；
env5自然Victory，decision_count=2、truncated=false，其余尚未收齐。不将这一局
当作首批胜率或学习提升。此时无Traceback/OOM/RuntimeError，但尚无global step1
更新指标，也没有step4 checkpoint；只确认真实RL采样已启动，不宣布四步验收通过。
日志中的`default_player_Qwen3-30B`为冻结的旧日志标签，实际权重是manifest所记SpB，
本轮未改名或切换模型。

## Level7 step4 → 96 断点续训（aborted，用户要求释放四卡）

9月10日20:40:34北京时间主动停止，完成到global step31，退出130，最后完整
checkpoint为step24；产物保留。[停止记录与温度A/B](temperature-ab.md)说明四卡
后续用途，以下内容保留原批准与运行历史，不代表仍在续训。Level6八卡不变。

用户批准44 GPU4–7、最多48小时、继续到global step96，不自动重试；后续完整
checkpoint改为**8、16、…、96**，保留原step4，新增约12份/0.7TiB。
这些是磁盘存储量，不是GPU显存。43八卡Level6的保存频率和运行均不改。

原cfdd3b4的FSDP checkpoint保存了分rank模型、Adam、调度器和worker RNG，
但`SC2MultiProcessEnv.training_rng`、`reset_index`及manager账本计数未保存。
新候选隔离在`codex/fix-sc2-checkpoint-resume`，只增加显式启用的checkpoint状态
边界和续训归档；不改生成、loss、reward、Planner、BO、战斗或FSDP计算路径。

`+trainer.sc2_checkpoint_resume=True`在`data.pt`后、latest tracker前额外保存
`sc2_state.pt`：训练/验证环境RNG和计数、manager交互计数、空批次累计数、
训练主进程Python/NumPy/Torch CPU RNG。仅支持游戏已结束、memory=off的边界；
不是游戏中途恢复。未开启时保存/恢复路径保持原样。既有FSDP loader仍负责
权重、Adam、学习率和各rank RNG；不做权重转换，不换卡数，reference保留原始模型。

旧step4没有sc2_state，首次使用`+trainer.sc2_import_legacy_final=True`，严格核对
原run退出0、最终checkpoint、4批完整种子账本和32条自然结局后重建环境状态。
不支持猜测旧中途checkpoint或含跳过批次的旧前缀。原始seed0序列
2732、43567、42613、52416匹配；下一批应为reset_index4、8个45891。
manager的episode编号同步到3，下一次reset后为4；交互计数从原回答账本恢复。
旧run没有保存训练主进程RNG，不能追溯补造，因而**不声称全流程逐位等同不中断运行**。
新状态从后续保存点起包含该部分；当前承诺是模型/优化器状态续用和游戏种子序列连续。

公开依据是[NumPy1.26 RandomState.get_state](https://numpy.org/doc/1.26/reference/random/generated/numpy.random.RandomState.get_state.html)
及其`set_state`，配合冻结VERL的`_save_checkpoint/_load_checkpoint`与
`FSDPCheckpointManager`既有恢复边界；没有引入新采样算法。

入口`run_resume96_level7_node44.sh`使用`SC2_LEVEL7_RESUME_AUTHORIZED=1`与准确
`SOURCE_SHA`，新RUN_ID为
`20260910-spb-fsdp-layer-wrap-v1-spb2-l7-g2-opening-quota-resume4to96-4gpu-s0-r1`。
新输出位于同experiment的共享runs，独立44控制目录`l7-resume96-r1`、Ray目录
`/dev/shm/sc2_spb_l7_resume96_0910_r1`；不往原run或deployment写入恢复补丁。
原data.pt必须存在；显式恢复`global_step_4`，模型、Adam、数据加载返回并重建环境后，
新run的`checkpoints/resume_state.json`记录下一批seed。日志必须显示step5而不是step1。
48小时包括加载、训练、保存和退出；前序未完整或GPU被占则不启动，不循环重试。

总学习率调度预算仍是96；global step终点仍是96，实际新增92步，不是再训96步。
归档保存start_step=4，只验收5–96的指标和92×8局，最终还要求sc2_state.pt存在；
原1–4的指标和结局留在原run关联，不复制成新训练样本。

本地CPU回归：`PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:tests
/private/tmp/sc2-air-results.HBI61m/decoder/bin/python -m unittest
test_sc2_checkpoint_resume test_fsdp_rl_launch test_sc2_empty_policy_retry test_system_release -q`：
47项、46通过、1因本地无Torch跳过，7.027秒。实际reset方法验证后92批种子与连续
运行一致；损坏账本/计数/配置拒绝、归档5–8和完整入口配置差异通过。入口对照仅允许
resume开关/路径、epochs4→96和save4→8。带真实CPU Adam/Torch的trainer保存加载
测试尚待44固定依赖执行；这些CPU结果均不代表真实SpB恢复和后续学习验收已通过。

### 原Level7四步最终结果与续训部署

原run `cfdd3b4`已completed，控制退出0，北京时间2026-09-10 16:23:25结束。
四步32局均自然结束，22胜，无跳过批次；checkpoint完整，核心文件约59.19GiB。
step4保存991.603秒，不能把共享盘等待说成GPU OOM或模型失败。

| global step | 游戏seed（该批8局共用） | 胜局 | 训练reward均值 | reference KL | PPO K1 | 裁剪前梯度范数 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 2732 | 8/8 | 20.375 | 0.002 | -0.001 | 7.382 |
| 2 | 43567 | 7/8 | 11.819 | 0.008 | -0.003 | 309.312 |
| 3 | 42613 | 0/8 | -49.456 | 0.007 | -0.003 | 22.426 |
| 4 | 52416 | 7/8 | 11.750 | 0.023 | 0.002 | 219.753 |

这些是训练采样，不是独立评测；单个seed失败或首批高胜率不证明学习退化/提升。
前序门槛核验了result/.done一致、准确SHA、4步指标、13份checkpoint文件尺寸，
没有额外加载大权重或修改原run。

恢复候选冻结SHA `90b3fdc2fb7fadc1230861306b20c27540d9b736`，44独立部署
`/root/new-scii-spb-fsdp-20260910/code-90b3fdc`。完整私有Git bundle为同父目录下
`source-90b3fdc.bundle`，本地/远端SHA256均为
`93bc05bada56321e9a8dfa516fba14a29b7cd1926f27727b6bc3cb3293e90a48`；未推送GitHub。
44在CUDA隐藏时执行`test_sc2_checkpoint_resume test_fsdp_rl_launch`共10项全部
通过，2.217秒，日志`cpu-resume-90b3fdc.log`，工具退出0。实际trainer保存/恢复
方法使用CPU小模型与Adam回调验证权重、moment step4、scheduler、RNG、环境计数；
这不是SpB/FSDP GPU加载测试。真实原run账本导入也通过：下一reset4、episode3、
manager交互编号25、下一游戏seed45891，未启动额外测试游戏。

新续训于16:29:31提交受控会话`sc2_spb_l7_resume96_r1`，48小时硬上限到
9月12日16:29:31。启动前GPU4–7全空，共享盘约6049GiB可用，新增write/fsync
探针通过；GPU0外部任务未动。此段只记录提交及preflight，不把它当作实际恢复
已通过；以新run的`checkpoints/resume_state.json`、首条reset4/45891和step5日志
确认后才报告真实续训。源文件与原run均不原位修改。

16:35–16:37只读核验：四rank FSDP checkpoint加载返回，原调度预算仍96，
新`checkpoints/resume_state.json`记录global_step4、next_reset_index4、8个45891，
`model_optimizer_dataloader_load_returned=true`。实际`training_seeds.jsonl`第一行
同为reset4/45891，SC2已创建VeryHard对局并推进；不是重训step1或只加载原始模型。
加载期间四worker曾在`folio_wait_bit_common`，打开的是原step4的optim分片，
随后正常返回，不将该I/O等待认作运行错误。真实首批恢复支持状态链路验收，
但第5步完整更新、后续连续性和step8新增sc2_state.pt落盘仍待观察。

自动跟进工具只返回旧任务卡片，当前本地可读取的自动任务配置不含sc2-96，无法
安全保留旧字段更新，因此未修改或新增自动任务。训练本身由已启动的tmux与
48小时timeout继续控制；不声称已设置新的无人值守提醒或额外训练重试。

### 只读学习信号审计（9月10日晚）

[零advantage重算](advantage-zero-audit.md)与
[输出趋同／不同计划的奖励对照](sampling-reward-audit.md)记录已有批次证据。
已确认部分批次缺少组内回报差，以及同终局奖励时按Planner调用次数折扣的偏向；
未确认新的参数更新bug，未据此修改运行中的训练或新增GPU实验。

[Level6全胜／全败审计](l6-binary-outcomes.md)进一步对齐step1–28的组内采样、
真实终局，以及step20/27/28各一份留存回放。相同两段Marine计划在不同场景全胜
或全败；败局有真实产兵及连续波次战损，step28第二段后7分30秒才重新规划，
三个Reactor因0/50气等待超时。本次只记录证据，不修改采样、BO或执行反馈。

[温度0.4/1.0 A/B](temperature-ab.md)已完成，各8局、均4胜，胜负仍按两个诊断
seed分开。提高温度增加回答变化，但未改善这两个场景；合法小计划也出现
Reactor排在Refinery之前、缺气超时和连续战损。未修改43长训或将温度候选合入。

[Refinery先于Reactor的同种子顺序对照](refinery-order.md)已获批准，新增独立
纯评测入口；两轮历史回答干预与后续在线模型选择分别记录，不作为模型能力提升。

[gamma0.95/1.0短RL对照](gamma-ab.md)已获批准：同一Level6 step4模型/优化器，
两组各新增8步，八卡串行；新游戏种子配对，不冒称缺少SC2状态的历史无缝续跑。
此实验不修改43的长训，也不包含上述强制回答或顺序干预。
