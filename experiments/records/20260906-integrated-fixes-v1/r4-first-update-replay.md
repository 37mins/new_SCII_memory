# r4：恢复 step1 后只重放一次 optimizer update

## node47-r2：仅补CPATH的重试已完成（2026-09-07 18:39北京）

**三个重放验收点通过，但尚未发现实现bug，也没有解释最初漂移的根因。**
诊断SHA `ff4086c867a60a425ea3f192a76d7a2a64dc2719`，训练runtime仍为
`ca7885d886aaaafb39fc105b1d5d46cddf4bcc5a`。47八卡在18:37:37启动，
18:39:24退出0，耗时约107秒（上限1680秒），八卡完成日志与`.done`一致。
18:41只读检查GPU进程为空。没有启动游戏、第二次更新或后续训练。

| 验收项 | 实际结果 |
| --- | --- |
| 恢复准确 | 全184条、2087有效token更新前与保存old差值全部为0；不覆盖old/ref |
| 只更新一次 | 八卡均加载真实step1状态，Adam 22→23，仅一次optimizer update，未推进global step |
| 全batch复查 | 184条均检查；最大log-prob下降6.832047，绝对变化>1有10个token，>20为0 |
| Loss／裁剪 | 八卡记录的局部log-prob/entropy直接梯度与冻结loss的CPU计算一致，padding梯度0；裁剪切片与公式差0，全局norm 24.807060→0.999999953 |
| AdamW参数链 | 共524288个固定参数元素；与PyTorch2.7.1 CPU参考相比，参数差最大误差9.31e-10，m完全一致，v最大差2.22e-16；实际参数变化最大9.53e-7 |

CPU/CUDA运算差如上报告，未另设或放宽通过阈值。原始8份梯度/动量/参数块约
23 MB，保存在新RUN，非完整4B参数证据。没有归约前梯度或完整Jacobian，因此
不能把“Adam切片对账一致”扩大为“模型反向及FSDP所有环节均已排除”。
作为历史重放交叉核对，更新后offset1的下一mini八条概率与历史对应记录差0；
这只是附加核对，不能代替已完成的全184条检查。

全批old-current K1：回答均值0.007935、有效token均值0.016143；不是reference KL。
1994/2087个token差0，24个绝对变化>0.01，17个>0.1。14个ratio<0.8、2个>1.28，
这是**更新后全批越界计数**，不是本次backward时的actor/pg_clipfrac。
最差为rank2/offset3/token1 ` Launch`：log-prob -3.167999→-10.000046，
条件概率4.2088%→0.00454%，ratio=0.001079；该回答**不在本次更新的八条内**：

```text
- Launch OFFENSE with 8 marine
- marine
- supplydepot
- mule
- siegetank
- marauder
```

完整输入、前三个最差回答、八卡对账和artifact哈希见
[结果JSON](r4-first-update-node47-r2-results.json)。原回答不含answer标签，末尾含EOS；
以上只为便于阅读省略EOS，未改模型输出。不能用这个未参与更新样本的advantage
解释其概率方向；共享参数产生跨样本影响是可能路径，不是已定位的数值根因。
本次唯一环境修正CPATH解决的是诊断部署错误，不计为r4训练器bug修复。

后续不盲改代码、不加入early-stop、不扩展精度候选或自动补GPU测试。现有证据
支持恢复、一次计数、记录切片的裁剪与Adam计算一致；仍缺参数梯度生成／归约前后
及特定输入敏感性的因果证据。没有具体实现偏差，不提出训练代码修复。
step8配对评测结论仍见[已完成评测](real-rl-r4.md#配对评测完成结果)，不能用本次
重放完成代替学习验收，也未因此续到96步。

本次实际执行：两launcher的`bash -n`、`git diff --check`、诊断Python相对
6e80929的`git diff --exit-code`均通过；正式重放使用冻结入口
`diagnostic-ff4086c/experiments/records/20260906-integrated-fixes-v1/run_r4_first_update_node47.sh`。
已有`analyze_r4_first_update.py`在47的Torch2.7.1、CUDA_VISIBLE_DEVICES为空下
退出0；附加CPU核对只读取保存切片、诊断batch及冻结函数，无模型forward。

### 本次批准与启动配置

用户明确批准“只补CPATH，新RUN重试，最多28分钟，其他配置不变”。r1失败与
全部产物不变；新RUN为`20260906-integrated-fixes-v1-r4-step2-first-update-8gpu-node47-r2`。
唯一环境修正是恢复47既有头文件CPATH；另按授权登记新RUN/临时路径及1680秒
上限（1665秒运行+15秒退出），这些不是训练配置变化。旧46默认仍为r1/1800秒。
诊断Python、三项验收及历史runtime ca7885d均不改；没有重算或覆盖old/ref。
复用47已有依赖、头文件、原始step1 checkpoint和内存盘runtime，另建新SHA诊断目录。
启动前必须检查八卡全空；失败不继续自动重试。本次环境修正不计作训练器bug修复。

## 47迁移（用户09-07明确批准；环境失败，0次更新）

18:01北京47八卡无GPU进程，共享盘可读；系统盘满，沿用现有
`/dev/shm/spb-unitid-20260906/python`（Torch2.7.1、Transformers4.55.2、Ray2.48.0）。
只将节点从46改为47，仍是唯一一次八卡30分钟重放，不额外保留一个46待运行任务。
入口`run_r4_first_update_node47.sh`只设置节点和依赖路径，复用原启动器；
该启动器默认仍为46。诊断Python与1a468af逐字相同，runtime仍是ca7885d。
部署/临时文件位于新`/dev/shm/r4-first-update-node47-20260907`，用独立screen，
日志/参数链证据仍写共享runs下
`20260906-integrated-fixes-v1-r4-step2-first-update-8gpu-node47-r1`。
无需复制或下载权重/依赖；旧部署、checkpoint及未运行的46记录不变。
共享入口继续强制八卡空闲、唯一RUN、完整恢复、一次更新和1785+15秒上限。
恢复概率不符时仍在更新前共同停止，不因换节点而放宽。实际启动状态以下续记。

18:04:25北京，screen `sc2_r4_first47`实际启动，诊断SHA
`6e809293d6c458a2f58c5cd4302b67828344894d`，runtime为ca7885d；启动前47的
4/4 CPU回归、两入口shell语法及诊断Python相对1a468af无差异核验通过。
18:06:05退出1，`.done=failed`，八卡已释放。[失败证据](r4-first-update-node47-r1-failure.json)
包含真实manifest身份、八卡加载事件、零概率行、零参数链文件及原始artifact哈希。
不能将启动器已建RUN或checkpoint加载返回称为三项验收通过。

八卡都已从r4 step1加载model/optim/extra并确认Adam计数22、FP32参数；但首个
full-batch forward在Triton辅助模块编译时报：

```text
fatal error: Python.h: No such file or directory
```

全部八份概率trace均0行，没有执行optimizer update，也没有梯度/动量/参数差
产物；因此恢复概率是否与old一致尚未验证，更没有新的KL根因证据。
原因是本次迁移入口遗漏了旧47入口已有的
`CPATH=/dev/shm/spb-unitid-20260906/include/python3.12:/dev/shm/spb-unitid-20260906/include`。
依据为`20260906-qwen-kl-metric-v1/run_node47.sh:16`、旧部署记录及现场头文件。
这属于诊断部署返工，不是r4训练器原有bug，不能计入项目修复成果。

只做CPU `gcc -x c -fsyntax-only -`检查，上述CPATH下`#include <Python.h>`退出0；
没有安装、下载或修改依赖，也没有重放第二次。原driver位于
`/dev/shm/r4-first-update-node47-20260907/replay-r1.driver.log`，完整console和失败
状态在共享node47-r1 RUN中保留。按整个启动至退出约100秒保守计占用2分钟，
原总预算剩余最多28分钟；若用户批准重试，仅补已有CPATH，用新SHA和RUN，
仍只一次更新、全184条检查，不增加GPU总预算。当前暂停，未自行重试。

最新执行状态（2026-09-07 17:11北京）：46仍八卡被占、无本次tmux/RUN，GPU预算未使用。
恢复前置检查的诊断SHA为`1a468afed041c029b60f9c28164301ab0e017642`，已推送并
部署到46独立`/root/new-scii-r4-first-update-20260907/diagnostic-1a468af`，HEAD正确且clean；
`replay_r4_first_update.py` SHA256为`09235ea2ab7ce9a20b4c78ae4d92331188f6491c3a267e1043bc021e3041aa52`。
原runtime-ca7885d、diagnostic-2a70dc6、CPU暂存及所有既有结果均保留。
尝试将既有自动跟进`r4`更新为本次三项验收时，产品返回该任务已不存在、可能被
用户手动删除；没有自动重建。因此下文自动跟进描述是历史状态，当前无已确认的
空卡自动启动任务。需要恢复后台跟进时先确认用户意图，不能声称已经排队运行。

状态：准备完成，**立即并行授权已明确；46被外部任务占满，未启动GPU**。用户2026-09-07明确批准八卡30分钟、一次更新、
不启动游戏；随后允许寻找其他空节点并行，不必占用44或等待其评测结束。
13:16北京，46八卡无GPU进程；现有Python依赖核实为Torch2.7.1+cu128、
Transformers4.55.2、Ray2.48.0。启动前必须再次确认归属和共享盘写入状态。
这不是新的稳定性修复，也不是r32重跑；44的r4真实RL继续保持不变。

13:33北京再次核实46八卡显存0、无compute进程。诊断源码
`2a70dc6a6d816cc57480493d2af3ffbfb9eaccdf`已推送私有origin并部署为
`/root/new-scii-r4-first-update-20260907/diagnostic-2a70dc6`，工作树clean。
创建tmux/启动GPU的请求被平台拒绝：仍认定“44 step8及配对评测结束”为前提，
未接受用户后来“其他空节点可以并行做”的授权解释。该命令未执行，不能写成
已跑上或实验失败；等待用户明确确认立即并行，不绕过权限检查。

后续用户已明确批准：现在46八卡并行一次更新重放、最多30分钟，不等待step8及
配对评测结束。14:03北京复查46仍八卡空闲、诊断部署clean；44完成标志为5，
有正在写入的`global_step_6`，此后小文件检查也超时。启动暂缓仅为避免共享盘
竞争，不再以step8为前提，不修改或停止44。等待期间准备
`analyze_r4_first_update.py`，只做CPU切片Adam参考计算和全批最差token定位。

14:11北京，完成标志更新为6且pending为空；14:12创建诊断tmux的SSH返回0。
但启动器在GPU空闲检查处退出：46的vLLM进程于14:07启动，另一个训练于14:08
启动，共占满八卡。14:14确认新RUN目录不存在、driver日志0字节，故未进行
CUDA初始化或模型加载，GPU预算仍未使用。不能把tmux创建成功说成实验跑上。
42/43/47/48复查也无完整空闲八卡。没有终止任何外部任务或修改44。

已创建本线程条件跟进`r4`（每5分钟）：仅在46八卡全空、共享盘可读且没有
checkpoint保存竞争时启动本次唯一重放；有进程或结果不重复启动，完成/失败
后停止推进并暂停跟进。安静等待资源不代表GPU实验运行中。
CPU对账工具及新增断言在46运行3/3通过，提交`7210b20`已推送；正式GPU入口
仍是原`2a70dc6`，不因离线分析脚本提交而改变冻结运行时。

2026-09-07 16:39北京只读复核：46八卡均被外部`yht_axbench` Python占用，
`sc2_r4_first46`会话及预定RUN仍不存在；未进行重放，八卡30分钟预算未使用。
44的step8独立评测已于16:30自然完成；16:39该节点GPU1–4又被外部
`cjy_mob` Python占用，只有0、5、6、7空闲。没有停外部任务、改变卡数、
重跑训练/初评或启动第9步。评测结果另见[r4记录](real-rl-r4.md#配对评测完成结果)，
不能作为KL根因证据。本轮继续复用本入口，不另建诊断或扩展数值候选。

复核仍未发现可以直接提出修复的实现bug：原记录中第一次更新前的八条回答
最大actor-old差为0；第二次更新所取八条回答的更新前最大差为0.00579834。
后者只是第一次更新后**下一mini的八条**，不是原184条全批，不能据此排除
其余样本已经漂移，也不能拿r32的7.98冒充r4第一次更新结果。源码主链路与
Adam计数核对不替代缺失的参数梯度/动量/参数差。唯一下一项仍是本次单更新重放。

## 输入、更新和证据

沿用r4 runtime `ca7885d886aaaafb39fc105b1d5d46cddf4bcc5a`，在46另建
`/root/new-scii-r4-first-update-20260907/runtime-ca7885d`，逐文件核对冻结源码。
独立诊断入口有自己的提交，不用最新工作树替代历史runtime。
历史RUN为`20260906-integrated-fixes-v1-qwen-instruct2507-l6-8gpu-fp32-rl96-gate8-s0-r4-train8`，
位于本experiment共享`runs`目录。

从其`checkpoints/global_step_1/actor`加载八份model/optim/extra，保留原文件；
恢复Adam计数22、m/v、scheduler与RNG，不用原模型零动量假装复现step2。
读取`ppo_diagnostics/update-0002-rank-00..07.pt`：184条回答，含原4条copy，
保持token、mask、old/ref、advantage和rank内顺序。每卡仅取offset0，全球8条，
调用冻结actor的`update_policy`一次；mini1/micro1/epoch1即一次optimizer update，
不是一个global step。计数必须22→23，不推进global-step scheduler。
前后都用原train模式整批forward检查184条，不生成新回答、不启动SC2。

模型仍为Qwen3-4B-Instruct-2507；FP32参数/温度、BF16 autocast、eager、cache、
lr/loss/ref KL/裁剪等来自历史effective config。唯一配置路径调整是将诊断文件
写到新目录；无old/ref重算覆盖、early-stop、内部算子或游戏模块改动。

每卡记录实际loss→logprob梯度、FSDP归约后裁剪前梯度、全局裁剪后梯度、
更新前后m/v/step及参数差。参数张量为每个flat shard固定16段、每段4096元素，
保存offset与原参数映射；另外记录完整shard的梯度范数并跨卡合成。
**不是保存4B模型全部梯度/动量，也不等于核验了归约前梯度。**
使用PyTorch的[step前后观测接口](https://github.com/pytorch/pytorch/blob/v2.7.1/torch/optim/optimizer.py)
及tensor gradient hook，只复制detach值，不返回替代梯度或修改step参数。
离线按[PyTorch2.7.1 AdamW实现](https://github.com/pytorch/pytorch/blob/v2.7.1/torch/optim/adam.py)
核对所存切片的m/v与参数差；CPU/CUDA运算舍入差异须单独报告。

step1 RNG被恢复，但历史step2 rollout结束、更新前的RNG没有保存，不能声称
逐bit重放全部随机过程。当前Qwen dropout=0、固定batch无shuffle；额外forward
使用原诊断器的fork_rng。更新前184条对old的误差仍需如实报告，不能悄悄rebase。

## 执行与判读

09-07用户将验收收敛为恢复准确、Adam22→23、全184条前后比较。原诊断入口
虽保存全部pre概率，却只拦非有限值；本次仅补诊断前置检查，不修改ca7885d。
每卡23条有效token先与保存的old比较，写出误差分布，再跨八卡MIN共同决定
是否继续。沿用历史r4更新前actor-old最大差0的恢复预期：任何非零差异先停在
更新前保留证据，报告恢复前提不满足；不直接定性训练bug，不自动放宽、rebase、
换精度或重跑。它不是cached/full路径必须逐token相等的验收标准。
这属于诊断入口自身补齐，不计作项目原有bug修复成果。原2a70dc6诊断部署保留。

17:02北京复查46仍全卡被外部任务占用、共享盘可读、目标RUN与tmux均不存在。
新检查在46独立`cpu-restore-gate.VnqkiG`目录，用现有Torch2.7.1依赖执行
`CUDA_VISIBLE_DEVICES='' /usr/bin/python3 -B tests/test_r4_first_update_replay.py -v`，
4/4通过：含观察器保持Adam22→23不变、最后一条回答差异可检出及忽略padding。
本地launcher的`bash -n`及`git diff --check`通过。未启动CUDA或加载checkpoint。
下一次正式重放应使用含此前置检查的新诊断提交；历史runtime仍为ca7885d，
未使用的RUN ID及八卡30分钟总预算不变。不能继续使用仅做finite检查的旧诊断。

CPU observer与无observer对照必须保持梯度、参数、Adam状态一致后再上卡。
46固定依赖下实际执行`tests/test_r4_first_update_replay.py -v`，3/3通过；
冻结runtime的`tests/test_r6_kl_metrics.py -v`，3/3通过；启动脚本`bash -n`通过。
均为CPU测试，CUDA_VISIBLE_DEVICES为空，未加载大模型，不能计作GPU重放完成。
冻结提交并备份远端后，使用独立tmux调用：

```bash
bash experiments/records/20260906-integrated-fixes-v1/run_r4_first_update_node46.sh \
  /root/new-scii-r4-first-update-20260907/runtime-ca7885d
```

新RUN：`20260906-integrated-fixes-v1-r4-step2-first-update-8gpu-node46-r1`。
位于同一experiment的共享`runs`，保存launcher、两份SHA、配置、输入哈希、
checkpoint元数据、八卡日志/参数链和全批概率汇总。不复用历史目录。
加载+forward+backward+保存合计最多八卡30分钟，超时保留failed产物；不自动
补跑第二次更新或扩预算。共享盘正保存checkpoint时不启动大文件加载。

判读分别回答：恢复是否可信；第一次更新实际发生了多大变化；Adam切片账是否
与参考实现相符；还有哪些参数/Jacobian证据缺失。报告有效token误差分布、
最差样本、old-current K1和梯度，不把reference KL混进去。不以退出0、单阈值
或低KL宣称根因已修好。若一次更新没有覆盖异常时点，结论为证据不足，先汇报。
