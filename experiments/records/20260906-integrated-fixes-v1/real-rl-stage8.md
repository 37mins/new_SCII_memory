# Level 6 真实 RL：96 步目标，step-8 验收

状态：2026-09-07 用户明确批准仅补八卡评测入口的`env.rollout.n=8`。
r2初始评测的失败产物保留，见[失败证据](real-rl-r2-launch.json)。r3从已推送的
`aa1d20dbe87ddaf67213e8fa67e15e4eb2e56835`于09-07 00:16北京启动；
00:44:55初始评测完成（1/8胜，八局自然结束，退出0）；00:45:21自动启动train8。
08:13:09北京因第6批的空BO同帧循环中断，原始结果为`failed / exit 1`。
已完成5个global step、145次optimizer更新；五步各25项checkpoint（各约49.83GB）
保留。第6批7局自然落败，最后1局未结束，不伪造第48局结局。step-8评测与后续
96步均未启动。08:18核验本任务进程退出、GPU compute进程为空；没有删除文件。
冻结源码仍为`aa1d20d`，未改prompt、PPO或游戏行为。
见[失败现场与退出证据](real-rl-r3-livelock-stop.json)。
见[首步证据](real-rl-r3-step1.json)、[连续训练证据](real-rl-r3-training-progress.json)。
见[启动证据](real-rl-r3-launch.json)及[初始基准](real-rl-r3-initial-eval.json)。
训练路线和预算不变。复用本experiment及现有
开发分支，不新增学习机制。原r6、旧部署、r32超时失败与CPU定位结果均保留。
代码基点`9f518311aca700849af9d6109782b206fd3d2768`；本轮仅补实验配置入口和
八局/checkpoint评测的结果记录，不修改模型、训练器loss或SC2行为。

## 冻结配置

原始`Qwen/Qwen3-4B-Instruct-2507`，revision
`cdbee75f17c01a7cc42f958dc650907174af0554`，无项目SFT、无RL续训。
模型manifest SHA-256 `c82106ca091f4797af1217cca7578618bb2a5cd7c4136ca0d2d06c0a77ed00a7`。
使用已有FSDP参数FP32和FP32温度归一化配置；独立reference也显式选择相同
参数/归一化配置并从原模型计算，不混入历史BF16概率。BF16 autocast、HF生成
cache、eager保留，不带入RMSNorm、全FP32 forward、early-stop或优化器保护。

Level6 Harder/Air、Flat128/Base75689、canonical g2、env seed0；训练每个
global step八条rollout，GiGPO mean_norm、gamma .95、lr1e-6、mini8/micro1、
epoch1、grad_clip1、entropy .001、reference KL .01、原PPO/参考KL截断均冻结。
prompt/解析器/微操/自动战斗/reward不变：6条BO、4096/96 tokens、history0、
memory off、step shaping off。既有PPO前/后观测开启，生成大词表观测关闭；
不安装r32逐optimizer整批重算的observer。此为配置整合后的真实训练，不是
与r32四卡重放的单变量A/B，也不宣称旧复制补齐计权或所有数值问题已修复。

44八卡运行，复用固定Python和共享权重，不下载/安装。入口：
`bash experiments/records/20260906-integrated-fixes-v1/run_real_rl_node44.sh <mode>`，
mode依次为`initial-eval`、`train8`、`step8-eval`。
套件RUN前缀`20260906-integrated-fixes-v1-qwen-instruct2507-l6-8gpu-fp32-rl96-gate8-s0-r3`，
三个独立run附加对应mode，不覆盖旧目录。输出沿用本experiment共享runs目录。
启动前保存clean/pushed SHA、依赖/模型/SC2哈希及完整Hydra有效配置。

## 运行与评测边界

总计划96个global step；第一段`total_training_steps=96, total_epochs=8`。
本数据每epoch一个batch，故运行8次真实rollout/PPO循环后自然退出；不是KL
early-stop，不把optimizer update当global step。constant LR从开始就按总计划
记录，SAVE_FREQ=1，无checkpoint保留数量上限。每步保存8卡model/optim/extra
和data.pt，25项必需文件及tokenizer/config；验收逐步核验，不只检查最后一步。
预计第一段4–8小时，12小时为运维上限；每个独立评测3小时上限，不自动重试。

初始模型与step-8分别在新进程作八局独立评测：相同八卡、greedy生成、96-token
预算、同一地图/对手/BO/系统与env seed0，validation RNG base=1000并从首个
reset开始，实际八个游戏seed写入各自`evaluation_seeds.jsonl`并逐项比对。
`val_kwargs.n=8`与`val_rollout_n=8`同时设置，模型不更新。step-8评测通过现有
`resume_path`载入8卡checkpoint，manifest明确记录来源，不把它写成初始权重。
两次自然游戏及原始prompt/回答全量保留；训练rollout与独立评测分开统计。

逐局报告胜负、自然终止、reward及模型是否参与；只在真实可评估模型动作上
计算格式/语义/执行接受率，并展示空答、重复、截断、非法动作等原文。有效
动作不等于单位建成。八局仅用于发现明显退化，不足以证明小幅涨分或架构优势。
若游戏状态不同，原始回答不是同输入因果对照，不能用单条不同回答下结论。

不要求step-8胜率上涨，不以KL>0.015、单个token变化或有限梯度单独判失败。
已确认NaN/Inf、空策略批次、归因错误、未更新或checkpoint缺失则停止并保留
证据。明显退化须由持续输出异常/有效动作下降及配对评测等具体证据支持，
不临时发明阈值，也不把正常小样本胜负波动定成bug。

第一段及独立评测通过后才续到96，不提前启动第9步。续训复用同一代码与
step-8的模型/Adam/scheduler/RNG/dataloader；另用唯一run保留9–96产物。
SC2环境自己的随机数状态不在现有checkpoint中；续训前明确记录其seed序列，
不冒充与不中断96步逐位相同。本次不顺手修改环境checkpoint机制。

## 入口变更说明

`scripts/train_sc2.sh`现在接受显式Hydra CLI overrides并保存在原有command
record中，默认行为不变。`run_record.py`记录96步总目标与resume路径，允许
step-8的val日志，并按声明的八局数量核验评测；它不修改训练/评测数据。
本地26项启动/结果测试通过，包含8步预算、8卡checkpoint评测、八局完整性；
GPU/真实游戏验收尚未完成。

## r3评测入口修正

相对冻结r2，唯一非路径配置变化为两段evaluation显式`env.rollout.n=8`，
满足现有训练器的八卡初始化检查；实际validation与generation仍各声明8局。
train8所有配置保持不变；另换RUN前缀/Ray临时目录，防止覆盖r2。
测试现在读取真实评测入口的override块，不再手写一份遗漏同一项的参数：
旧入口复现1项失败（其余25项通过），修复后本地26项全过；服务器99项全过
（8.620秒，无跳过）。三段真实入口dry-run、资产哈希及训练器`_validate_config`
全部通过；归一化新旧路径后，两段评测仅差`env.rollout.n`，train8零差异。
只改实验入口、测试和文档；不改共享train_sc2入口、模型、PPO或SC2模块。
备份`/private/tmp/new-scii-before-eval8-entry-r3-20260907.bundle`，SHA-256
`5e8621c7797563614b4a42f92cf6967ed993f988e57408c039f31422b9c2fffc`。

## r3初始独立评测

未训练的Instruct-2507，八局1胜7负，均为自然游戏结局；七局使用Planner，
一局在模型决策前落败。原始568行回答日志中302行是实际应用、可评估的模型
动作，284行合法（94.04%）；不把已结束环境占位行计入分母。18项不合法中
17项为`marine x N`不符合当前names-only协议，1项为marauder超过剩余待办。
该协议保持冻结，未因这些输出新增修改。例：`- marine x 23`、
`- marine / - mule / - siegetank / - marauder / - siegetank / - marauder`。

八局等权平均环境reward为-38.313125；原`val/text/test_score=-29.500944`
按展开后的真实回答计权，71次决策的获胜局权重更大，不能混称八局均值。
两者均保留，训练reward/loss不改。实际评测seeds与预先登记的八个数完全一致。
这些是初始模型的基准，不是学习提升或step-8验收结果。

产物限制：八个worker在同一秒使用同一日志目录并写`replay.SC2Replay`，
实际仅保留一份回放。代码依据为`base_player.py`按秒生成`log_path`、
`envSC2.py`固定回放文件名。八局结局和按env索引的prompt/回答仍可追溯，
但不能声称八份回放齐全。此处未修改运行代码，也没有额外重跑评测。

## r3首步与重新采样

第1个global step完成完整八局、21次optimizer更新和25项checkpoint保存。
八局均自然落败，环境reward均值-47.746；真实动作155/163合法（95.09%）。
裁剪前grad_norm均值1.582、PPO clipfrac 0.004、更新过程PPO K1 0.095、
reference capped K3 0.101，已记录数值均有限。response clip为0。
这些为控制台三位小数值，不是峰值；显示的`actor/lr=0.000`只是舍入，
实际保存的scheduler确认`_last_lr=[1e-6]`，学习率未改。

八卡各21项pre、21项update、21项post记录齐全。pre与old重算差为0；
post相对old的token加权K1为0.305956，绝对差P50=0、P95=0.000217、
P99=15.624187、最大48.321297；37/2179 tokens绝对差超过1，15个超过20。
这是分布描述，不是新增失败阈值，也不等于cached/full生成路径已经一致。
163条真实样本沿用旧copy补齐为168条，这5条仍重复计权；未将它误称为零权重padding。

checkpoint约49.83GB，25项文件的ZIP目录结构可读，八份模型分片SHA-256已记录；
尚未做完整tensor重载/所有文件CRC扫描，step-8评测将实际载入该步checkpoint。
首步耗时2207秒，其中采样1427秒、actor更新97秒、保存642秒。
01:33第2批118条真实可评估回答中116条合法。回答按已完成八局reset的累计
step_index边界区分，不能假设计数器每个global step归零。
这说明首次完整更新后能够重新采样，未观察到整体格式崩溃；不是学习提升或
连续8步验收通过。尾部概率变化、游戏效果、回放缺口仍保留，不据此改代码。

## r3连续运行

第2步完成31次optimizer更新，八卡各31项pre/update/post记录齐全且有限。
246条真实决策补齐为248条，沿用旧copy计权，未冒充零权重padding。

| global step | 训练胜局 | 八局等权reward均值 | 真实动作合法率 | grad_norm均值（裁剪前） | PPO clipfrac |
|---|---:|---:|---:|---:|---:|
| 1 | 0/8 | -47.746 | 155/163（95.09%） | 1.582 | 0.004 |
| 2 | 0/8 | -47.404 | 240/246（97.56%） | 0.511 | 0.002 |
| 3 | 2/8 | -29.833 | 218/230（94.78%） | 0.885 | 0.002 |
| 4 | 0/8 | -45.997 | 363/404（89.85%） | 0.289 | 0.002 |
| 5 | 0/8 | -48.721 | 93/99（93.94%） | 0.272 | 0.001 |

第2步更新过程PPO K1为0.065、reference capped K3为0.155；更新后整批
token加权old-policy K1为1.127070，最大log-prob绝对差139.982353。
不同阶段及汇总口径不能混称同一KL；尾部概率变化仍未解释，不因数值有限就宣布修复。
response clip仍为0。第2批回答由step-1权重生成，不是step-2权重的独立评测。

保存耗时798秒（13.3分钟），期间共享盘有空间、进程存活并处于I/O等待，
未因观察超时或低GPU利用率重启。保存结束后核验25项文件ZIP目录结构及八份
模型分片SHA-256；没有额外做完整tensor重载。第3批由原进程自动接上。
第3–5步分别完成29/51/13次optimizer更新，八卡pre/update/post均齐全、有限。
补齐复制分别为2/4/5条；response clip均为0。第4步仍出现`marine x N`协议错误，
各批还存在超剩余待办数量的动作，没有顺手更改prompt或解析器。

第3、5批各有1局在模型决策前结束，终局占位已排除，非空整批仍正常更新。
控制台reward/length只汇总PPO中存在的7条trajectory，因此第3、5步reward
分别显示-26.952/-48.538，而非上表的八局等权值。第3步控制台胜率0.25仍与
完整8局相符；不能笼统说所有指标都排除了这局。这里只澄清统计，不改reward/loss。

第3–5步全批更新后token加权old-policy K1为0.783614/0.689516/0.100066，
尾部最大绝对差仍为100.625/100.211/76.250；不是cached/full路径已一致的证据。
三步checkpoint的75项文件均经ZIP目录检查，小型状态和诊断哈希已记录；
停训后补齐新增24份model分片哈希，连同前两步共40份。没有完整扫描optimizer
分片哈希或做tensor重载。只能确认五步更新与保存完成、原始训练胜局2/40；
第6批运行错误阻断八步验收，不能把高动作合格率或有限梯度称为项目修好。

## r3第6批失败：空BO同帧循环

env4停在游戏18:13、SC2帧6123。Commander剩余待办`{'siegetank': 3}`，
但本方观测中没有被识别的`factory`，冻结Planner的前置条件过滤将tank删去。
最终query与模型回答分别为：

```text
The remaining BO list in this phase:
none

<answer>
Here are the Build Orders with attack instructions:

</answer>
```

输入还提供了同样的空one-shot。因此这一回答符合空候选输入，不能据此归因
为模型退化。解析结果`[]`返回后，`llm_player.py:660–663`仍执行：

```python
while self.current_bo_index >= len(self.bo_data):
    self.bo_data = self.commander.renew_bo(self.get_obs())
    self.current_bo_index = 0
```

空队列使`0 >= 0`持续成立，本帧无法返回。04:26:53–08:12:54北京记录了
1,891条同输入、同空回答，均`done=false`、reward=0，未进入第6批PPO更新。
它们被标为`semantic_action_evaluable=false`，故“可评估动作合法率”会漏掉此故障。
这不是空PPO批次已经发生更新，也不是因KL超过某个值停训。

过滤依据见冻结`planner.py:803–886,1147`；query拼接见`:1168–1184`。
观测中的`2  with techlab`可由独立TechLab名称生成，不证明两座Factory被漏写；
本局factory为何不在观测中尚未确认。此前混合worker日志的20:19不是本局时间。

已保留原失败结果、所有日志及step1–5 checkpoint，只终止核验属于r3的训练
进程和遗留SC2进程。直接阻塞是同帧无界刷新；未实施任何候选修复。下一项
拟仅限制空BO刷新为每帧最多一次，让游戏帧返回，再检查现有执行/微操路径仍可运行。
先请求用户批准CPU复现与回归；不同时改前置过滤、prompt、奖励或开始GPU重跑。
本次归档验证为JSON解析、24份新增模型哈希与读取结果交叉核对及`git diff --check`，
均通过；不是新增代码修复测试，也没有启动新GPU实验。

09-07 08:41北京追加只读来源核对：从197指定r6运行的冻结归档读取源码，
归档SHA仍为`cad52ff4…99ce32`。r6的`llm_player.py:660`刷新循环与
`planner.py:820–887`前置过滤函数，分别与r3对应片段逐字相同；并非这次
`env.rollout.n=8`入口修正引入。r6另有`ct >= 20`人工终止信号，当前没有；
该差异改变轨迹收集范围，但不能证明r6当时也触发过相同卡点。
旧625/768胜的结果仍成立，不能据此宣称旧循环正确或新模型退化。

候选边界的源码依据：`llm_player.py:596`先执行维护与自动战斗，`:665`
仅在队列非空时执行BO；`base_player.py:162`等待整个`run`返回。故拟修改
的是同帧刷新控制流，不是添加战略动作或关掉微操。尚未实施或做行为回归，
仍待用户批准。完整来源与比较范围已追加到失败JSON，不使用哈希不符的本地旧包。

随后用户批准仅修同帧循环、先CPU回归。09-07 09:03北京，候选`42329b1`
在44固定依赖下105/105通过；唯一运行代码变化为上述`while → if`，详见
[单点修复与CPU证据](empty-bo-frame.md)。未上GPU，r3仍保留failed身份，
不能把新CPU结果追写成r3的第6步或step-8验收结果。

续训前还有编排事项：`scripts/run_record.py:123`固定验收1..target，`:128`
按target计算局数，会误判独立的9–96续训run。训练器已支持恢复counter与状态；
目前只读定位，未改记录器/启动器，也未启动第9步。后续只需区分续训区间与总目标，
不得借此改训练机制或伪造前8步日志。
