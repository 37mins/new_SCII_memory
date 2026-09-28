# 分组原始分数判定 v2

2026-09-15。状态：实现及 CPU 验证通过；19:37用户要求查空卡并启动上述A/B，
按已说明的四卡串行/3小时预算执行。资源暂不满足，未启动，尚无学习效果结论。
开发基线为 clean `bbc9f53e5df2db1bfa5a3884ba351f3518e844b4`，继续使用
`codex/feature-zero-signal-sampling`，不改原 r6、已有部署和 run。
修改前备份：`/private/tmp/sc2-zero-signal-before-structural-bbc9f53.bundle`。

## 要解决的边界

v1 的 `advantage != 0` 曾把组内完全相同的回报减均值产生的约 `5.72e-6`
浮点残差放行，实际 A/B 两组都更新4步、7/32胜，B没有跳过任何组。
这说明 v1 没有实现预期的信号筛选，不证明 GiGPO 公式错误。
最近 SceneA8/B8 的采样对照两组都关闭过滤，不能当作本候选已验收。

v2 只修改 `verl/trainer/ppo/sc2_task_signal.py` 中的接受判定：仍调用原奖励、
invalid penalty、折扣与 GiGPO；在减均值前检查每个比较组的原始分数是否相同。
episode 项取每个 trajectory 的第一条计分，与冻结函数一致；step 项使用原
`build_step_group`，只比较同一 anchor 组内的折扣回报（包含既有无效动作惩罚）。
至少一个启用的分支存在分数差异，且实际联合 advantage 非零，才保留整组。
无分数差异的组即使有 reduction residual 也跳过；不设置新的 epsilon。

依据是 [DAPO（2025）§3.2](https://arxiv.org/html/2503.14476v2#S3.SS2)
和算法1的零组过滤/重新采样思想。这里保留 GiGPO 的 episode 与 step 两项，
并非照搬数学题的全胜/全败过滤：全败但回报不同、同胜但 step 回报不同仍保留。
这是一项采样稳定性候选，不是论文已经验证过的 SC2 修复。

## 保留的行为与使用

不改 `core_gigpo.py`、loss、reward、模型、optimizer、prompt、history、BO、
解析器、微操、cache、精度和采样温度。默认关闭；启用的参数仍为：

```bash
bash scripts/train_sc2.sh \
  trainer.sc2_skip_zero_signal=True \
  trainer.sc2_signal_max_attempts=3 \
  trainer.sc2_signal_max_batches=4
```

这是配置示例，不是资源授权。复用已有 `_collect_signal_batch`：跳过整组后，
下一次 reset 取种子表的下一项；不把不同场景混成一个比较组，不复用旧 rollout。
直到找到一组或触及次数上限。跳过不运行 old/ref/actor/Adam/scheduler，
不增长 global step。所有游戏都计入预算及胜率；空策略/非有限值/SC2错误仍失败，
不伪造样本、不无限补采。无需另建采样框架或修改种子生成算法。

接受后的 batch 原样进入原训练流程；原多卡复制补齐也未在此修改。
v2 不保证训练中的微小浮点误差全部消失，不人为消除 episode/step 真实差异，
也不处理两个非零分支相消后仅剩残差等未验证情形。

每次采样继续写 `signal_sampling.jsonl`，增加 `signal_kind`：
`exact_zero`、`reduction_residual_only`、`zero_joint_advantage`、
`episode_score_difference`、`step_score_difference_only`；同时记录原始 episode
return 是否不同、episode score 是否不同及有分数差的 step 组数。
`step_score_difference_only` 可能只是调用时序或格式差，不称为战术收益；
`episode_score_difference` 也不能代替胜负和独立评测。

## 验证与下一项实验

CPU 用生产计分函数覆盖相同负回报的残差、相差一个 FP32 ULP 的真实分数、
不同 anchor 的 singleton、折扣/无效动作差异、step权重为零、mask及输入不变。
实际 driver 的小型替身测试验证：残差组不更新、下个场景整组进入训练、预算内
能继续两次实际 global step、预算耗尽不假称 step8完成。替身不验证 GPU 参数更新。

本轮完成51项相关测试：43节点原Torch 2.7.1环境隐藏CUDA，核心28项及历史
log适配5项通过；本地完整Git工作树执行启动/收尾9项、feedback入口3项、
FSDP入口5项、A4训练/评测配置1项通过，均退出0。
系统hash验证通过，25个登记文件中只更新本候选判定文件的hash；其余24个不变。
实际验证的候选文件SHA256为
`b81deb2c289fbb4473cdec1c8876d3c0059b6b9aee44c512caeb6909b4e65bb7`，
核心测试文件为`c281ab1bea08eb5f93d72b0274479740e629f47125be8d95ddb0f811eaad39ff`。

CPU开发测试目录为43的 `/root/new-scii-structural-signal-cpu-20260915-r1`，
由基线Git archive加上述两个候选文件构成，不冒称clean正式运行。成功命令：

```bash
CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONPATH=.:tests:/root/new-scii-spb-level5-96-20260908/python \
/usr/bin/python3 -m unittest test_sc2_task_signal test_scene_signal_audit -v

python3 -B -m unittest discover -s tests -p test_signal_sampling_record.py -v
python3 -B -m unittest discover -s tests -p test_feedback_rl_launch.py -v
python3 -B -m unittest discover -s tests -p test_fsdp_rl_launch.py -v
PYTHONPATH=tests python3 -B -m unittest \
  test_a4_continuation.A4ContinuationTest.test_training_and_evaluation_keep_existing_configs -v
```

服务器同目录 `signal-cpu-final.log` 保存33项通过，SHA256为
`4ab7446c3c313fe56fa55771f00d08bc6a3cca3428b596ae500bc168ba9c3e63`。
两次较宽测试集合曾把依赖Git元数据的入口测试放进仅有archive的目录：
`combined-cpu.log` 42项中3项因无 `.git` 出错，`runtime-cpu.log` 41项中1项同因出错；
这些入口已在本地真实Git工作树通过，没有修改测试或生产代码去绕过Git校验。
失败日志仍保留，hash分别为
`7e2635111de60732c36162589f8fd85c6f01fd67c2f1572da88a91bc6da8f640`、
`220ef33cf1271f9139ae1339d280ef03e9c54ad6295f26cc44e1214dc1888c24`。
这是本轮CPU部署方式的失误，不记作项目学习bug的修复。

待 CPU 通过后提出同输入路线的短 RL A/B：同一 A4 完整模型/optimizer，
Level6/g2，同一四场景表 `[1098341402,42613,17089,3236598124]`，每场景8局，
其余配置沿用 v1；A关闭过滤，B开启 v2。合计64局、最多各新增4个global step，
B可以更少，末尾各保存实际最后一个完整checkpoint。复用现有四卡串行入口，
总上限3小时、最多约120 GiB新checkpoint；本轮已获用户启动指令，运行节点待空卡确认。
正式运行前需记录新 SHA、唯一 RUN_ID、完整有效配置与私有远端备份；不复用 v1 授权。

首先检验是否真的跳过残差组、是否能持续取得有分数差的组及实际完成多少更新，
分别报告 episode 与 step-only 信号。只有连续更新及保存通过，才申请共同种子
独立评测预算。不以过滤后胜率、非零优势或少更新导致的低 KL 宣称学习提升。
若固定预算内仅有一组或没有有信号组，应明确候选不足：过滤不创造奖励差，
不能把信号来源不足的问题掩盖成项目修好，也不自动加游戏或改奖励。

北京时间18:06–18:16资源快照：43八卡被既有srl任务使用；47 SSH超时，不能据此视作空闲。
没有停止任务或使用GPU。GitHub原代理不可达，单次直连也超时；尚未推送，
正式GPU运行前需完成私有远端备份和资源授权。

## 本轮启动核验（19:37–19:39，北京时间）

43、44均八卡被srl训练占用；46仅GPU0–1空闲，其余为cjy/yht任务；48仅GPU0–1
空闲，其余六卡有任务；197八卡均在使用。42/47 SSH超时，198认证失败，不把
不可连接节点视作空闲。没有停进程，没有使用两节点各两卡改变本轮分布式配置。

生产候选仍为 `4ec470581f321ebbf7d0d5279f32f9e53c1a06bf` 的判定代码，
本轮只给旧编排增加 `SIGNAL_REPLICATE`（默认r1），避免硬编码r1碰到已完成结果；
训练配置、三小时共享截止时间及空卡检查不变。本轮预留 `SIGNAL_REPLICATE=r2`：

- `20260914-zero-task-signal-sampling-v1-spb2-A4-l6-A-32games-4gpu-r2`
- `20260914-zero-task-signal-sampling-v1-spb2-A4-l6-B-32games-4gpu-r2`

目前尚未创建正式run或tmux训练会话，也没有后台占卡/等待程序。没有消耗GPU预算。
43已有完整历史bundle备份 `runtime-4ec4705.bundle`，SHA256为
`933bc8017067a96867cdccf99cca4b9ee30b13ac3fbdde2bd27a6e5a17dddeb1`，
位置 `/root/new-scii-structural-signal-cpu-20260915-r1/`。新编排提交也须备份后才启动。

本轮仅编排入口的10项本地测试通过：
`python3 -B -m unittest discover -s tests -p test_signal_sampling_record.py -v`；
`bash -n` 和 `git diff --check` 通过。初次新增ID测试9/10通过，发现macOS Bash3.2
对裸 `[[ ... ]]` 失败并未按期望随 `set -e` 退出，已用显式 `|| exit 2` 修正；
该返工只属于本轮编排，不记为RL修复。与生产候选 `4ec4705` 比较，
`verl/`、`gigpo/`、`agent_system/` 和模型/训练配置无diff。

19:49用户批准先按两卡方向执行；先验收A4模型与Adam四分片到两卡的完整恢复和
一次更新，不用初始optimizer代替，不在恢复未验收时直接启动完整A/B。
现有 `FSDPCheckpointManager.load_checkpoint` 按当前world_size读取同分片数文件，
没有现成4→2恢复入口；本轮尚未实施重分片或加载checkpoint，也未消耗GPU预算。
19:49–19:50重新检查：46 GPU0已被cjy的vLLM进程3902217占用（工作目录
`/mnt/data0601/cjy/lineargroup`），仅GPU1空闲；48 GPU0–1新增yangkun任务，
现八卡全占；43/44/197也全占，47仍SSH超时。没有同节点空闲两卡，未启动任何
训练、重放或后台等待；没有停止其他任务。需要协调两卡后再做恢复验收。
