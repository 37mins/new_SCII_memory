# 服务器清理候选（C1已清理，其余待确认）

2026-09-15 09:33 北京时间补记：用户仅批准C1，已删除84个缓存子目录，释放
5,057,613,824 bytes（4.7103GiB）内存盘。C2–C4和P1未执行。下文保留09:19的
原始盘点快照；执行结果与逐目录凭据见 [C1清理记录](cache-cleanup-20260915/README.md)。

核验：2026-09-15 09:07–09:19 北京时间。范围为43本地项目目录和共享盘 `/mnt/data0601/yangjinyue/new_SCII`；未将其他节点的本地盘纳入清单。**只读盘点：删除0个文件、停止0个进程。**

43系统盘剩34GiB，内存盘剩415GiB，共享盘剩约2.3TiB。共享新项目目录发现70个大checkpoint，模型/optimizer/extra/data文件逻辑尺寸合计约3.46TiB；这不是都可删。下面候选容量按实际分配块统计，不跟随软链接。

| 编号 | 候选 | 预计释放 | 建议与代价 |
| --- | --- | --- | --- |
| C1 | 31个旧临时目录内84个HF/Triton/Inductor缓存子目录 | 4.71GiB，内存盘 | 推荐先清；检查时未见进程打开/映射其父目录。可以重建；保留Ray日志与回放。 |
| C2 | 旧A4→A8续训的model/optim分片 | 59.19GiB，43系统盘 | 可考虑明确退役；不是当前A4，也不是昨晚新A8。删除后无法恢复该旧A8或直接按原入口评测。 |
| C3 | 已结束零组过滤A/B各step8的model/optim分片 | 合计118.38GiB，共享盘 | 候选未采用，两组各7/32胜、B跳过0组；若确认不再复测可退役。两组模型哈希不同，不能称为相同文件。 |
| C4 | 旧L6长训12个中间点的model/optim分片 | 合计710.28GiB，共享盘 | 不建议首轮直接删。需确认无需逐点复评/重放并核对跨节点引用；保留该run的step4/24/60及续训run的step64。 |
| P1 | 三组旧SC2进程共24个 | PSS合计约18.27GiB主机内存 | 这不是文件清理，须单独批准终止。三个仍被引用的临时目录已从C1排除。 |

**C2–C4不是可再生成缓存。** 未核实到这些训练状态的另一份完整备份；删后不能直接回滚到它们。仅保留日志不等于保留checkpoint。当前加载器在纯评测下也强制读取optimizer，不能承诺“只删optimizer仍能照旧评测”：`verl/utils/checkpoint/fsdp_checkpoint_manager.py:76`；本轮不修改加载器。

建议先批准C1；若系统盘需腾空间，再单独确认C2的旧A8退役。C3/C4作为下一档，不把“不涨分”或“run failed”自动当作可删除依据。清理系统盘/共享盘不会直接解决所有主机OOM；C1和P1才直接涉及43内存，但不保证足以让B重试通过。

## C1：精确缓存父目录

下表**只清列出的子目录**，不删除整行父目录，不删除Ray session日志、SC2数据或checkpoint。无活跃引用是本次快照，执行前必须重查。

| 父目录 | 仅限子目录 | MiB |
| --- | --- | ---: |
| `/dev/shm/a4s_0913_r1` | `hf`、`torchinductor_root`、`triton` | 206.69 |
| `/dev/shm/a4t_0911_r1` | `hf`、`torchinductor_root`、`triton` | 332.90 |
| `/dev/shm/a8e_0911_r1` | `hf`、`torchinductor_root`、`triton` | 134.16 |
| `/dev/shm/me_A4_0913_r1` | `hf`、`torchinductor_root`、`triton` | 108.07 |
| `/dev/shm/me_A5scene_0913_r1` | `hf`、`torchinductor_root`、`triton` | 107.98 |
| `/dev/shm/oc_0913_r1` | `hf`、`torchinductor_root`、`triton` | 92.04 |
| `/dev/shm/sc2_coverage_b5_update3_0914_r2` | `torchinductor_root`、`triton` | 135.84 |
| `/dev/shm/sc2_example_order_0912_r1` | `torchinductor_root`、`triton` | 53.80 |
| `/dev/shm/sc2_feedback_A_0911_r3` | `hf`、`torchinductor_root`、`triton` | 63.77 |
| `/dev/shm/sc2_feedback_B_0911_r3` | `hf`、`torchinductor_root`、`triton` | 99.11 |
| `/dev/shm/sc2_feedback_rl4_A_0911_r2` | `hf`、`torchinductor_root`、`triton` | 645.38 |
| `/dev/shm/sc2_feedback_rl4_B_0911_r2` | `hf`、`torchinductor_root`、`triton` | 1159.57 |
| `/dev/shm/sc2_fixed_feedback_0911_r1` | `torchinductor_root`、`triton` | 58.54 |
| `/dev/shm/sc2_original_a4_0912_r1` | `torchinductor_root`、`triton` | 53.98 |
| `/dev/shm/sc2_phase_deadline_0914_r1` | `hf`、`torchinductor_root`、`triton` | 84.47 |
| `/dev/shm/sc2_prereq_context_0914_r1` | `torchinductor_root`、`triton` | 60.94 |
| `/dev/shm/sc2_spb_l5_96_0908_r5` | `torchinductor_root` | 0.00 |
| `/dev/shm/sc2_spb_l5_empty_0908_r1` | `torchinductor_root` | 0.00 |
| `/dev/shm/se_A4_0913_r1` | `hf`、`torchinductor_root`、`triton` | 60.96 |
| `/dev/shm/se_A5scene_0913_r1` | `hf`、`torchinductor_root`、`triton` | 61.14 |
| `/dev/shm/ss_0_0914_r1` | `hf`、`torchinductor_root`、`triton` | 60.96 |
| `/dev/shm/ss_1_0914_r1` | `hf`、`torchinductor_root`、`triton` | 61.05 |
| `/dev/shm/ss_2_0914_r1` | `hf`、`torchinductor_root`、`triton` | 60.96 |
| `/dev/shm/ss_3_0914_r1` | `hf`、`torchinductor_root`、`triton` | 60.87 |
| `/dev/shm/ss_4_0914_r1` | `hf`、`torchinductor_root`、`triton` | 60.96 |
| `/dev/shm/ss_5_0914_r1` | `hf`、`torchinductor_root`、`triton` | 101.13 |
| `/dev/shm/ss_6_0914_r1` | `hf`、`torchinductor_root`、`triton` | 60.78 |
| `/dev/shm/ss_7_0914_r1` | `hf`、`torchinductor_root`、`triton` | 79.84 |
| `/dev/shm/ss_A_0915_r1` | `hf`、`torchinductor_root`、`triton` | 239.12 |
| `/dev/shm/zs_A_0914_r1` | `hf`、`torchinductor_root`、`triton` | 259.59 |
| `/dev/shm/zs_B_0914_r1` | `hf`、`torchinductor_root`、`triton` | 258.71 |

## C2：43本地旧A8（1个checkpoint）

该实验已完成、global4→8，属于早先旧采样顺序的续训，后来的Coverage/零组/本轮训练均从原A4另起。删除范围仅为下列checkpoint的 `actor/model_world_size_4_rank_0..3.pt` 和 `actor/optim_world_size_4_rank_0..3.pt`，共8个普通文件；逐文件绝对路径在配套JSON里。

`/root/new-scii-execution-feedback-local-20260911/shared/yangjinyue/new_SCII/experiments/20260911-execution-failure-feedback-v1/runs/20260911-execution-failure-feedback-v1-spb2-A4-l6-s4plus4-4gpu-s0-r1/checkpoints/global_step_8`

## C3：旧零组过滤A/B（2个checkpoint）

两个run均completed/exit0，既有小型结果及回放保留；不是昨晚的 `scene-A` 或失败的 `scene-B`。每个仅列model/optim共8个分片，不删除run目录。

- `/mnt/data0601/yangjinyue/new_SCII/experiments/20260914-zero-task-signal-sampling-v1/runs/20260914-zero-task-signal-sampling-v1-spb2-A4-l6-A-32games-4gpu-r1/checkpoints/global_step_8`
- `/mnt/data0601/yangjinyue/new_SCII/experiments/20260914-zero-task-signal-sampling-v1/runs/20260914-zero-task-signal-sampling-v1-spb2-A4-l6-B-32games-4gpu-r1/checkpoints/global_step_8`

## C4：旧Level6长训中间点（12个checkpoint）

原run以失败结束，已有记录为完成63次更新后SC2失联；失败状态不否定之前成功保存的中间点。该run的step4曾用于gamma对照，step24曾用于失败反馈评测，step60用于续训，因此保留这些完整状态；续训另一个run的step64也保留。

父目录：
`/mnt/data0601/yangjinyue/new_SCII/experiments/20260910-spb-fsdp-layer-wrap-v1/runs/20260910-spb-fsdp-layer-wrap-v1-spb2-l6-g2-opening-quota-rl96-8gpu-s0-r1/checkpoints`

仅列以下12个点内model/optim分片，共192个普通文件；未发现普通文件硬链接（本次nlink均1），其他节点的加载/排队引用尚未核实：
`global_step_8, 12, 16, 20, 28, 32, 36, 40, 44, 48, 52, 56`。
这档删除会失去这些中间模型的复评及完整重放能力，不作为默认清理项。

## P1：仍在引用临时目录的旧进程

| 对应运行 | PID | 临时目录 | PSS合计GiB |
| --- | --- | --- | ---: |
| 昨晚失败B | 592720–592727 | `/dev/shm/ss_B_0915_r1` | 6.08 |
| 旧L6长训 | 3581177–3581184 | `/dev/shm/sc2_spb_fsdp_rl96_0910_r1` | 5.95 |
| 旧L6从60续训 | 3608382–3608389 | `/dev/shm/sc2_spb_l6_resume60_0911_r1` | 6.23 |

检查时三组均为 `SC2_x64`、sleeping，cwd分别对应已登记的冻结源码。没有关闭它们；执行前还需核对PID启动时间/父进程与run状态，不能仅凭旧PID操作。43目前占八卡的另一任务位于其他用户目录，不在清理范围。

## 明确保留

- 原始SpB/Qwen权重、SC2安装、当前8.1GiB Python依赖及头文件。
- 原A4和B4完整状态、单场景产生局部改善证据的A5、Coverage A8/B8。
- 昨晚 `20260914-zero-task-signal-sampling-v1-spb2-A4-l6-scene-A-rl4-4gpu-r1/checkpoints/global_step_8`，以及失败B的全部日志与结果。
- 原r6/r7历史基线、Qwen r4数值审计所需状态；此次不因体积大而列删。
- experiment records、manifest、effective config、.done、原始回答、reward记录、回放、Git bundles与冻结源码。代码部署总体不足1GiB，不是主要释放对象。
- 凭据、他人文件、未经核实来源的目录全部排除。

完整逐路径、大小与checkpoint inode/mtime清单见 [JSON](storage-cleanup-candidates-20260915.json)。该文件是待确认清单，不是删除脚本。获得具体编号批准后才重新核验并执行，不因此重启实验、变更训练代码或自动扩大清理范围。
