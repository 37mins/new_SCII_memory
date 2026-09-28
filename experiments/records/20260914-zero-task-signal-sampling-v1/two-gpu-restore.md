# 两卡部署验收

2026-09-15：用户批准先用两卡，并允许约60 GiB临时完整状态文件；
A/B仍最多64局、各最多新增4个global step，总上限3小时，不自动重试或长训。
临时状态放46独立 `/dev/shm/sc2-a4-two-gpu-20260915-r1`，原A4与部署不改。
这只解决四卡checkpoint到两卡的兼容性，不是新的RL机制或学习修复。

父状态仍为 `continue_a4.py` 登记的A-r2 step4。使用今天已校验的共享副本：
`/mnt/data0601/yangjinyue/new_SCII/staging/20260915-scene-B-r2/shared/yangjinyue/new_SCII/experiments/20260911-execution-failure-feedback-v1/runs/20260911-execution-failure-feedback-v1-spb2-original-l6-g2-A-rl4-4gpu-s0-r2/checkpoints/global_step_4`。
原始副本来源及23个文件hash见 `scene-B-r2-checkpoint-copy.json`。

原加载器按world_size拼文件名，保存的是DTensor模型状态与原生扁平Adam状态，
不是直接可跨卡数读取的DCP。采用 [PyTorch 2.7官方FSDP接口](https://docs.pytorch.org/docs/2.7/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel.scatter_full_optim_state_dict)：
先四个CPU/Gloo进程恢复原布局，再导出FULL_STATE_DICT和full_optim_state_dict；
两卡使用官方scatter_full_optim_state_dict重新分片。不手工拼Adam，不重置step。
CPU转换保留DTensor的本地值、全局shape/stride/placement，只把设备mesh移到CPU。

`check_fsdp_reshard_cpu.py` 使用小网络测试四→二恢复、Adam计数7→8及一次更新
对普通模型/Adam参考的一致性。首两次CPU测试因参考优化器共享step张量失败，
已按每个普通参数分别克隆状态，第三次通过：恢复值完全相同、一次更新通过、
CUDA未初始化。失败脚本和日志在上述独立目录保留；这次测试返工不算项目bug修复。
随后增加device_mesh，使小测试覆盖真实DTensor保存类型；CPU/Gloo四→二同样通过，
恢复模型/Adam逐值相同、一次更新与普通Adam参考相符，计数7→8、CUDA未初始化。
对应 `cpu-dtensor/result.json` 及 `cpu-dtensor.log`。Torch为2.7.1+cu128。

真实SpB的隐藏GPU导入检查在Triton autotuner初始化报 `0 active drivers`；
没有开始大状态加载。正式转换允许GPU0/1提供CUDA导入上下文，四个Gloo进程
按rank%2选择设备；FSDP模型/Adam和转换计算仍在CPU。不改FLA/Triton代码。
`export_a4_full_cpu.py` 仅转换一次，不forward、不backward、不更新。
导出模型及每个Adam状态张量hash、计数和参数组。真实两卡验收必须再核对这些值、
scheduler及明确的RNG映射，并验证一次更新；小CPU测试不冒充真实A4验收。
四→二不能保持原四rank逐位随机轨迹；A/B必须采用相同的两卡映射及游戏种子。

尚未启动GPU恢复或SC2。正式A/B启动还需两卡加载入口、完整有效配置及CPU回归，
不能将本文件或转换完成当作训练已启动。其余候选和训练语义保持冻结。

CPU转换入口 `run_export_node46.sh` 从clean SHA启动，先验父run完成状态和内存。
控制目录 `export-r1`、`budget.json` 只创建一次；从转换开始计时3小时，单次CPU
转换最多30分钟，失败不自动重试。现有46 Torch/Transformers及共享权重原位读取。

20:26:13（北京时间）启动 `export-r1`，但GPU0已被新增vLLM PID3960530占用，
空卡检查返回非零，控制退出码1。未到budget.json创建处、未加载真实A4、未创建
full状态、未开始SC2/PPO。原失败控制和冻结 `code-1e12d91` 保留。没有自动重试。
冻结SHA `1e12d91bf82494205b0dffb40f078f463b0bd2a3` 的完整私有bundle已在46校验，
SHA256 `c7ed0975fc7344caaad073e91c437caf7bddd1a1352fdcc14202f47b5e488bdd`。
20:28 GPU0再次释放，但没有据此自动重试。已请求用GPU1仅提供CUDA导入上下文
先做CPU转换；该新尝试未批准前不执行。

后续用户明确确认新尝试：`export-r2` 仅物理GPU1提供Triton导入上下文，
四个进程仍是CPU/Gloo计算。只改此次导出入口的GPU可见性及新控制目录，
不取消空卡检查、不覆盖r1、不单卡训练。r1未创建budget.json，因此从本次
真正转换启动起计原三小时总预算；单次导出仍最多30分钟，失败不自动重试。

## r2结果与路径修正准备

r2冻结SHA `b45bd825bb4d96cd66699ff668ffb690638250b4`，20:38:21启动，
20:38:33退出1。错误在AutoModel导入阶段：
`ModuleNotFoundError: No module named 'transformers_modules.SpikingBrain-2'`。
Transformers 4.55本地动态模块名直接取目录basename；导出入口直接传带`2.0`
的原始模型目录，导致相对导入断裂。标准训练入口使用`<run>/model`，没有这次
新增工具的路径问题。未读取A4大张量，full目录为空，无forward/PPO/游戏。

`tests/test_spb_export_model_alias.py` 在46隐藏CUDA用小模型复现原路径失败，
同一内容的`model_source`目录别名通过；1项CPU测试3.628秒、退出0，CUDA未初始化。
`alias-cpu.log`和r2失败日志保留；后者SHA256
`ce86afd10ae2178003b97bf70b35a1b45519b38d917eb761bd95fb706054af4e`。
这属于本轮导出工具返工，不记作原项目RL修复。

r3只准备无点目录别名及新控制记录；不改模型、依赖或训练机制。读取已有
budget.json，不重置deadline：2026-09-15 23:38:21北京时间。真实重试需用户
另行确认；未获批前不部署、不启动。

用户随后确认r3，但21:00–21:01预检查GPU1已被PID4036797占用，cwd为
`/mnt/data0601/cjy/lineargroup`。GPU3空闲，r3目录仍未创建、未消耗新尝试。
准备 `SC2_EXPORT_GPU` 显式选择一张卡，默认仍1；不自动搜卡或自动重试，实际
选择写入launch.json。已询问是否允许46任意一张已确认空闲卡仅作导入上下文，
批准前不改到其他卡启动。两卡训练授权与本次单卡上下文授权明确区分。

## r3资源拦截

用户已明确允许46任意一张空卡仅提供本次转换的导入上下文。21:04:12核对GPU3
显存0后，以冻结SHA `dde8dd748804a1d10dde5cd71c655456c8d7a4f6` 在21:05:13
启动r3控制器。启动前GPU3已被新增vLLM PID4043089占用，空卡检查退出1；
`export-r3/launch.json`未创建，因此转换进程没有执行、没有读取A4或做PPO/游戏。
21:06:56该PID工作目录为`/mnt/data0601/cjy/lineargroup`，GPU3显存62845MiB；
46全部八卡均有显存占用。未停止该任务，也未自动启动第四次尝试。

冻结完整私有bundle在46校验SHA256为
`dc5bce3c45b19d866d9ab2af06b2412473327a49cced12208eb3ef13d3c703f0`。
r1/r2/r3失败、原checkpoint与部署均保留。现需协调稳定可用的卡位，不能把
某次瞬时空闲当作资源已预留。两卡真实恢复和RL A/B仍未验收/未启动。
