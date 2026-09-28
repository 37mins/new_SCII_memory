# 空 BO 同帧循环：单点修复

用户于2026-09-07批准仅修同帧循环，先做CPU回归；不启动GPU实验。
状态：09:03北京固定Linux依赖下105/105通过，无跳过、退出0；
候选SHA `42329b1e0f97a225c404a4b20fac768d1a99485d`已推送私有分支。
这是CPU验收，不是SC2/RL验收。完整命令、输出与依赖见[CPU记录](empty-bo-frame-cpu.json)。
代码基点`f90ae0ed9f51a1e0a13c871ec87c589aa8078d89`的运行源码仍对应r3
`aa1d20d`。复用现有隔离分支，不修改44上的r3 deployment、五份checkpoint或失败结果。

## 改动与依据

r3第6批在帧6123收到空候选，模型按空示例返回空BO，原循环同帧重复1,891次。
详见[原始现场](real-rl-r3-livelock-stop.json)。该循环和前置过滤也存在于指定
成功r6的冻结归档中；不能将这次卡住直接归因于新模型或数值精度。

唯一运行代码改动是`players/llm_player.py:660`的`while`改为`if`：

```python
if self.current_bo_index >= len(self.bo_data):
    self.bo_data = self.commander.renew_bo(self.get_obs())
    self.current_bo_index = 0
```

每次`run`至多刷新一次。若仍为空，现有非空队列判断跳过执行，`run/on_step`
返回，下一帧再取新观测；若已有BO或刷新得到BO，仍在本帧走原执行流程。
原维护/自动战斗调用在刷新之前，位置和实现均未变。没有新增补造动作、人工
终止、重试次数、奖励或异常吞噬。前置条件长期缺失仍可能导致每帧空计划；
本修复只消除同帧不返回，不宣称解决生产策略或学习效果。

`configs/system-v1.sha256`仅同步这一文件的校验值；训练配置值不变。
prompt、前置过滤、解析器、BO待办、reward、微操、cache、精度及优化器均冻结。
修改前文件SHA为`54644b5d9176139e4a8c9650f3f5e99bd03651d53f3ad19bb6e01329808a9578`，
修改后为`8dd13723f38687ea3e35a282b9a3f2f190a257190974501e59f6cbc51f0d1c11`。

## CPU 回归

复用现有测试的源码方法提取器，执行完整`LLMPlayer.run`、`BasePlayer.on_step`
及`_finish_current_bo`；仅模拟Commander、SC2服务和维护调用边界，不导入大模型。
测试替身在同帧第二次刷新时明确抛错，避免原同步循环挂住CPU测试。

```bash
CUDA_VISIBLE_DEVICES='' python3 -B -m unittest discover -s tests -p test_empty_bo_frame.py -v
```

同一套六项测试在未改代码上：三项空计划路径复现重复刷新，另外三项通过；
只改上述一行后6/6通过。覆盖单帧返回、连续空计划、新观测、下一帧恢复Marine
命令提交、已有/耗尽队列的正常执行、原Planner异常传播；不把命令提交当作单位建成。
测试准备阶段补齐了替身的Units相加与structures_to_text接口，不属于项目bug修复。

本地完整测试命令`CUDA_VISIBLE_DEVICES='' python3 -B -m unittest discover -s tests -v`
发现102项，77通过、24因缺固定Linux依赖跳过、1因缺gymnasium导入失败；
不改测试去隐藏环境缺依赖。随后从上述clean/pushed候选SHA，在44独立目录
`/root/new-scii-empty-bo-frame-20260907/code-42329b1`完成全量回归，
105/105通过、无跳过，unittest耗时8.417秒。Python3.12.3、Torch2.7.1+cu128、
Transformers4.55.2、Gymnasium1.2.3均复用旧固定依赖；CUDA可见设备数为0。
测试后Git仍clean，没有修改旧部署或安装依赖。
20项源码冻结校验通过；prompt及训练器等其余19项哈希不变。

## 备份与验收范围

备份`/private/tmp/new-scii-before-empty-bo-frame-20260907-f90ae0e.bundle`已通过
`git bundle verify`，记录完整历史和修改前HEAD。SHA-256：
`c4c9b20167c7c0d0d2ac445791207fac8e0f7e13512e096eb4ecf30a01263592`。
回滚使用该HEAD或原r3的`aa1d20d`及其配置，不手工修改旧run。

CPU控制流通过不代表真实SC2连续采样、step-8配对评测或96步通过。
既有全量测试中使用了微型合成Torch模型；没有加载4B权重、启动游戏或GPU训练。
下一次上卡须另行确认，不将本次CPU结果冒充真实连续RL修复成功。
