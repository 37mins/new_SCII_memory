# 空可用技能列表：最小崩溃修复

2026-09-16，用户批准继续修复。仅本地候选及 CPU 回归完成，未部署或重启 GPU。
开发基线 `cd71e6bcced735ccdaf4494830f4c740f2ac55b6`，沿用当前分支；
失败运行的冻结源码 `2e2cf8b4bda86edb46d92f84026ee3a924ab7f11` 不变。

## 证据与修改

46 的 A-r2 在第四批、游戏 13:27 报错。现有日志
`/dev/shm/sc2-v2-pair-node46-20260915-r1/pair-r2/train-A.log:24450–24491`
显示 `obs_to_text_dict` 查询建筑技能时，传入的 CommandCenter 被
`build_progress == 1.0` 过滤掉，`units=[]`；
`/root/new-scii-spb-unitid-20260906/python/sc2/client.py:320` 的 `assert units`
失败，随后包装层报 `SC2 subprocess returned no data.`。

只在 `BasePlayer.abilities_to_text` 完成原过滤后增加两行：没有已建成单位时
直接返回 `[Empty]`，不发送空技能查询。返回值复用该函数原本的无技能文本约定，
不新增观测字段，不伪造动作或终止。非空查询、名称映射、过滤和异常传播不变。
同步 `configs/system-v1.sha256` 中该文件的哈希；没有修改 Planner、BO、
reward、训练配置、优化器、微操或 v2 过滤。

这修复的是环境观测边界崩溃，不是 step6/7 全败或 RL 信号不足的修复。
两个全败批次已自然结束并完成更新，发生在这次异常之前。

## 回归与回滚

新增 `tests/test_sc2_empty_abilities.py`，CPU 提取并执行真实方法、模拟 SC2 查询。
原代码 5 项中 2 项复现空查询断言；增加两行后 5/5 通过，覆盖空输入、
全部未建成、混合列表、已建成但无技能，以及非空查询错误继续向上传播。
它不是 SC2 实局验收。

```bash
PYTHONPATH=tests python3 -B -m unittest \
  test_sc2_empty_abilities test_system_release.SourceFreezeTest \
  test_signal_sampling_record test_wait_for_pair_node46 -v
```

上述共 24 项通过。修改前完整 Git bundle：
`/private/tmp/sc2-before-empty-abilities-cd71e6b.bundle`，`git bundle verify` 通过，
SHA256 `fc5d78632cb17dc26199ab541c366fcff1b6bed096144a26fd5540ec3fe58745`。
旧 deployment、A4、失败日志与已有结果均未修改。后续 GPU 运行使用新 SHA、
新 RUN_ID，A/B 都包含这个边界修复，仍只以 v2 过滤开关区分两组。
旧三小时预算已到期，不自动重试；重启需要新的明确预算。
