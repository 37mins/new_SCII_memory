# PPO 更新诊断

`PPO_DIAGNOSTICS=1 bash scripts/train_sc2.sh` 为当前固定长度、纯文本 actor
启用观测，默认关闭。有效配置显式记录 `actor.ppo_diagnostics_dir`，产物在
新 run 的 `ppo_diagnostics/`，各 rank 独立文件，已有同名产物拒绝覆盖。

每个 global step 保存一份真实 batch 的 `.pt`：tokens、attention/position、
old/ref log-prob、advantage；JSONL 保存每个 micro-batch 的新 log-prob。
三种 phase：`pre_update` 在零次参数更新时以 train mode 重算整批，`update`
直接观察训练原有 forward，`post_update` 在全部更新后再次测量同一整批。
每条记录包含已发生的 optimizer 更新数、样本偏移、有效 token 数和最差 token。

K1 是 masked mean(old-new)。K3 调用当前 VERL `kl_penalty`，沿用指数边界
与10上限，字段明确叫 `*_k3_capped_mean`；不是完整词表精确KL。另记录绝对
log-prob 差、ratio越过上下界比例、实际损失裁剪比例。统计只使用有效token，
不回写loss、mask或probability，不调整优化器、学习率、KL系数或cache。

额外整批forward使用 `torch.no_grad()` 和 `torch.random.fork_rng()` 保存/
恢复CPU及当前CUDA随机状态；不切换模型train/eval模式。只支持当前固定
text actor，启用时对dynamic batch/multimodal/loss_mask明确拒绝。增添两遍
teacher-forced forward和日志I/O，因此用短测验证，不默认加到长训。

依据：[PPO原始多次小批次更新说明](https://spinningup.openai.com/en/latest/algorithms/ppo.html)、
本仓库冻结的 `compute_policy_loss`/`kl_penalty`/`append_to_dict`，以及
[PyTorch RNG保护接口](https://docs.pytorch.org/docs/2.7/random.html#torch.random.fork_rng)。
这不是新的训练算法，也没有擅自加入KL early-stop阈值。

```bash
python3 -B -m unittest discover -s tests -p test_ppo_diagnostics.py -v
```

回归执行真实 `update_policy()`；使用带随机扰动的小模型验证诊断开关前后
RNG、每次loss/梯度、最终权重、原指标完全相同。模型仅为CPU测试替身，
不能替代8卡FSDP和4B模型验证。第一次真实模型更新前必须逐rank核对起点
差异，并与同一次实际训练forward一致性比较；否则不批准长训。

`old_log_probs` 是当前HF路径的eval重算值，不是generate输出的采样后log-prob。
本诊断可区分eval重算/训练前向和更新漂移，但不单独证明top-p采样分布与
重算分布等价，也不能把其通过说成所有rollout一致性问题已排除。

## Integrated Fixes v1 上的生成端观测

新增显式开关 `ROLLOUT_LOGPROB_DIAGNOSTICS=1`，默认关闭，与PPO诊断一起
使用。调用HF现有 `output_logits=True / output_scores=True`，不更改生成参数。
按实际生成token记录两种log-prob：raw logits经同一temperature归一化的
`rollout_unwarped_log_probs`，以及包含top-p等处理的`rollout_sampling_log_probs`。
仅保存选中token的[B,T]张量；完整词表logits/scores仅在单次generate内暂存，
约每条96-token回答额外116MB（FP32、约151936词表、两套输出）。不额外生成
回答，不关闭KV cache，不改变loss；诊断字段只随已有batch流转和保存。

新增JSON字段区分两种generate概率与eval重算的差异。首项用于定位生成/重算
路径差异；第二项包含采样截断带来的正常重归一化，不能要求其为零。PPO
pre-update仍比较eval与train；后续逐次更新定位真实参数漂移。BF16前向及
temperature除法的精度顺序也可能带来误差，不能拿FP32小模型阈值代替4B模型
容差；报告均值、最大值和最差token，未解释的差异不标通过。

依据为[Transformers官方生成接口](https://huggingface.co/docs/transformers/main/main_classes/text_generation)
及固定4.55.2版本的`generation/utils.py`，其中logits未经过processor而scores
已经过处理。测试实际调用小型Qwen的HF rollout：开关前后所有原batch字段及
随机状态完全相同，raw概率与整段前向一致，采样概率与raw的差异单独验证；
另验证EOS之后的padding为零、原logits未被修改及诊断开关不影响真实actor更新。

## 温度归一化精度候选

固定文本、无remove-padding、非fused actor支持显式
`actor.fp32_normalization=true`，默认关闭。仅在取出response logits后先转FP32，
再除原temperature，随后调用现有log-prob和entropy函数。loss公式、温度值和
生成cache不变。其他forward分支不在此候选支持范围；生产启动器没有默认启用。
依据是[PyTorch AMP对in-place运算的说明](https://docs.pytorch.org/docs/2.7/amp.html)：
原BF16 `div_` 不会自动提高精度，随后转FP32无法挽回已经发生的舍入。
CPU回归检查真实forward的概率、entropy及梯度与显式公式一致；GPU固定token
A/B另验证真实FSDP与CE内核，不能用CPU替身结果代替真实模型验收。

2026-09-06固定token A/B中，温度精度误差最大降到2.81e-6，同32回答的
4次optimizer更新后最大漂移13.50→0.229。见
[完整实验记录](../../experiments/records/20260906-qwen-kl-metric-v1/README.md#温度归一化精度-abcompleted用户追加授权)。
这不放行长训：cached/full仍有BF16前向残差，独立reference worker与完整
更新序列未验收。`--hf-full-precision` 和 `--observe-fp32-head` 仅是该record
下零更新诊断入口，后者只作额外计算，不替换模型输出。

`install_qwen3_rowwise_rms_norm(model)` 是已否决的独立诊断候选，仅替换Qwen3RMSNorm
归约实现为已有FlashAttention 2.6.3 kernel；FP32归一化、输入dtype转换、
最后乘weight的次序不变。模型参数对象和checkpoint不变，生成cache不变。
不匹配Qwen模块时报错。当前只有固定token诊断通过 `--rowwise-rms-norm`
显式调用，默认训练不启用；其GPU结果以同一record为准，不把CPU替身当作
真实Triton kernel验证。实际kernel回归入口为
`python3 -B tests/test_qwen_rowwise_norm.py --gpu`。

2026-09-06真实GPU回归通过，但cached/full最坏误差0.8088→0.8197，没有解决
残差，不用于生产。后续 `--model-fp32` 在共享Qwen forward上使用官方AMP禁用
上下文，覆盖Actor和HF生成，仅供诊断；它将cached/full最大差降至0.000232，
却使4次更新后最大漂移0.229→112.03，亦被否决。两者都没有关闭生成cache。
完整结果见同一experiment record，不因零更新改善而提前放行训练。
