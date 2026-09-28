"""Real CPU logits/generation/actor probes; no GPU or SC2 games."""
import os
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import torch
from omegaconf import OmegaConf
from transformers import GPT2Config, GPT2LMHeadModel, LogitsProcessorList
from verl.utils.sc2_no_thinking import NoThinkingConstraint, NoThinkingLogitsProcessor, load_constraint, non_thinking_prompt


PIECES = {0:"", 1:"<think>", 2:"</think>", 3:"<", 4:"thi", 5:"nk>",
          6:"marine", 7:"</answer>", 8:"<answer>", 9:"/"}


class TorchNoThinkingTest(unittest.TestCase):
    def setUp(self): self.c=NoThinkingConstraint(PIECES)

    def test_generation_teacher_masks_and_normalization_match_all_positions(self):
        labels=torch.tensor([[8,6,7,0],[3,4,6,0]])
        for dtype in (torch.float32,torch.bfloat16):
            raw=torch.randn(2,4,10,dtype=dtype)
            teacher=self.c.mask_teacher(raw.clone(),labels)
            for b in range(2):
                for t in range(4):
                    gen=self.c.mask_generation(raw[b,t].clone()[None,:],[labels[b,:t].tolist()])
                    torch.testing.assert_close(gen.softmax(-1)[0],teacher[b,t].softmax(-1))
                    self.assertTrue(torch.isfinite(teacher[b,t].log_softmax(-1)[labels[b,t]]))

    def test_entropy_backward_is_finite_and_banned_gradient_zero(self):
        raw=torch.randn(2,4,10,requires_grad=True)
        labels=torch.tensor([[8,6,7,0],[3,4,6,0]])
        logits=self.c.mask_teacher(raw.clone(),labels)
        pd=logits.softmax(-1)
        entropy=logits.logsumexp(-1)-(pd*logits).sum(-1)
        lp=logits.log_softmax(-1).gather(-1,labels[...,None]).squeeze(-1)
        (-lp.mean()-0.01*entropy.mean()).backward()
        self.assertTrue(torch.isfinite(entropy).all())
        self.assertTrue(torch.isfinite(raw.grad).all())
        self.assertEqual(raw.grad[:,:,1:3].abs().sum().item(),0)
        self.assertEqual(raw.grad[1,2,5].item(),0)

    def test_prompt_prefix_is_not_treated_as_generated_response(self):
        ids=torch.tensor([[1,2,3,4]])
        masked=NoThinkingLogitsProcessor(self.c,2)(ids,torch.zeros(1,10))
        self.assertTrue(torch.isneginf(masked[0,5]))

    def test_real_HF_CPU_generate_forbidden_preference_and_split_spelling(self):
        model=GPT2LMHeadModel(GPT2Config(vocab_size=10,n_positions=32,n_embd=8,
            n_layer=1,n_head=1,bos_token_id=0,eos_token_id=7,pad_token_id=0)).eval()
        c=self.c
        class Prefer:
            def __call__(self,input_ids,scores):
                scores.fill_(-100)
                step=input_ids.shape[1]-2
                desired=[1,3,4,5,2,7][step]
                scores[:,desired]=100
                scores[:,6]=50
                return scores
        output=model.generate(torch.tensor([[1,2]]),attention_mask=torch.ones(1,2,dtype=torch.long),
            max_new_tokens=6,do_sample=True,temperature=0.8,top_p=0.9,top_k=0,
            logits_processor=LogitsProcessorList([Prefer(),NoThinkingLogitsProcessor(c,2)]))
        generated=output[0,2:].tolist()
        self.assertEqual(generated,[6,3,4,6,6,7])
        self.assertNotIn("<think","".join(PIECES[t] for t in generated))

    def test_actual_actor_and_reference_forward_use_masked_support(self):
        import verl.workers.actor.dp_actor as module
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.scores=torch.nn.Parameter(torch.randn(1,6,10))
            def forward(self,**kw): return NS(logits=self.scores*1.0)
        config=OmegaConf.create(dict(ulysses_sequence_parallel_size=1,use_torch_compile=False,
                                     fp32_normalization=True))
        actor=module.DataParallelPPOActor(config,Model())
        ref=module.DataParallelPPOActor(config,Model())
        ref.actor_module.load_state_dict(actor.actor_module.state_dict())
        actor.device_name=ref.device_name="cpu"
        actor.no_thinking=ref.no_thinking=self.c
        data=dict(input_ids=torch.tensor([[0,0,3,4,6,7]]),responses=torch.tensor([[3,4,6,7]]),
                  attention_mask=torch.ones(1,6),position_ids=torch.arange(6)[None,:])
        # Substitute only the CUDA flash-CE kernel with equivalent CPU math.
        cpu_logprobs=lambda logits,labels:logits.log_softmax(-1).gather(-1,labels[...,None]).squeeze(-1)
        with patch.object(module,"logprobs_from_logits",cpu_logprobs):
            entropy,lp=actor._forward_micro_batch(data,0.8,True)
            _,rp=ref._forward_micro_batch(data,0.8,True)
        torch.testing.assert_close(lp,rp)
        torch.testing.assert_close((lp-rp).exp(),torch.ones_like(lp))
        (-lp.mean()-0.01*entropy.mean()).backward()
        self.assertTrue(torch.isfinite(actor.actor_module.scores.grad).all())
        self.assertEqual(actor.actor_module.scores.grad[:,:,1:3].abs().sum().item(),0)

    def test_actual_HF_rollout_passes_constraint_and_preserves_response_slice(self):
        from verl import DataProto
        from tensordict import TensorDict
        from verl.workers.rollout.hf_rollout import HFRollout
        model=GPT2LMHeadModel(GPT2Config(vocab_size=10,n_positions=32,n_embd=8,
            n_layer=1,n_head=1,bos_token_id=0,eos_token_id=7,pad_token_id=0)).eval()
        original=model.generate
        seen=[]
        class Prefer:
            def __call__(self,input_ids,scores):
                scores.fill_(-100);scores[:,1]=100;scores[:,6]=50
                return scores
        def generate(**kwargs):
            seen.append(kwargs.get('logits_processor'))
            kwargs['logits_processor']=LogitsProcessorList([Prefer(),*kwargs.get('logits_processor',[])])
            return original(**kwargs)
        model.generate=generate
        cfg=OmegaConf.create(dict(do_sample=True,temperature=0.8,response_length=3,top_p=0.9,top_k=0,n=1))
        rollout=HFRollout(model,cfg)
        data=DataProto(batch=TensorDict(dict(input_ids=torch.tensor([[1,2]]),
            attention_mask=torch.ones(1,2,dtype=torch.long),position_ids=torch.arange(2)[None,:]),batch_size=1),
            meta_info=dict(eos_token_id=7,pad_token_id=0))
        with patch('torch.autocast',return_value=__import__('contextlib').nullcontext()), \
                patch('verl.workers.rollout.hf_rollout.get_torch_device',
                      return_value=NS(empty_cache=lambda:None)):
            baseline=rollout._generate_minibatch(data)
            rollout.no_thinking=self.c
            constrained=rollout._generate_minibatch(data)
        self.assertIsNone(seen[0]);self.assertIsInstance(seen[1][0],NoThinkingLogitsProcessor)
        self.assertEqual(baseline.batch['responses'].tolist(),[[1,1,1]])
        self.assertEqual(constrained.batch['responses'].tolist(),[[6,6,6]])
        self.assertEqual(constrained.batch['prompts'].tolist(),[[1,2]])
        self.assertFalse(torch.cuda.is_initialized())

    def test_actual_collector_prompt_has_closed_thinking_only_when_enabled(self):
        from agent_system.multi_turn_rollout.rollout_loop import TrajectoryCollector
        from transformers import AutoTokenizer
        path=os.environ['SC2_TEST_TOKENIZER_PATH']
        tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True,trust_remote_code=False)
        cfg=OmegaConf.create(dict(data=dict(sc2_no_thinking=True,max_prompt_length=128,truncation='error'),
            actor_rollout_ref={k:dict(no_thinking_tokenizer_path=path,name='hf') for k in ('actor','ref','rollout')}))
        gen=NS(non_tensor_batch=dict(raw_prompt=[[]],data_source=['sc2']))
        obs=dict(text=['Return only a build order.'])
        collector=TrajectoryCollector(cfg,tokenizer)
        row=collector.preprocess_single_sample(0,gen,obs)
        text=tokenizer.decode(row['input_ids'][row['attention_mask'].bool()],skip_special_tokens=False)
        self.assertTrue(text.endswith('<|im_start|>assistant\n<think>\n\n</think>\n'))
        cfg.data.sc2_no_thinking=False
        for k in ('actor','ref','rollout'):cfg.actor_rollout_ref[k].no_thinking_tokenizer_path=None
        row=TrajectoryCollector(cfg,tokenizer).preprocess_single_sample(0,gen,obs)
        text=tokenizer.decode(row['input_ids'][row['attention_mask'].bool()],skip_special_tokens=False)
        self.assertTrue(text.endswith('<|im_start|>assistant\n'))
        self.assertNotIn('<think>',text)

    def test_deployed_tokenizer_when_path_supplied(self):
        path=os.environ.get("SC2_TEST_TOKENIZER_PATH")
        if not path: self.fail("SC2_TEST_TOKENIZER_PATH required; do not silently skip real tokenizer proof")
        from transformers import AutoTokenizer
        t=AutoTokenizer.from_pretrained(path,local_files_only=True,trust_remote_code=False)
        c=load_constraint(path)
        self.assertIn(151667,c.blocked[""]);self.assertIn(151668,c.blocked[""])
        vocab=t.get_vocab()
        # Byte-by-byte spelling deliberately bypasses the dedicated added tokens.
        for text in ("<think>","</think>","<THINK>","</ThInK>"):
            ids=[vocab[ch] for ch in text]
            state=""
            blocked=False
            for token in ids:
                if token in c.blocked[state]: blocked=True;break
                state=c.advance(state,token)
            self.assertTrue(blocked,text)
        prompt=t.apply_chat_template([dict(role="user",content="Give an answer.")],
            tokenize=False,add_generation_prompt=True,enable_thinking=False)
        self.assertTrue(non_thinking_prompt(prompt).endswith("<think>\n\n</think>\n"))
        print("REAL_TOKENIZER_PROOF",c.proof(),flush=True)
        self.assertFalse(torch.cuda.is_initialized())


if __name__=="__main__": unittest.main()
