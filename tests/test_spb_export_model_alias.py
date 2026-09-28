"""CPU regression for Transformers 4.55's dotted local module directory."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


@unittest.skipUnless(importlib.util.find_spec('transformers'), 'Pinned Transformers required')
class ModelAliasTest(unittest.TestCase):
    def test_neutral_directory_alias_preserves_relative_imports(self):
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)/'SpikingBrain-2.0-instruct'
            source.mkdir()
            (source/'config.json').write_text(json.dumps({
                'model_type': 'export_path_probe',
                'auto_map': {'AutoConfig': 'configuration_probe.ProbeConfig',
                             'AutoModelForCausalLM': 'modeling_probe.ProbeModel'}}))
            (source/'configuration_probe.py').write_text(
                'from transformers import PretrainedConfig\n'
                'class ProbeConfig(PretrainedConfig):\n'
                '    model_type = "export_path_probe"\n')
            (source/'layer.py').write_text('from torch.nn import Linear\n')
            (source/'modeling_probe.py').write_text(
                'from transformers import PreTrainedModel\n'
                'from .configuration_probe import ProbeConfig\n'
                'from .layer import Linear\n'
                'class ProbeModel(PreTrainedModel):\n'
                '    config_class = ProbeConfig\n'
                '    def __init__(self, config):\n'
                '        super().__init__(config)\n'
                '        self.layer = Linear(2, 2)\n')
            config = AutoConfig.from_pretrained(source, trust_remote_code=True,
                                                local_files_only=True)
            with self.assertRaises(ModuleNotFoundError):
                AutoModelForCausalLM.from_config(config, trust_remote_code=True)
            alias = Path(directory)/'model_source'
            alias.symlink_to(source, target_is_directory=True)
            config = AutoConfig.from_pretrained(alias, trust_remote_code=True,
                                                local_files_only=True)
            model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
            self.assertEqual(tuple(model.layer.weight.shape), (2, 2))
            self.assertEqual(alias.resolve(), source)
            self.assertFalse(torch.cuda.is_initialized())


if __name__ == '__main__':
    unittest.main()
