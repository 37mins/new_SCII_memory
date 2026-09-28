"""CPU-only watcher tests; no SSH, NVIDIA calls, model, or game."""
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

PATH = Path(__file__).resolve().parents[1]/'experiments/records/20260914-zero-task-signal-sampling-v1/wait_for_pair_node46.py'
spec = importlib.util.spec_from_file_location('wait_for_pair', PATH)
watcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(watcher)


class WaitForPairTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root/'deadline_unix.txt').write_text('100')
        self.now = 0
        self.start_patch(patch.object(watcher.time, 'time', side_effect=lambda: self.now))
        self.start_patch(patch.object(watcher.time, 'sleep', side_effect=self.advance))
        self.verify = self.start_patch(patch.object(watcher, 'verify_runtime'))
        self.launch = self.start_patch(patch.object(watcher, 'launch', return_value=0))

    def start_patch(self, p):
        result = p.start()
        self.addCleanup(p.stop)
        return result

    def advance(self, seconds):
        self.now += seconds

    def test_busy_then_free_launches_once_with_original_deadline(self):
        with patch.object(watcher, 'probe', side_effect=[dict(ready=False, gpu_pids=['123']), dict(ready=True)]):
            self.assertEqual(watcher.watch(self.root), 0)
        self.launch.assert_called_once()
        claim = json.loads((self.root/'auto-launch-claim.json').read_text())
        self.assertEqual(claim['deadline_unix'], 100)
        self.assertEqual((self.root/'deadline_unix.txt').read_text(), '100')

    def test_launch_failure_never_retries(self):
        self.launch.return_value = 7
        with patch.object(watcher, 'probe', return_value=dict(ready=True)):
            self.assertEqual(watcher.watch(self.root), 7)
        self.launch.assert_called_once()

    def test_busy_or_probe_error_expires_without_launch(self):
        for error in (False, True):
            with self.subTest(error=error), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp); (root/'deadline_unix.txt').write_text(str(self.now + 35))
                with patch.object(watcher, 'probe', return_value=dict(ready=False),
                                  side_effect=RuntimeError('probe failed') if error else None):
                    self.assertEqual(watcher.watch(root), 124)
        self.launch.assert_not_called()

    def test_existing_attempt_or_pair_prevents_duplicate_launch(self):
        for name in ('auto-launch-claim.json', 'pair-r2'):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp); (root/'deadline_unix.txt').write_text('100'); (root/name).touch()
                self.assertEqual(watcher.watch(root), 0)
        self.launch.assert_not_called()

    def test_frozen_source_check_failure_stops_before_gpu_probe(self):
        self.verify.side_effect = RuntimeError('dirty')
        with patch.object(watcher, 'probe') as probe, self.assertRaisesRegex(RuntimeError, 'dirty'):
            watcher.watch(self.root)
        probe.assert_not_called(); self.launch.assert_not_called()

    def test_probe_only_targets_authorized_cards(self):
        memory = 'MemAvailable: 800000000 kB\n'
        for pids, ray_pids, expected in [('', '', True), ('123', '', False), ('', '456\n', False)]:
            with patch.object(watcher.subprocess, 'check_output', side_effect=[pids, '0, 4, 0\n1, 4, 0\n2, 4, 0\n3, 4, 0']) as query, \
                 patch.object(watcher.Path, 'read_text', return_value=memory), \
                 patch.object(watcher.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0 if ray_pids else 1, ray_pids)):
                self.assertEqual(watcher.probe()['ready'], expected)
                for call in query.call_args_list:
                    self.assertEqual(call.args[0][1:3], ['-i', '0,1,2,3'])


if __name__ == '__main__':
    unittest.main()
