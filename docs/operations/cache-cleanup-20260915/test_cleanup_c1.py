import tempfile
import unittest
from pathlib import Path

import cleanup_c1 as cleanup


class CacheCleanupTests(unittest.TestCase):
    def test_exact_allowlist(self):
        manifest = Path(__file__).parent.parent / "storage-cleanup-candidates-20260915.json"
        targets, excluded = cleanup.load_targets(manifest)
        self.assertEqual(len(targets), 84)
        self.assertEqual(len({p.parent for p in targets}), 31)
        self.assertFalse({str(p.parent) for p in targets} & excluded)

    def test_changed_manifest_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text('{"cache": [{"path":"/dev/shm"}]}')
            with self.assertRaises(ValueError):
                cleanup.load_targets(path)

    def test_snapshot_change_and_symlink_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            path = root / "kernel.so"
            path.write_bytes(b"test")
            before = cleanup.snapshot(root)
            path.write_bytes(b"changed")
            self.assertNotEqual(before, cleanup.snapshot(root))
            (root / "link").symlink_to(path)
            with self.assertRaises(ValueError):
                cleanup.snapshot(root)

    def test_model_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            (root / "model.safetensors").touch()
            with self.assertRaises(ValueError):
                cleanup.snapshot(root)
