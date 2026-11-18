"""Loopback integration checks for independent D0, D1 and P placement."""

import json
from pathlib import Path
import tempfile
import threading
import unittest

from .three_node import _Handler, _Server, init_data, protect, restore, validate_topology


class ThreeNodeTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.identity = {"model": "fixture", "dtype": "bf16", "tp": 2,
                         "layout": "rank-major-kv-v1", "protected_tokens": 1024,
                         "chunk_tokens": 256}
        self.topology = {"placement": {"D0": "data0", "D1": "data1", "P": "parity"},
                         "rs": {"data": 2, "parity": 1},
                         "hosts": {"data0": "host-a", "data1": "host-b", "parity": "host-c"}}
        self.servers = []
        for node, role in (("data0", "D0"), ("data1", "D1")):
            directory = self.root / node
            directory.mkdir()
            for rank in range(2):
                raw = bytes(((i * (rank + 1) + (17 if role == "D0" else 91)) % 256)
                            for i in range(1025))
                (directory / f"rank{rank}.bin").write_bytes(raw)
            init_data(directory, role, [directory / "rank0.bin", directory / "rank1.bin"],
                      self.identity, self.topology["hosts"][node])
        self.data0 = self._serve(self.root / "data0")
        self.data1 = self._serve(self.root / "data1")
        for node in ("data0", "data1"):
            directory = self.root / node
            rank0 = (directory / "rank0.bin").read_bytes()
            rank1 = (directory / "rank1.bin").read_bytes()
            self.assertEqual((directory / "local-xor.bin").read_bytes(),
                             bytes(a ^ b for a, b in zip(rank0, rank1)))

    def tearDown(self):
        for server in self.servers:
            server.shutdown()
            server.server_close()
        self.temp.cleanup()

    def _serve(self, root):
        server = _Server(("127.0.0.1", 0), _Handler)
        server.root = root
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.servers.append(server)
        return server.server_address

    def test_two_data_nodes_one_independent_parity_node(self):
        parity_dir = self.root / "parity"
        manifest = protect(parity_dir, self.data0, self.data1, self.topology, "host-c")
        self.assertEqual(manifest["placement"], {"D0": "data0", "D1": "data1", "P": "parity"})
        self.assertEqual({p.name for p in parity_dir.iterdir()},
                         {"manifest.json", "parity-rank0.bin", "parity-rank1.bin"})
        parity = self._serve(parity_dir)
        for lost, survivor in (("D0", self.data1), ("D1", self.data0)):
            output = self.root / f"restored-{lost}"
            result = restore(output, lost, survivor, parity, self.topology)
            self.assertEqual(len(result), 2)
            self.assertEqual(json.loads((output / "recovery-commit.json").read_text())["state"],
                             "COMMITTED")
            original = self.root / ("data0" if lost == "D0" else "data1")
            for rank in range(2):
                self.assertEqual((output / f"rank{rank}.bin").read_bytes(),
                                 (original / f"rank{rank}.bin").read_bytes())

    def test_rejects_layout_mismatch(self):
        right = self.root / "data1" / "manifest.json"
        data = json.loads(right.read_text())
        data["identity"]["layout"] = "incompatible"
        right.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "layout"):
            protect(self.root / "rejected", self.data0, self.data1, self.topology, "host-c")

    def test_rejects_colocated_parity(self):
        self.topology["hosts"]["parity"] = self.topology["hosts"]["data1"]
        with self.assertRaisesRegex(ValueError, "distinct hosts"):
            validate_topology(self.topology)

    def test_rejects_wrong_topology_at_restore(self):
        parity_dir = self.root / "parity"
        protect(parity_dir, self.data0, self.data1, self.topology, "host-c")
        parity = self._serve(parity_dir)
        changed = json.loads(json.dumps(self.topology))
        changed["hosts"]["parity"] = "different-host"
        with self.assertRaisesRegex(ValueError, "mismatched three-node"):
            restore(self.root / "rejected", "D0", self.data1, parity, changed)

    def test_corrupt_parity_cannot_commit_recovery(self):
        parity_dir = self.root / "parity"
        protect(parity_dir, self.data0, self.data1, self.topology, "host-c")
        block = parity_dir / "parity-rank0.bin"
        raw = bytearray(block.read_bytes())
        raw[0] ^= 1
        block.write_bytes(raw)
        parity = self._serve(parity_dir)
        output = self.root / "corrupt-restore"
        with self.assertRaisesRegex(ValueError, "served block differs"):
            restore(output, "D0", self.data1, parity, self.topology)
        self.assertFalse((output / "recovery-commit.json").exists())

    def test_rejects_wrong_data_host_identity(self):
        right = self.root / "data1" / "manifest.json"
        data = json.loads(right.read_text())
        data["host_id"] = "host-a"
        right.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "data host identity"):
            protect(self.root / "rejected", self.data0, self.data1, self.topology, "host-c")

    def test_rejects_wrong_parity_host_identity(self):
        with self.assertRaisesRegex(ValueError, "parity host identity"):
            protect(self.root / "rejected", self.data0, self.data1, self.topology, "host-b")


if __name__ == "__main__":
    unittest.main()
