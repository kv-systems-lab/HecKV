# HecKV: three-node KV cache protection and recovery

This source snapshot defines **three physically separate roles** for BF16, TP=2 KV protection:

| Role | Contents |
| --- | --- |
| `data0` | D0's rank 0 and rank 1 serialized KV, plus local XOR parity |
| `data1` | D1's rank 0 and rank 1 serialized KV, plus local XOR parity |
| `parity` | GF(256) RS(2,1) parity P for each rank and the committed manifest; no full data shard |

For each TP rank, `P = D0 + 2·D1` in GF(256). A lost data node is reconstructed from the surviving data node and the separate parity node. Both restored ranks must pass full SHA-256 verification before the recovery commit is written. The placement tolerates one missing data role under these conditions; it does not claim simultaneous failures.

## Logical reference implementation

`experiments/kv_ec/three_node.py` supplies a bounded-memory reference protocol for already serialized rank KV files. Its three endpoints are validated against `artifact/three_node_topology.example.json`. The data hosts calculate their local XOR parity; the dedicated parity host reads both data streams to construct P and persists only P and a manifest. Recovery reads the surviving D shard and P from separate endpoints. The reference uses TCP and files; it does not install recovered KV into a live SGLang GPU slot or claim GDR performance.

```text
# Each data host: place rank0.bin, rank1.bin, and identity.json in its own directory.
python -m experiments.kv_ec.three_node init-data --role D0 --root DATA0_DIR --identity IDENTITY_JSON --host-id DATA0_HOST_ID
python -m experiments.kv_ec.three_node init-data --role D1 --root DATA1_DIR --identity IDENTITY_JSON --host-id DATA1_HOST_ID
python -m experiments.kv_ec.three_node serve --root DATA_DIR --bind HOST_IP --port PORT

# Dedicated parity host:
python -m experiments.kv_ec.three_node protect --root PARITY_DIR --data0 DATA0_IP:PORT --data1 DATA1_IP:PORT --topology artifact/three_node_topology.example.json --host-id PARITY_HOST_ID
python -m experiments.kv_ec.three_node serve --root PARITY_DIR --bind PARITY_IP --port PORT

# Fresh target replacing lost D0 (swap roles for D1 failure):
python -m experiments.kv_ec.three_node restore --output NEW_DIR --lost D0 --survivor DATA1_IP:PORT --parity PARITY_IP:PORT --topology artifact/three_node_topology.example.json
```

`identity.json` must specify `model`, `dtype`, `tp`, `layout`, `protected_tokens`, and `chunk_tokens`; the two data nodes must have compatible serialized layouts and equal byte lengths for corresponding ranks. Example host addresses are documentation placeholders. Actual host separation and process isolation must be established by deployment; strings in a manifest cannot prove physical placement. The TCP reference service must run on a trusted network or behind authenticated transport.

Logic checks: `python -m unittest experiments.kv_ec.test_three_node -q`.

## Repository scope

The repository includes the SGLang-derived runtime and the current three-node reference path. It excludes experiment controllers, measured results, model weights, KV snapshots, and site-specific logs. No three-node performance measurement has been run or reported. The license and source-file notices are retained.
