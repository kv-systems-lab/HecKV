"""Three-host hierarchical placement for serialized TP=2 KV shards.

D0 and D1 stay on separate data hosts, each with local rank XOR. The third host stores only RS parity and a
manifest. This bounded-memory TCP reference path does not claim GPU KV installation or GDR.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import socketserver
import tempfile

import numpy as np

from .multinode.rs_codec import encode, recover

BLOCK_BYTES = 8 * 1024 * 1024
ROLES = ("D0", "D1")
PLACEMENT = {"D0": "data0", "D1": "data1", "P": "parity"}


def validate_topology(topology):
    """Require three distinct declared hosts and one RS(2,1) stripe."""
    if topology.get("placement") != PLACEMENT or topology.get("rs") != {"data": 2, "parity": 1}:
        raise ValueError("expected D0/data0, D1/data1, P/parity RS(2,1) placement")
    hosts = topology.get("hosts")
    if not isinstance(hosts, dict) or set(hosts) != {"data0", "data1", "parity"}:
        raise ValueError("exactly three host identities are required")
    if any(not isinstance(host, str) or not host.strip() for host in hosts.values()):
        raise ValueError("host identities must be nonempty strings")
    if len(set(hosts.values())) != 3:
        raise ValueError("data and parity roles must be on distinct hosts")
    return topology


def _digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(BLOCK_BYTES), b""):
            h.update(block)
    return h.hexdigest()


def _atomic_json(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.rename(path)


def init_data(root, role, paths, identity, host_id):
    """Register two existing serialized rank files on one data host."""
    if role not in ROLES or len(paths) != 2:
        raise ValueError("one data role and exactly two TP rank files required")
    if not isinstance(host_id, str) or not host_id.strip():
        raise ValueError("data host identity is required")
    required = ("model", "dtype", "tp", "layout", "protected_tokens", "chunk_tokens")
    if any(key not in identity for key in required):
        raise ValueError("incomplete serialized KV identity")
    if identity["tp"] != 2 or identity["dtype"] != "bf16":
        raise ValueError("this artifact requires BF16 TP=2")
    if identity["protected_tokens"] <= 0 or identity["chunk_tokens"] <= 0:
        raise ValueError("invalid protection boundary")
    if identity["protected_tokens"] % identity["chunk_tokens"]:
        raise ValueError("protection boundary is not committed on a chunk boundary")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    blocks = []
    rank_paths = [Path(src).resolve() for src in paths]
    for rank, src in enumerate(rank_paths):
        if src.parent != root.resolve() or src.name != f"rank{rank}.bin":
            raise ValueError("rank files must be root/rank0.bin and root/rank1.bin")
        if not src.is_file() or src.stat().st_size == 0:
            raise ValueError(f"missing or empty rank {rank}")
        blocks.append({"rank": rank, "bytes": src.stat().st_size, "sha256": _digest(src)})
    if blocks[0]["bytes"] != blocks[1]["bytes"]:
        raise ValueError("local TP rank XOR requires equal serialized lengths")
    parity_path = root / "local-xor.bin"
    xor_hash = hashlib.sha256()
    source_hashes = (hashlib.sha256(), hashlib.sha256())
    with parity_path.open("xb") as output, rank_paths[0].open("rb") as first, rank_paths[1].open("rb") as second:
        for offset in range(0, blocks[0]["bytes"], BLOCK_BYTES):
            size = min(BLOCK_BYTES, blocks[0]["bytes"] - offset)
            left, right = _read_exact(first, size), _read_exact(second, size)
            source_hashes[0].update(left)
            source_hashes[1].update(right)
            raw = np.bitwise_xor(np.frombuffer(left, dtype=np.uint8),
                                 np.frombuffer(right, dtype=np.uint8)).tobytes()
            output.write(raw)
            xor_hash.update(raw)
        output.flush()
        os.fsync(output.fileno())
    if any(digest.hexdigest() != block["sha256"] for digest, block in zip(source_hashes, blocks)):
        parity_path.unlink()
        raise ValueError("rank source changed while computing local XOR")
    manifest = {"format": "heckv-three-node-data-v1", "role": role,
                "node": "data0" if role == "D0" else "data1",
                "host_id": host_id,
                "identity": identity, "blocks": blocks,
                "local_xor": {"bytes": blocks[0]["bytes"], "sha256": xor_hash.hexdigest()}}
    _atomic_json(root / "manifest.json", manifest)
    return manifest


class _Handler(socketserver.StreamRequestHandler):
    def handle(self):
        request = json.loads(self.rfile.readline())
        root = self.server.root
        manifest = json.loads((root / "manifest.json").read_text())
        if request == {"op": "manifest"}:
            raw = json.dumps(manifest, sort_keys=True).encode()
            self.wfile.write(json.dumps({"bytes": len(raw)}).encode() + b"\n")
            self.wfile.write(raw)
            return
        rank = request.get("rank")
        if request.get("op") != "block" or type(rank) is not int or rank not in (0, 1):
            raise ValueError("invalid block request")
        if manifest["format"] == "heckv-three-node-parity-v1":
            block = manifest["parity"][rank]
            path = root / f"parity-rank{rank}.bin"
        else:
            block = manifest["blocks"][rank]
            path = root / f"rank{rank}.bin"
        if path.stat().st_size != block["bytes"] or _digest(path) != block["sha256"]:
            self.wfile.write(b'{"error":"served block differs from committed manifest"}\n')
            return
        self.wfile.write(json.dumps(block).encode() + b"\n")
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(BLOCK_BYTES), b""):
                self.wfile.write(chunk)


class _Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def serve(root, host, port):
    root = Path(root).resolve()
    if not (root / "manifest.json").is_file():
        raise FileNotFoundError(root / "manifest.json")
    with _Server((host, port), _Handler) as server:
        server.root = root
        server.serve_forever()


def _connect(endpoint, request):
    sock = socket.create_connection(endpoint, timeout=30)
    stream = None
    try:
        sock.settimeout(60)
        stream = sock.makefile("rb")
        sock.sendall(json.dumps(request).encode() + b"\n")
        header = json.loads(stream.readline())
        if "error" in header:
            raise ValueError(header["error"])
        return sock, stream, header
    except BaseException:
        if stream is not None:
            stream.close()
        sock.close()
        raise


def _read_exact(stream, length):
    result = stream.read(length)
    if len(result) != length:
        raise EOFError("truncated remote KV block")
    return result


def fetch_manifest(endpoint):
    sock, stream, header = _connect(endpoint, {"op": "manifest"})
    try:
        return json.loads(_read_exact(stream, header["bytes"]))
    finally:
        stream.close()
        sock.close()


def _compatible(left, right):
    if left["format"] != "heckv-three-node-data-v1" or right["format"] != left["format"]:
        raise ValueError("unsupported data manifest")
    if ((left["role"], right["role"]) != ROLES or
            left.get("node") != "data0" or right.get("node") != "data1"):
        raise ValueError("expected independent D0 and D1 data hosts")
    if left["identity"] != right["identity"]:
        raise ValueError("model, rank layout, or protection boundary mismatch")
    if len(left["blocks"]) != 2 or len(right["blocks"]) != 2:
        raise ValueError("both TP ranks are required")
    for rank in range(2):
        if left["blocks"][rank]["rank"] != rank or right["blocks"][rank]["rank"] != rank:
            raise ValueError("TP rank order mismatch")
        if left["blocks"][rank]["bytes"] != right["blocks"][rank]["bytes"]:
            raise ValueError("RS inputs require identical serialized lengths per rank")


def _block_stream(endpoint, rank, expected):
    sock, stream, header = _connect(endpoint, {"op": "block", "rank": rank})
    if header != expected:
        stream.close()
        sock.close()
        raise ValueError("remote block manifest mismatch")
    return sock, stream


def protect(parity_root, data0, data1, topology, parity_host_id):
    """Run on the dedicated parity host; no D0/D1 files persist there."""
    validate_topology(topology)
    if parity_host_id != topology["hosts"]["parity"]:
        raise ValueError("parity host identity does not match topology")
    if data0 == data1:
        raise ValueError("D0 and D1 require different data endpoints")
    left, right = fetch_manifest(data0), fetch_manifest(data1)
    _compatible(left, right)
    if (left.get("host_id") != topology["hosts"]["data0"] or
            right.get("host_id") != topology["hosts"]["data1"]):
        raise ValueError("data host identity does not match topology")
    root = Path(parity_root)
    root.mkdir(parents=True, exist_ok=True)
    if (root / "manifest.json").exists():
        raise FileExistsError(root / "manifest.json")
    parity = []
    for rank in range(2):
        sockets = []
        temporary = root / f"parity-rank{rank}.partial"
        try:
            for endpoint, expected in ((data0, left["blocks"][rank]), (data1, right["blocks"][rank])):
                sockets.append(_block_stream(endpoint, rank, expected))
            digests = [hashlib.sha256(), hashlib.sha256()]
            parity_hash = hashlib.sha256()
            length = left["blocks"][rank]["bytes"]
            with temporary.open("xb") as output:
                for offset in range(0, length, BLOCK_BYTES):
                    size = min(BLOCK_BYTES, length - offset)
                    chunks = [_read_exact(stream, size) for _, stream in sockets]
                    for digest, chunk in zip(digests, chunks):
                        digest.update(chunk)
                    encoded = encode(*(np.frombuffer(chunk, dtype=np.uint8) for chunk in chunks))[2]
                    raw = encoded.tobytes()
                    output.write(raw)
                    parity_hash.update(raw)
                output.flush()
                os.fsync(output.fileno())
            for source, digest in zip((left, right), digests):
                if digest.hexdigest() != source["blocks"][rank]["sha256"]:
                    raise ValueError("source changed during parity creation")
            temporary.rename(root / f"parity-rank{rank}.bin")
            parity.append({"rank": rank, "bytes": length, "sha256": parity_hash.hexdigest()})
        finally:
            for sock, stream in sockets:
                stream.close()
                sock.close()
            temporary.unlink(missing_ok=True)
    manifest = {"format": "heckv-three-node-parity-v1", "node": "parity",
                "host_id": parity_host_id,
                "placement": PLACEMENT, "topology": topology,
                "codec": "GF(256) RS(2,1), P=D0+2D1", "identity": left["identity"],
                "sources": {"D0": left, "D1": right}, "parity": parity,
                "state": "COMMITTED"}
    _atomic_json(root / "manifest.json", manifest)
    return manifest


def restore(output_root, lost_role, survivor, parity_node, topology):
    """Run on a fresh target; fetch one survivor and P from different hosts."""
    validate_topology(topology)
    if lost_role not in ROLES:
        raise ValueError("only one erased data node can be recovered")
    if survivor == parity_node:
        raise ValueError("survivor and parity require different endpoints")
    committed = fetch_manifest(parity_node)
    surviving_role = "D1" if lost_role == "D0" else "D0"
    source = fetch_manifest(survivor)
    if (committed.get("state") != "COMMITTED" or committed.get("node") != "parity" or
            committed.get("host_id") != topology["hosts"]["parity"] or
            committed.get("placement") != PLACEMENT or committed.get("topology") != topology or
            committed["sources"][surviving_role] != source or
            source["role"] != surviving_role or
            source.get("node") != PLACEMENT[surviving_role] or
            source.get("host_id") != topology["hosts"][PLACEMENT[surviving_role]]):
        raise ValueError("uncommitted or mismatched three-node protection inputs")
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=False)
    results = []
    for rank in range(2):
        expected = committed["sources"][lost_role]["blocks"][rank]
        survivor_meta = source["blocks"][rank]
        parity_meta = committed["parity"][rank]
        if survivor_meta["bytes"] != parity_meta["bytes"] or expected["bytes"] != parity_meta["bytes"]:
            raise ValueError("incompatible rank byte lengths")
        sockets = []
        path = root / f"rank{rank}.bin"
        try:
            sockets.append(_block_stream(survivor, rank, survivor_meta))
            sockets.append(_block_stream(parity_node, rank, parity_meta))
            survivor_hash, parity_hash, recovered_hash = (hashlib.sha256() for _ in range(3))
            with path.open("xb") as output:
                for offset in range(0, expected["bytes"], BLOCK_BYTES):
                    size = min(BLOCK_BYTES, expected["bytes"] - offset)
                    surviving = _read_exact(sockets[0][1], size)
                    parity = _read_exact(sockets[1][1], size)
                    survivor_hash.update(surviving)
                    parity_hash.update(parity)
                    arr = np.frombuffer(surviving, dtype=np.uint8)
                    p = np.frombuffer(parity, dtype=np.uint8)
                    d0, d1 = recover(data0=None if lost_role == "D0" else arr,
                                     data1=None if lost_role == "D1" else arr,
                                     parity=p, lengths=(size, size))
                    recovered = (d0 if lost_role == "D0" else d1).tobytes()
                    output.write(recovered)
                    recovered_hash.update(recovered)
                output.flush()
                os.fsync(output.fileno())
            if (survivor_hash.hexdigest() != survivor_meta["sha256"] or
                    parity_hash.hexdigest() != parity_meta["sha256"] or
                    recovered_hash.hexdigest() != expected["sha256"]):
                raise ValueError("full-shard SHA-256 validation failed")
            results.append({"rank": rank, "bytes": expected["bytes"], "sha256": recovered_hash.hexdigest()})
        finally:
            for sock, stream in sockets:
                stream.close()
                sock.close()
    _atomic_json(root / "recovery-commit.json", {"state": "COMMITTED", "lost_role": lost_role,
                                                "identity": committed["identity"], "ranks": results})
    return results


def _endpoint(value):
    host, port = value.rsplit(":", 1)
    return host, int(port)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init-data")
    init.add_argument("--root", type=Path, required=True)
    init.add_argument("--role", choices=ROLES, required=True)
    init.add_argument("--identity", type=Path, required=True)
    init.add_argument("--host-id", required=True)
    service = sub.add_parser("serve")
    service.add_argument("--root", type=Path, required=True)
    service.add_argument("--bind", default="127.0.0.1")
    service.add_argument("--port", type=int, required=True)
    create = sub.add_parser("protect")
    create.add_argument("--root", type=Path, required=True)
    create.add_argument("--data0", type=_endpoint, required=True)
    create.add_argument("--data1", type=_endpoint, required=True)
    create.add_argument("--topology", type=Path, required=True)
    create.add_argument("--host-id", required=True)
    recover_cmd = sub.add_parser("restore")
    recover_cmd.add_argument("--output", type=Path, required=True)
    recover_cmd.add_argument("--lost", choices=ROLES, required=True)
    recover_cmd.add_argument("--survivor", type=_endpoint, required=True)
    recover_cmd.add_argument("--parity", type=_endpoint, required=True)
    recover_cmd.add_argument("--topology", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "init-data":
        result = init_data(args.root, args.role,
                           [args.root / f"rank{i}.bin" for i in range(2)],
                           json.loads(args.identity.read_text()), args.host_id)
    elif args.command == "serve":
        return serve(args.root, args.bind, args.port)
    elif args.command == "protect":
        result = protect(args.root, args.data0, args.data1,
                         json.loads(args.topology.read_text()), args.host_id)
    else:
        result = restore(args.output, args.lost, args.survivor, args.parity,
                         json.loads(args.topology.read_text()))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
