# HecKV
﻿
**HecKV: Hierarchical Erasure-Coded KV Cache Protection and Recovery for Distributed LLM Inference**
﻿
HecKV is a research prototype for reliable KV cache protection and recovery in distributed large language model (LLM) inference.
﻿
It combines **intra-node XOR coding** for GPU-level KV protection with **inter-node Reed–Solomon (RS) coding** for node-level recovery. HecKV further coordinates KV reconstruction with inference-state restoration to support decoding resumption without redundant prefix recomputation.
﻿
This repository contains an SGLang-based implementation and a three-node reference implementation for erasure-coded KV protection and recovery.
﻿
## License
﻿
This project is licensed under the Apache License 2.0 License.
