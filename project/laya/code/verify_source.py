"""执行固定上游源码中的纯函数和答案解码，对照博客实现，无需 torch 或权重。

需要 numpy。仅对已核验的 Laya checkout 使用（会执行指定源码的 AST 子集）。
python verify_source.py --source /tmp/laya-research-source
"""
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import random
import types
from typing import Dict, List, Optional, Union, Any

import numpy as np
from decision_math import decode


class ToyTokenizer:
    """字符级 tokenizer 仅用于审计预算控制流，不代表真实 BPE token 数。"""
    mask_token = "[MASK]"
    mask_token_id, cls_token_id, sep_token_id = 1, 2, 3

    def __call__(self, text, **kwargs):
        return {"input_ids": [ord(c) + 10 for c in text]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((Path(__file__).resolve().parents[1] / "data/sources.json").read_text())
    for entry in manifest["files"]:
        digest = hashlib.sha256((args.source / entry["path"]).read_bytes()).hexdigest()
        if digest != entry["sha256"]:
            raise ValueError(f"来源内容已变化：{entry['path']}")

    namespace = dict(json=json, math=math, np=np, Dict=Dict, List=List, Optional=Optional,
                     Union=Union, Any=Any, QTYPES={"choice": 0, "score": 1, "noul": 2},
                     QTYPE_NAMES={0: "choice", 1: "score", 2: "noul"},
                     _DEFAULT_NOUL_LABELS={"false": "false", "true": "true"},
                     TEMP_MIN=.5, TEMP_MAX=5.)
    names = {"serialize_state", "render_criterion", "_resolve_noul_labels", "render_options",
             "build_sequence", "confidence_from_probs", "temp_bucket", "clamp_temperature"}
    common = ast.parse((args.source / "laya/common.py").read_text())
    selected = [n for n in common.body if isinstance(n, ast.FunctionDef) and n.name in names]
    exec(compile(ast.Module(body=selected, type_ignores=[]), "upstream-common", "exec"), namespace)
    agent = ast.parse((args.source / "laya/agent.py").read_text())
    cls = next(n for n in agent.body if isinstance(n, ast.ClassDef) and n.name == "Agent")
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_decode_answers")
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "upstream-agent", "exec"), namespace)
    rng, checked = random.Random(13), 0
    for kind in ["choice", "score", "noul"]:
        for k in ([2] if kind == "noul" else [1, 2, 5, 20]):
            criteria = {f"label{i}": f"description {i}" for i in range(k)} if kind == "choice" else [str(i) for i in range(k)]
            for temp in [.5, 1., 2., 5.]:
                z = [rng.uniform(-5, 5) for _ in range(k)]
                upstream = namespace["_decode_answers"](
                    types.SimpleNamespace(temperature_by_options={}, temperature=[temp]*3),
                    np.array([z]), np.array([[.5, .5]]), [{"markers": list(range(k))}],
                    ["q"], {"q": {"t": kind, "crit": criteria}}, 0)["q"]
                upstream.pop("action")
                assert upstream == decode(z, kind, criteria, temp), (kind, k, temp)
                checked += 1

    # 直接运行上游 build_sequence，观察四 token 下限如何挤占 state。
    tok = ToyTokenizer()
    q = {"t": "choice", "ins": "intent?", "crit": {f"label{i}": "a long option description" for i in range(77)}}
    ids, markers = namespace["build_sequence"](tok, "X" * 1000, q, 512, 192)
    seps = [i for i, v in enumerate(ids) if v == tok.sep_token_id]
    state_tokens = len(ids) - seps[-2] - 2
    option_tokens = seps[-2] - markers[0]
    assert len(markers) == 77 and option_tokens == 308
    assert len(ids) == 512

    # list 默认左截断是在 Agent._encode_state；这里分别测试 build_sequence 的方向参数。
    short_q = {"t": "noul", "ins": "ok?", "crit": {}}
    text = "A" * 300 + "Z" * 300
    left, _ = namespace["build_sequence"](tok, text, short_q, 128, 80, truncate_left=True)
    right, _ = namespace["build_sequence"](tok, text, short_q, 128, 80, truncate_left=False)
    assert left[-2] == ord("Z") + 10 and right[-2] == ord("A") + 10
    assert namespace["clamp_temperature"](.10058280825614929) == .5
    assert namespace["clamp_temperature"](float("nan")) == 1
    print(json.dumps({"source_commit": manifest["commit"], "hashes_verified": len(manifest["files"]),
                      "decode_parity_cases": checked, "budget_probe": {
                          "tokenizer": "toy character tokenizer; not real BPE",
                          "options": len(markers), "declared_head_max_len": 192,
                          "actual_option_block_tokens": option_tokens,
                          "remaining_state_tokens": state_tokens, "total_tokens": len(ids)},
                      "truncation_direction": "passed", "temperature_clamp": "passed",
                      "scope": "真实上游纯函数/解码逻辑；不包括神经网络、权重或 GPU"},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
