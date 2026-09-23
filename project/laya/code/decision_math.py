"""Laya 数学机制的零依赖教学实现，不含编码器或预训练权重。

python3 project/laya/code/decision_math.py --self-test
python3 project/laya/code/decision_math.py

所有示例 logits 均为人工输入，不代表模型预测或性能。
"""
from __future__ import annotations

import argparse
import json
import math
import random
import unittest


def softmax(logits, temperature=1.0):
    if not logits or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("需要非空 logits 和有限的正温度")
    scaled = [v / temperature for v in logits]
    offset = max(scaled)
    values = [math.exp(v - offset) for v in scaled]
    total = sum(values)
    return [v / total for v in values]


def entropy_confidence(probs):
    if len(probs) < 2:
        return 1.0
    entropy = -sum(p * math.log(max(p, 1e-12)) for p in probs)
    return max(0.0, min(1.0, 1 - entropy / math.log(len(probs))))


def decode(logits, kind, criteria=None, temperature=1.0):
    """与上游答案字段语义一致；省略 action（需另一个神经网络输出）。"""
    probs = softmax(logits, temperature)
    conf = round(entropy_confidence(probs), 4)
    if kind == "noul":
        if len(probs) != 2:
            raise ValueError("noul 的顺序必须是 [false, true]")
        return {"type": kind, "noul": round(probs[1], 4),
                "confidence": round(max(probs), 4)}
    if kind == "choice":
        keys = list(criteria)
        if len(keys) != len(probs):
            raise ValueError("选项与 logits 不匹配")
        return {"type": kind, "choice": keys[max(range(len(probs)), key=probs.__getitem__)],
                "probabilities": dict(zip(keys, map(lambda p: round(p, 4), probs))),
                "confidence": conf}
    if kind == "score":
        if len(criteria) != len(probs):
            raise ValueError("等级与 logits 不匹配")
        return {"type": kind, "score": round(sum(i * p for i, p in enumerate(probs)), 4),
                "legend": {str(i): c for i, c in enumerate(criteria)},
                "probabilities": {str(i): round(p, 4) for i, p in enumerate(probs)},
                "confidence": conf}
    raise ValueError("kind 必须是 choice、score 或 noul")


def reward(probs, target, kind="choice", w_sph=0.75, w_rps=1.0):
    """复现 proper_reward 的标量数学；默认球面权重采用微调 notebook 的 0.75。"""
    if len(probs) != len(target) or not probs:
        raise ValueError("概率与目标长度必须相同且非空")
    log_score = sum(t * max(math.log(max(p, 1e-12)), -9.21)
                    for p, t in zip(probs, target))
    spherical = sum(p * t for p, t in zip(probs, target)) / max(
        math.sqrt(sum(p * p for p in probs)), 1e-9)
    value = log_score + w_sph * spherical
    if kind == "score":
        cdf_p = cdf_t = rps = 0.0
        for p, t in zip(probs, target):
            cdf_p += p
            cdf_t += t
            rps += (cdf_p - cdf_t) ** 2
        value -= w_rps * rps / (max(2, len(probs)) - 1)
    return value


def rlcd_logit_step(logits, target, kind="choice", sigma=0.4, group=4, lr=0.05, seed=13):
    """在一条人工 logit 向量上演示 RL + soft CE 的解析梯度。

    复现零和噪声、组均值基线和样本标准差；不含 batch/DDP/AdamW，
    不将这一小步称为完整模型训练或泛化实验。
    """
    if group < 2 or sigma <= 0 or len(logits) != len(target):
        raise ValueError("group >= 2、sigma > 0，且向量长度相同")
    rng = random.Random(seed)
    noise, rewards = [], []
    for _ in range(group):
        eps = [rng.gauss(0, sigma) for _ in logits]
        mean = sum(eps) / len(eps)
        eps = [e - mean for e in eps]
        noise.append(eps)
        rewards.append(reward(softmax([z + e for z, e in zip(logits, eps)]), target, kind))
    baseline = sum(rewards) / group
    std = math.sqrt(sum((r - baseline) ** 2 for r in rewards) / (group - 1))
    advantages = [(r - baseline) / (std + 1e-6) for r in rewards]
    # z_sample 被 detach；d log pi(z_sample | logits) / d logits = eps / sigma^2。
    grad_rl = [-sum(a * eps[i] for a, eps in zip(advantages, noise)) / (group * sigma**2)
               for i in range(len(logits))]
    probs = softmax(logits)
    grad_ce = [p - t for p, t in zip(probs, target)]
    updated = [z - lr * (r + c) for z, r, c in zip(logits, grad_rl, grad_ce)]
    return {"before": probs, "after": softmax(updated), "logits": updated,
            "rewards": rewards, "advantages": advantages,
            "noise": noise, "grad_rl": grad_rl, "grad_ce": grad_ce}


class MechanismChecks(unittest.TestCase):
    def test_temperature_preserves_argmax(self):
        for t in [0.5, 1.0, 2.0, 5.0]:
            p = softmax([1001, 999, 1000], t)
            self.assertAlmostEqual(sum(p), 1)
            self.assertEqual(max(range(3), key=p.__getitem__), 0)
        self.assertGreater(entropy_confidence(softmax([3, 0], .5)),
                           entropy_confidence(softmax([3, 0], 5)))

    def test_confidence_semantics(self):
        logits = [math.log(.1), math.log(.9)]
        self.assertAlmostEqual(decode(logits, "noul")["confidence"], .9)
        self.assertAlmostEqual(decode(logits, "choice", ["no", "yes"])["confidence"], .531)

    def test_score_expectation_not_argmax(self):
        ans = decode(list(map(math.log, [.2, .3, .5])), "score", ["low", "mid", "high"])
        self.assertEqual(ans["score"], 1.3)
        self.assertEqual(decode([0], "score", ["only"])["score"], 0)

    def test_rps_penalises_distant_miss(self):
        target = [1, 0, 0]
        near, far = [.1, .8, .1], [.1, .1, .8]
        self.assertAlmostEqual(reward(near, target), reward(far, target))
        self.assertGreater(reward(near, target, "score"), reward(far, target, "score"))

    def test_projected_policy_gradient_finite_difference(self):
        z, target = [.3, -.2, .1], [.2, .6, .2]
        step = rlcd_logit_step(z, target)
        for eps in step["noise"]:
            self.assertAlmostEqual(sum(eps), 0)
        fixed_samples = [[x + e for x, e in zip(z, eps)] for eps in step["noise"]]
        def loss(mu):
            logp = [-sum((v - m)**2 for v, m in zip(sample, mu)) / (2 * .4**2)
                    for sample in fixed_samples]
            rl = -sum(a * p for a, p in zip(step["advantages"], logp)) / 4
            ce = -sum(t * math.log(p) for t, p in zip(target, softmax(mu)))
            return rl + ce
        for i in range(3):
            left, right = z.copy(), z.copy()
            left[i] -= 1e-6
            right[i] += 1e-6
            numerical = (loss(right) - loss(left)) / 2e-6
            self.assertAlmostEqual(numerical, step["grad_rl"][i] + step["grad_ce"][i], places=6)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(MechanismChecks)
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        raise SystemExit(0 if result.wasSuccessful() else 1)
    logits = [math.log(.1), math.log(.9)]
    result = {
        "note": "人工 logits 教学输出；未运行 Laya 权重",
        "same_probs_as_choice": decode(logits, "choice", ["negative", "positive"]),
        "same_probs_as_noul": decode(logits, "noul"),
        "ordinal": decode(list(map(math.log, [.2, .3, .5])), "score", ["low", "mid", "high"]),
        "rlcd_one_step": rlcd_logit_step([0, 0, 0], [.1, .2, .7]),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
