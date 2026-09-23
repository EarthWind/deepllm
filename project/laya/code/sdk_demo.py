"""Laya 中文工单示例：默认仅展示请求；--run 才下载权重并执行真实推理。"""
import argparse
import json

STATE = {"body": "本月被重复扣费两次，请今天退回多扣的钱，否则我会取消订阅。"}
QUESTIONS = {
    "department": {
        "type": "choice", "instructions": "Which team should handle this ticket?",
        "criteria": {"billing": "invoices, duplicate charges, refunds",
                     "technical": "bugs, outages, errors", "sales": "pricing, new contracts"},
    },
    "urgency": {
        "type": "score", "instructions": "How urgent is this request?",
        "criteria": ["no time constraint", "needs attention soon", "explicit same-day deadline"],
    },
    "refund_requested": {
        "type": "noul", "instructions": "Does the customer explicitly request a refund?",
        "criteria": {"false": "No refund is requested", "true": "A refund is explicitly requested"},
        "labels": {"false": "B", "true": "A"},
    },
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--local-checkpoint", help="已固定 HF revision 的本地多语言权重目录")
    args = parser.parse_args()
    if not args.run:
        print(json.dumps({"note": "请求预览；不是模型输出。加 --run 执行推理。",
                          "state": STATE, "questions": QUESTIONS}, ensure_ascii=False, indent=2))
        return
    import laya
    if args.local_checkpoint:
        with laya.load(args.local_checkpoint, device=args.device) as agent:
            result = agent.predict(STATE, QUESTIONS)
    else:
        # 仅按需加载多语言权重；不预加载三个模型。
        with laya.Router(default="multilingual", device=args.device) as router:
            result = router.predict(STATE, QUESTIONS, lang="zh-CN")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    # 保留答案给读者审查，不将示例阈值直接接到退款、删除等业务操作。


if __name__ == "__main__":
    main()
