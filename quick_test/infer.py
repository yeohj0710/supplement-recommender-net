# infer_v2.py
import json, argparse
import numpy as np
import onnxruntime as ort
from pathlib import Path

LABEL_JSON = Path("label_order.json")
QUEST_JSON = Path("questions_v2.json")
MODEL_PATH = Path("survey_v2.onnx")

MAINSTREAM = {
    "종합비타민",
    "비타민C",
    "비타민D",
    "오메가3",
    "프로바이오틱스(유산균)",
    "마그네슘",
    "미네랄",
}


def make_session(p):
    try:
        return ort.InferenceSession(
            str(p), providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
    except:
        return ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])


def neutral_prior(labels):
    v = np.array([1.0 if c in MAINSTREAM else 0.2 for c in labels], dtype=np.float32)
    return v / np.clip(v.sum(), 1e-8, None)


def safety_mask(answers, labels):
    m = np.ones(len(labels), dtype=np.float32)
    if answers[7] >= 4:
        for c in ["비타민A", "가르시니아", "아르기닌"]:
            if c in labels:
                m[labels.index(c)] = 0.0
    return m


def run_once(sess, x):
    name_in = sess.get_inputs()[0].name
    name_out = sess.get_outputs()[0].name
    return sess.run([name_out], {name_in: x.astype(np.float32)})[0][0]


def format_top3(labels, probs):
    idx = np.argsort(-probs)[:3]
    p = probs[idx]
    if p.sum() <= 0:
        p[:] = 1 / 3
    else:
        p = p / p.sum()
    p = np.round(p * 100, 1)
    return [(labels[i], float(p[j])) for j, i in enumerate(idx)]


def power_from_mu(mu):
    return 1.0 + 18.0 * (mu**2.2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=str(MODEL_PATH))
    parser.add_argument("--labels", type=str, default=str(LABEL_JSON))
    parser.add_argument("--questions", type=str, default=str(QUEST_JSON))
    args = parser.parse_args()

    labels = json.loads(Path(args.labels).read_text(encoding="utf-8"))
    questions = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    sess = make_session(args.model)

    print("=== 10문항 리커트 설문(1~5) ===")
    answers = []
    for i, q in enumerate(questions, 1):
        while True:
            try:
                v = int(input(f"{i}) {q} [1~5]: ").strip())
                if 1 <= v <= 5:
                    answers.append(v)
                    break
            except:
                pass
            print("1~5 사이의 정수를 입력하세요.")
    x = np.array(answers, dtype=np.float32).reshape(1, -1)
    probs = run_once(sess, x)
    m = safety_mask(answers, labels)
    probs = probs * m
    if probs.sum() <= 0:
        probs[:] = 1 / len(probs)
    probs = probs / np.clip(probs.sum(), 1e-8, None)
    s = ((np.array(answers, dtype=np.float32) - 1.0) / 4.0) ** 2.6
    mu = float(s.mean())
    sig = float(s.std())
    if sig < 0.02:
        np_main = neutral_prior(labels)
        probs = 0.2 * probs + 0.8 * np_main
        probs = probs / np.clip(probs.sum(), 1e-8, None)
    tau = power_from_mu(mu)
    probs = np.power(np.clip(probs, 1e-8, 1.0), tau)
    probs = probs / np.clip(probs.sum(), 1e-8, None)
    if mu > 0.95 and sig < 1e-6:
        k = int(np.argmax(probs))
        probs[:] = 0.0
        probs[k] = 1.0
    top3 = format_top3(labels, probs)
    print("\n=== 추천 상위 3 ===")
    for name, pct in top3:
        print(f"- {name}: {pct:.1f}%")


if __name__ == "__main__":
    main()
