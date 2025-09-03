import os
import json
import argparse
import numpy as np
import onnxruntime as ort

DEF_CATEGORIES = [
    "비타민C",
    "오메가3",
    "칼슘",
    "루테인",
    "비타민D",
    "밀크씨슬(실리마린)",
    "프로바이오틱스(유산균)",
    "비타민B",
    "마그네슘",
    "가르시니아",
    "종합비타민",
    "아연",
    "차전자피 식이섬유",
    "미네랄",
    "비타민A",
    "철분",
    "포스파티딜세린",
    "엽산",
    "아르기닌",
    "콘드로이친",
    "코엔자임Q10",
    "콜라겐",
]

DEF_QUESTIONS = [
    "하루 중 쉽게 피로해지고 에너지가 부족하다",
    "뼈·관절 통증이나 뻣뻣함, 무릎/허리 부담이 자주 느껴진다",
    "스트레스가 높고 불안·예민함이 지속되거나 수면의 질이 낮다",
    "소화가 더디고 속 더부룩함·가스·변비 등 장 불편이 자주 있다",
    "감기·염증 등 잔병치레가 잦거나 회복이 느리다",
    "피부 탄력·건조·트러블이 걱정되거나 모발·손톱 상태가 좋지 않다",
    "눈이 쉽게 피로하고 침침하거나 밤에 시야가 불편하다",
    "음주가 잦거나 기름진 음식·약물 복용 등으로 간 건강이 걱정된다",
    "혈압·혈중 지질·심박·복부지방 등 대사/심혈관 건강이 걱정된다",
    "임신 준비·임신/수유 중이거나 월경 과다·빈혈 증상이 걱정된다",
]


def load_categories(meta_path, data_dir):
    cats = None
    if os.path.isfile(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            m = json.load(f)
            cats = m.get("categories")
    if cats is None:
        npz_path = os.path.join(data_dir, "dataset.npz")
        if os.path.isfile(npz_path):
            d = np.load(npz_path, allow_pickle=True)
            cats = d["categories"].tolist()
    if cats is None:
        cats = DEF_CATEGORIES
    return cats


def load_questions(meta_path, data_dir):
    qs = None
    data_meta = os.path.join(data_dir, "meta.json")
    if os.path.isfile(data_meta):
        try:
            with open(data_meta, "r", encoding="utf-8") as f:
                m = json.load(f)
                qs = m.get("questions")
        except:
            qs = None
    if qs is None and os.path.isfile(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                m = json.load(f)
                qs = m.get("questions")
        except:
            qs = None
    if qs is None:
        npz_path = os.path.join(data_dir, "dataset.npz")
        if os.path.isfile(npz_path):
            d = np.load(npz_path, allow_pickle=True)
            if "questions" in d:
                qs = d["questions"].tolist()
    if qs is None or len(qs) != 10:
        qs = DEF_QUESTIONS
    return qs


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def preprocess(ans):
    a = np.array(ans, dtype=np.float32).reshape(1, -1)
    a = np.clip(a, 1.0, 5.0)
    z = (a - 1.0) / 4.0
    return z


def run_onnx(session, z):
    out = session.run(None, {"input": z})[0]
    p = sigmoid(out.astype(np.float32))
    return p


def to_percent(p):
    v = np.clip(p * 100.0, 0.0, 100.0)
    return np.round(v, 1)


def sort_top(pct, cats, topk):
    idx = np.argsort(-pct)
    idx = idx[:topk]
    return [(cats[i], float(pct[i])) for i in idx]


def parse_answers(s):
    t = [x for x in s.replace("/", ",").replace(" ", ",").split(",") if x != ""]
    v = [float(x) for x in t]
    if len(v) != 10:
        raise ValueError("answers length must be 10")
    return v


def print_text_result(title, pairs):
    print(title)
    for i, (label, pct) in enumerate(pairs, 1):
        print(f"{i:>2}. {label}: {pct:.1f}%")
    print()


def anchors_all(v):
    return [float(v)] * 10


def anchors_spike(q):
    a = [1.0] * 10
    a[q] = 5.0
    return a


def do_single(session, cats, answers, topk, outfmt):
    z = preprocess(answers)
    p = run_onnx(session, z)[0]
    pct = to_percent(p)
    pairs = sort_top(pct, cats, topk)
    if outfmt == "json":
        print(
            json.dumps(
                {
                    "answers": answers,
                    "results": [{"label": k, "percent": v} for k, v in pairs],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        print_text_result("결과", pairs)


def do_anchors(session, cats, mode, topk, outfmt):
    payloads = []
    if mode in ("all1", "all"):
        payloads.append(("모두 1", anchors_all(1)))
    if mode in ("all3", "all"):
        payloads.append(("모두 3", anchors_all(3)))
    if mode in ("all5", "all"):
        payloads.append(("모두 5", anchors_all(5)))
    if mode in ("spike", "all"):
        for q in range(10):
            payloads.append((f"Q{q+1}=5, 나머지=1", anchors_spike(q)))
    results = []
    for title, ans in payloads:
        z = preprocess(ans)
        p = run_onnx(session, z)[0]
        pct = to_percent(p)
        pairs = sort_top(pct, cats, topk)
        results.append(
            {
                "case": title,
                "answers": ans,
                "results": [{"label": k, "percent": v} for k, v in pairs],
            }
        )
    if outfmt == "json":
        print(json.dumps({"anchors": results}, ensure_ascii=False, indent=2))
    else:
        for r in results:
            print_text_result(
                r["case"], [(x["label"], x["percent"]) for x in r["results"]]
            )


def prompt_int(msg, default=3):
    while True:
        s = input(msg).strip()
        if s == "":
            return float(default)
        try:
            v = float(s)
            if 1.0 <= v <= 5.0:
                return v
        except:
            pass
        print("1~5 사이 숫자를 입력하거나 Enter로 건너뛰세요.")


def do_interactive(session, cats, questions, topk, outfmt):
    print("=== 건강 설문 (1~5 척도) ===")
    print("지난 4주 기준으로 1~5를 입력하세요. Enter는 3으로 처리됩니다.")
    answers = []
    for i in range(10):
        q = questions[i] if i < len(questions) else f"문항 {i+1}"
        v = prompt_int(f"{i+1}) {q} [1~5] : ")
        answers.append(v)
    do_single(session, cats, answers, topk, outfmt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="artifacts/model.onnx")
    ap.add_argument("--meta", type=str, default="artifacts/meta.json")
    ap.add_argument("--data_dir", type=str, default="data")
    ap.add_argument("--answers", type=str, default=None)
    ap.add_argument(
        "--anchors",
        type=str,
        default=None,
        choices=[None, "all1", "all3", "all5", "spike", "all"],
    )
    ap.add_argument("--topk", type=int, default=22)
    ap.add_argument("--format", type=str, default="text", choices=["text", "json"])
    args = ap.parse_args()
    cats = load_categories(args.meta, args.data_dir)
    questions = load_questions(args.meta, args.data_dir)
    sess = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    if args.answers is None and args.anchors is None:
        do_interactive(sess, cats, questions, args.topk, args.format)
        return
    if args.answers is not None:
        ans = parse_answers(args.answers)
        do_single(sess, cats, ans, args.topk, args.format)
    if args.anchors is not None:
        do_anchors(sess, cats, args.anchors, args.topk, args.format)


if __name__ == "__main__":
    main()
