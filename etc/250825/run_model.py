import json, os, sys, math, numpy as np, argparse

try:
    import onnxruntime as ort
except:
    sys.exit("onnxruntime가 필요합니다. pip install onnxruntime==1.18.1")

MODEL_PATH = "artifacts/f_user.onnx"
SPEC_PATH = "artifacts/feature_spec.json"
CATS_PATH = "artifacts/cat_params.json"
CALIB_PATH = "artifacts/calibration.json"
GRAPH_PATH = "artifacts/graph.json"


def jload(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def prompt_bool(text, default=None):
    while True:
        s = (
            input(
                f"{text} (y/n){' ['+('y' if default else 'n')+']' if default is not None else ''}: "
            )
            .strip()
            .lower()
        )
        if s == "" and default is not None:
            return 1 if default else 0
        if s in ("y", "yes", "1"):
            return 1
        if s in ("n", "no", "0"):
            return 0


def prompt_number(text, mn=None, mx=None, default=None):
    while True:
        s = input(
            f"{text}{' ['+str(default)+']' if default is not None else ''}: "
        ).strip()
        if s == "" and default is not None:
            return float(default)
        try:
            v = float(s)
            if mn is not None and v < mn:
                continue
            if mx is not None and v > mx:
                continue
            return v
        except:
            continue


def prompt_single(text, options, default=None):
    idx_default = None
    if default is not None:
        for i, o in enumerate(options, 1):
            if o.get("value") == default:
                idx_default = i
    print(text)
    for i, o in enumerate(options, 1):
        print(f"  {i}. {o.get('text','')} ({o.get('value')})")
    while True:
        s = input(
            f"선택 번호 입력{' ['+str(idx_default)+']' if idx_default else ''}: "
        ).strip()
        if s == "" and idx_default:
            return options[idx_default - 1].get("value")
        if s.isdigit():
            i = int(s)
            if 1 <= i <= len(options):
                return options[i - 1].get("value")


def ask(q):
    k = q.get("kind")
    if k == "boolean":
        return prompt_bool(q.get("text", ""))
    if k == "number":
        return prompt_number(q.get("text", ""), q.get("min"), q.get("max"))
    if k in ("likert", "single"):
        return prompt_single(q.get("text", ""), q.get("options", []))
    if k == "text":
        return input(q.get("text", "") + ": ").strip()
    return None


def build_x(spec, feats):
    vals = []
    for f in spec["factors"]:
        name = f["name"]
        mu = float(f["mu"])
        sigma = float(f["sigma"]) if float(f["sigma"]) != 0 else 1e-8
        v = feats.get(name, mu)
        a = (float(v) - mu) / sigma
        vals.append(a)
    x = np.array(vals, dtype=np.float32).reshape(1, -1)
    n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-8
    x = x / n
    return x


def platt(logit, a, b):
    s = a * logit + b
    return 1.0 / (1.0 + math.exp(-s))


def calibrate(logit, calib, label=None):
    if isinstance(calib, dict) and "method" in calib:
        if calib["method"] == "platt":
            return platt(logit, float(calib.get("a", 1.0)), float(calib.get("b", 0.0)))
    if isinstance(calib, dict) and label in calib:
        c = calib[label]
        return platt(logit, float(c.get("a", 1.0)), float(c.get("b", 0.0)))
    return 1.0 / (1.0 + math.exp(-logit))


def run_two_stage(plan):
    feats = {}
    demos = plan.get("demographics", [])
    for q in demos:
        v = ask(q)
        fk = q.get("featureKey")
        if fk is not None:
            feats[fk] = float(v) if isinstance(v, (int, float)) else 0.0
        if q.get("id") == "q_weight":
            h = feats.get("f_height_cm", None)
            w = feats.get("f_weight_kg", None)
            if h and w:
                bmi = w / ((h / 100.0) ** 2)
                feats["f_bmi"] = float(round(bmi, 1))
    dom_scores = {}
    scr = plan.get("screening", [])
    print("관심 분야를 파악합니다.")
    for q in scr:
        v = ask(q)
        fk = q.get("featureKey")
        if fk is not None:
            feats[fk] = float(v) if isinstance(v, (int, float)) else 0.0
        d = q.get("domain")
        w = q.get("weight", 1.0)
        if d:
            dom_scores[d] = dom_scores.get(d, 0.0) + float(v) * float(w)
    top_n = int(plan.get("top_n_modules", 3))
    picked = sorted(dom_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    picked = [k for k, _ in picked]
    if feats.get("f_pregnant", 0) >= 1 and "pregnancy" in plan.get("modules", {}):
        if "pregnancy" not in picked:
            picked.append("pregnancy")
    mods = plan.get("modules", {})
    need = {}
    for d in picked:
        m = mods.get(d)
        if not m:
            continue
        print(f"[{m.get('title',d)}] 세부 질문")
        for q in m.get("questions", []):
            v = ask(q)
            fk = q.get("featureKey")
            if fk is not None:
                feats[fk] = float(v) if isinstance(v, (int, float)) else 0.0
        gate = m.get("gate", {})
        feats_for_gate = [
            feats.get(k, 0.0) for k in gate.get("features", m.get("questions", []))
        ]
        if not feats_for_gate and m.get("questions", []):
            for q in m["questions"]:
                fk = q.get("featureKey")
                if fk:
                    feats_for_gate.append(feats.get(fk, 0.0))
        th = float(gate.get("threshold", 3.0))
        g = np.mean(feats_for_gate) if feats_for_gate else 0.0
        need[d] = g >= th
    return feats, picked, need, mods


def choose_results(scored, picked, need, mods, k):
    allow = set()
    for d in picked:
        if need.get(d, True):
            for c in mods.get(d, {}).get("categories", []):
                allow.add(c)
    pri = [t for t in scored if t[1] in allow]
    rest = [t for t in scored if t[1] not in allow]
    out = []
    for t in pri:
        if len(out) < k:
            out.append(t)
    for t in rest:
        if len(out) < k:
            out.append(t)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topk", type=int, default=3)
    args, _ = parser.parse_known_args()
    for p in [MODEL_PATH, SPEC_PATH, CATS_PATH, GRAPH_PATH]:
        if not os.path.exists(p):
            sys.exit("파일을 찾을 수 없습니다: " + p)
    graph = jload(GRAPH_PATH)
    if graph.get("mode") != "two_stage":
        sys.exit("graph.json이 two_stage 스펙이 아닙니다.")
    feats, picked, need, mods = run_two_stage(graph)
    spec = jload(SPEC_PATH)
    cats = jload(CATS_PATH)
    calib = jload(CALIB_PATH) if os.path.exists(CALIB_PATH) else {}
    x = build_x(spec, feats)
    sess = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
    out = sess.run(None, {"input": x})
    z = out[0].astype(np.float32).reshape(-1)
    n = np.linalg.norm(z) + 1e-8
    z = z / n
    scored = []
    for key, obj in cats.items():
        w = np.asarray(obj["w"], dtype=np.float32)
        b = float(obj["b"])
        logit = float(np.dot(w, z) + b)
        p = calibrate(logit, calib, label=key)
        score = max(0.0, min(100.0, 100.0 * p))
        name = obj.get("name", key)
        scored.append((name, key, score))
    scored.sort(key=lambda x: x[2], reverse=True)
    final = choose_results(scored, picked, need, mods, args.topk)
    print(
        "선택 도메인:",
        ", ".join([mods[d]["title"] if d in mods else d for d in picked]),
    )
    print(
        "도메인 필요 여부:",
        ", ".join(
            [
                f'{mods[d]["title"] if d in mods else d}={"Y" if need.get(d,True) else "N"}'
                for d in picked
            ]
        ),
    )
    print("Top", args.topk)
    for i, (name, key, score) in enumerate(final[: args.topk], 1):
        print(f"{i}. {name} ({key}) score={score:.1f}")
    print("입력 요약")
    for k in sorted(feats.keys()):
        print(f"{k}={feats[k]}")


if __name__ == "__main__":
    main()
