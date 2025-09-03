import os
import json
import argparse
import numpy as np
from math import ceil
from csv import writer


def categories():
    return [
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


def questions():
    return [
        "하루 중 쉽게 피로해지고 에너지가 부족하다고 느낀다",
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


def build_weight_matrix(cats):
    idx = {c: i for i, c in enumerate(cats)}
    W = np.zeros((10, len(cats)), dtype=np.float32)

    def s(q, items):
        for c, w in items:
            W[q, idx[c]] += w

    s(
        0,
        [
            ("비타민B", 1.0),
            ("코엔자임Q10", 0.9),
            ("철분", 0.8),
            ("아르기닌", 0.7),
            ("종합비타민", 0.5),
        ],
    )
    s(
        1,
        [
            ("칼슘", 1.0),
            ("비타민D", 0.9),
            ("콘드로이친", 0.8),
            ("콜라겐", 0.6),
            ("미네랄", 0.5),
        ],
    )
    s(2, [("마그네슘", 1.0), ("비타민B", 0.7), ("포스파티딜세린", 0.6)])
    s(3, [("프로바이오틱스(유산균)", 1.0), ("차전자피 식이섬유", 0.8), ("아연", 0.6)])
    s(4, [("비타민C", 1.0), ("아연", 0.8), ("비타민D", 0.7), ("종합비타민", 0.5)])
    s(
        5,
        [
            ("콜라겐", 1.0),
            ("비타민C", 0.7),
            ("비타민A", 0.6),
            ("미네랄", 0.5),
            ("비타민B", 0.4),
        ],
    )
    s(6, [("루테인", 1.0), ("비타민A", 0.7), ("오메가3", 0.6)])
    s(7, [("밀크씨슬(실리마린)", 1.0)])
    s(
        8,
        [
            ("오메가3", 1.0),
            ("코엔자임Q10", 0.8),
            ("마그네슘", 0.7),
            ("가르시니아", 0.6),
            ("차전자피 식이섬유", 0.6),
            ("미네랄", 0.5),
        ],
    )
    s(9, [("엽산", 1.0), ("철분", 0.9), ("오메가3", 0.5), ("종합비타민", 0.5)])
    return W


def compute_targets(X, W, cats, rng):
    Z = (X - 1) / 4.0
    raw = Z @ W
    denom = np.sum(W, axis=0, keepdims=True) + 1e-6
    y = raw / denom
    mean_z = np.mean(Z, axis=1, keepdims=True)
    std_z = np.std(Z, axis=1, keepdims=True)
    u = 1.0 - np.clip(std_z / 0.5, 0, 1)
    safe = ["종합비타민", "비타민C", "오메가3", "프로바이오틱스(유산균)", "비타민D"]
    prior = np.zeros((1, len(cats)), dtype=np.float32)
    for i, c in enumerate(cats):
        if c in safe:
            if c == "종합비타민":
                prior[0, i] = 1.0
            elif c == "비타민C":
                prior[0, i] = 0.6
            elif c == "오메가3":
                prior[0, i] = 0.55
            elif c == "프로바이오틱스(유산균)":
                prior[0, i] = 0.5
            elif c == "비타민D":
                prior[0, i] = 0.5
    base_strength = 0.15
    base_scale = u * mean_z
    y = y + base_strength * base_scale * prior
    eps = 0.01 + 0.02 * mean_z
    noise_sigma = 0.03 * (0.3 + (1.0 - u))
    noise = rng.normal(0, 1, size=y.shape).astype(np.float32) * noise_sigma
    y = y + eps + noise
    y = np.clip(y, 0.0, 1.0)
    return y


def sample_random(n, rng):
    modes = rng.choice([0, 1, 2], size=n, p=[0.6, 0.25, 0.15])
    X = np.zeros((n, 10), dtype=np.int64)
    for i, m in enumerate(modes):
        if m == 0:
            v = rng.normal(3.0, 0.7, size=10)
        elif m == 1:
            v = rng.choice([1, 5], size=10, p=[0.5, 0.5])
        else:
            v = rng.integers(1, 6, size=10)
        X[i] = np.clip(np.rint(v), 1, 5)
    return X


def anchors_all_1(k):
    return np.ones((k, 10), dtype=np.int64)


def anchors_all_3(k):
    return np.full((k, 10), 3, dtype=np.int64)


def anchors_all_5(k):
    return np.full((k, 10), 5, dtype=np.int64)


def anchors_one_5_rest_1(rep):
    X = []
    for _ in range(rep):
        for q in range(10):
            v = np.ones(10, dtype=np.int64)
            v[q] = 5
            X.append(v)
    return np.stack(X, axis=0)


def split_indices(n, train_ratio, val_ratio, rng):
    idx = np.arange(n)
    rng.shuffle(idx)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train_idx = idx[:n_train]
    val_idx = idx[n_train : n_train + n_val]
    test_idx = idx[n_train + n_val :]
    return train_idx, val_idx, test_idx


def save_npz(out_dir, cats, qtexts, X, y, split):
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(
        os.path.join(out_dir, "dataset.npz"),
        X_train=X[split[0]],
        y_train=y[split[0]],
        X_val=X[split[1]],
        y_val=y[split[1]],
        X_test=X[split[2]],
        y_test=y[split[2]],
        categories=np.array(cats, dtype=object),
        questions=np.array(qtexts, dtype=object),
    )
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(
            {"categories": cats, "questions": qtexts}, f, ensure_ascii=False, indent=2
        )


def save_csvs(out_dir, cats, X, y, split):
    os.makedirs(out_dir, exist_ok=True)
    headers = [f"Q{i+1}" for i in range(10)] + cats

    def write_csv(path, Xi, yi):
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = writer(f)
            w.writerow(headers)
            for a, b in zip(Xi, yi):
                w.writerow(
                    list(map(int, a.tolist()))
                    + list(map(lambda z: f"{z:.6f}", b.tolist()))
                )

    write_csv(os.path.join(out_dir, "train.csv"), X[split[0]], y[split[0]])
    write_csv(os.path.join(out_dir, "val.csv"), X[split[1]], y[split[1]])
    write_csv(os.path.join(out_dir, "test.csv"), X[split[2]], y[split[2]])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=20000)
    p.add_argument("--out", type=str, default="data")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train_ratio", type=float, default=0.8)
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--anchors_all1", type=int, default=300)
    p.add_argument("--anchors_all3", type=int, default=600)
    p.add_argument("--anchors_all5", type=int, default=600)
    p.add_argument("--anchors_spike_per_q", type=int, default=60)
    args = p.parse_args()
    rng = np.random.default_rng(args.seed)
    cats = categories()
    qtexts = questions()
    W = build_weight_matrix(cats)
    Xa1 = anchors_all_1(args.anchors_all1)
    Xa3 = anchors_all_3(args.anchors_all3)
    Xa5 = anchors_all_5(args.anchors_all5)
    Xsp = anchors_one_5_rest_1(args.anchors_spike_per_q)
    n_anchor = Xa1.shape[0] + Xa3.shape[0] + Xa5.shape[0] + Xsp.shape[0]
    n_random = max(0, args.n - n_anchor)
    Xr = (
        sample_random(n_random, rng)
        if n_random > 0
        else np.zeros((0, 10), dtype=np.int64)
    )
    X = np.concatenate([Xa1, Xa3, Xa5, Xsp, Xr], axis=0)
    y = compute_targets(X.astype(np.float32), W, cats, rng)
    idx_tr, idx_va, idx_te = split_indices(
        X.shape[0], args.train_ratio, args.val_ratio, rng
    )
    save_npz(args.out, cats, qtexts, X, y, (idx_tr, idx_va, idx_te))
    save_csvs(args.out, cats, X, y, (idx_tr, idx_va, idx_te))
    print(
        json.dumps(
            {
                "total": int(X.shape[0]),
                "train": int(len(idx_tr)),
                "val": int(len(idx_va)),
                "test": int(len(idx_te)),
                "anchors_all1": int(Xa1.shape[0]),
                "anchors_all3": int(Xa3.shape[0]),
                "anchors_all5": int(Xa5.shape[0]),
                "anchors_spike": int(Xsp.shape[0]),
                "random": int(Xr.shape[0]),
                "out_dir": args.out,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
