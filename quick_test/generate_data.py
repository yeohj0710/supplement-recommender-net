# make_dataset_v2.py
import json, math
from pathlib import Path
import numpy as np
import pandas as pd

RNG = np.random.default_rng(20250902)

QUESTIONS = [
    "하루 중 에너지 저하와 피로감이 잦다",
    "뼈·관절의 통증 또는 약화를 체감한다",
    "스트레스가 크고 수면의 질이 낮다",
    "소화불량·변비·복통 등 위장 문제가 있다",
    "감기·염증 등 면역 저하 증상이 잦다",
    "피부·모발·손톱의 탄력·윤기 저하가 느껴진다",
    "눈의 피로·침침함이 자주 느껴진다",
    "임신 준비 중이거나 임신 중이다",
    "혈압·혈중지질 등 심혈관 위험이 걱정된다",
    "체중 관리·대사 개선이 필요하다",
]

CATEGORIES = [
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

IDX = {c: i for i, c in enumerate(CATEGORIES)}


def impact_matrix():
    M = np.zeros((len(CATEGORIES), 10), dtype=np.float32)

    def w(q, pairs):
        for k, v in pairs:
            M[IDX[k], q] = v

    w(
        0,
        [
            ("비타민B", 1.0),
            ("코엔자임Q10", 0.9),
            ("철분", 0.8),
            ("마그네슘", 0.7),
            ("종합비타민", 0.6),
            ("비타민C", 0.6),
            ("아르기닌", 0.5),
            ("오메가3", 0.4),
            ("미네랄", 0.3),
        ],
    )
    w(
        1,
        [
            ("칼슘", 1.0),
            ("비타민D", 0.9),
            ("콘드로이친", 0.9),
            ("콜라겐", 0.7),
            ("마그네슘", 0.5),
        ],
    )
    w(
        2,
        [
            ("마그네슘", 1.0),
            ("비타민B", 0.8),
            ("포스파티딜세린", 0.75),
            ("오메가3", 0.5),
            ("비타민D", 0.5),
        ],
    )
    w(
        3,
        [
            ("프로바이오틱스(유산균)", 1.0),
            ("차전자피 식이섬유", 0.9),
            ("밀크씨슬(실리마린)", 0.6),
            ("마그네슘", 0.2),
            ("종합비타민", 0.2),
        ],
    )
    w(
        4,
        [
            ("비타민C", 1.0),
            ("아연", 0.9),
            ("비타민D", 0.7),
            ("프로바이오틱스(유산균)", 0.6),
            ("종합비타민", 0.4),
            ("오메가3", 0.3),
        ],
    )
    w(
        5,
        [
            ("콜라겐", 1.0),
            ("비타민C", 0.7),
            ("아연", 0.7),
            ("비타민B", 0.5),
            ("비타민A", 0.6),
            ("오메가3", 0.4),
            ("미네랄", 0.3),
        ],
    )
    w(6, [("루테인", 1.0), ("비타민A", 0.8), ("오메가3", 0.6)])
    w(
        7,
        [
            ("엽산", 1.0),
            ("철분", 0.9),
            ("종합비타민", 0.7),
            ("오메가3", 0.5),
            ("프로바이오틱스(유산균)", 0.3),
        ],
    )
    w(
        8,
        [
            ("오메가3", 1.0),
            ("코엔자임Q10", 0.9),
            ("아르기닌", 0.7),
            ("마그네슘", 0.4),
            ("종합비타민", 0.3),
        ],
    )
    w(
        9,
        [
            ("가르시니아", 1.0),
            ("비타민B", 0.6),
            ("코엔자임Q10", 0.5),
            ("차전자피 식이섬유", 0.5),
            ("프로바이오틱스(유산균)", 0.3),
            ("밀크씨슬(실리마린)", 0.3),
        ],
    )
    return M


PRIOR = np.array(
    [
        0.65,
        0.68,
        0.55,
        0.60,
        0.62,
        0.40,
        0.58,
        0.66,
        0.60,
        0.30,
        0.64,
        0.57,
        0.52,
        0.35,
        0.20,
        0.35,
        0.45,
        0.48,
        0.42,
        0.44,
        0.53,
        0.56,
    ],
    dtype=np.float32,
)

MAINSTREAM = {
    "종합비타민",
    "비타민C",
    "비타민D",
    "오메가3",
    "프로바이오틱스(유산균)",
    "마그네슘",
    "미네랄",
}


def neutral_prior():
    v = np.array(
        [1.0 if c in MAINSTREAM else 0.2 for c in CATEGORIES], dtype=np.float32
    )
    v = v / np.clip(v.sum(), 1e-8, None)
    return v


def severity(a, gamma=2.6):
    x = (a.astype(np.float32) - 1.0) / 4.0
    return np.power(x, gamma)


def focus_boost(s, beta=1.5, rho=1.6):
    return s * (1.0 + beta * np.power(s, rho))


def fused_score(M, s, lam=0.75):
    W = M * s
    mx = W.max(axis=1)
    sm = W.sum(axis=1)
    return lam * mx + (1.0 - lam) * sm


def interactions(s, z):
    z = z.copy()
    z[IDX["비타민B"]] += 0.6 * s[2] * s[9]
    z[IDX["차전자피 식이섬유"]] += 0.5 * s[3] * s[9]
    z[IDX["프로바이오틱스(유산균)"]] += 0.3 * s[3] * s[9]
    z[IDX["비타민C"]] += 0.4 * s[5] * s[0]
    z[IDX["콜라겐"]] += 0.4 * s[5] * s[0]
    z[IDX["오메가3"]] += 0.4 * s[4] * s[8]
    z[IDX["코엔자임Q10"]] += 0.3 * s[4] * s[8]
    return z


def safety_mask(answers):
    m = np.ones(len(CATEGORIES), dtype=np.float32)
    if answers[7] >= 4:
        for c in ["비타민A", "가르시니아", "아르기닌"]:
            m[IDX[c]] = 0.0
    return m


def softmax(x, t):
    x = x / t
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.clip(e.sum(), 1e-8, None)


def temp_from_mu(mu):
    return float(np.clip(1.2 - 1.0 * mu, 0.22, 1.2))


def pow_from_mu(mu):
    return 1.0 + 18.0 * (mu**2.2)


def generate(
    n=14000,
    out_csv="dataset_v2.csv",
    out_labels="label_order.json",
    out_questions="questions_v2.json",
):
    M = impact_matrix()
    rows = []
    np_main = neutral_prior()
    for _ in range(n):
        a = RNG.choice(
            np.arange(1, 6), size=10, p=np.array([0.17, 0.20, 0.26, 0.22, 0.15])
        )
        s = severity(a)
        sf = focus_boost(s)
        z = fused_score(M, sf) + 0.25 * PRIOR
        z = interactions(s, z)
        m = safety_mask(a)
        z = z * m + (-1e6) * (1 - m)
        mu = float(s.mean())
        sig = float(s.std())
        t = temp_from_mu(mu)
        p = softmax(z, t=t)
        if sig < 0.02:
            p = 0.2 * p + 0.8 * np_main
            p = p / np.clip(p.sum(), 1e-8, None)
        tau = pow_from_mu(mu)
        p = np.power(np.clip(p, 1e-8, 1.0), tau)
        p = p / np.clip(p.sum(), 1e-8, None)
        if mu > 0.95 and sig < 1e-6:
            k = int(np.argmax(p))
            y = np.zeros_like(p)
            y[k] = 1.0
        else:
            kappa = 110.0 + 40.0 * (mu**1.5)
            alpha = np.clip(p, 1e-6, None) * kappa
            y = RNG.dirichlet(alpha)
        y = y * m
        if y.sum() <= 0:
            y = np.ones_like(y) / len(y)
        else:
            y = y / y.sum()
        rows.append(list(a.astype(int)) + list(y.astype(np.float32)))
    cols = [f"Q{i}" for i in range(1, 11)] + CATEGORIES
    df = pd.DataFrame(rows, columns=cols)
    Path(out_csv).write_text(
        df.to_csv(index=False, encoding="utf-8-sig"), encoding="utf-8"
    )
    Path(out_labels).write_text(
        json.dumps(CATEGORIES, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    Path(out_questions).write_text(
        json.dumps(QUESTIONS, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    generate()
