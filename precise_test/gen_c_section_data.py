# gen_c_section_data.py
import os, argparse, json, random
import numpy as np
import pandas as pd


def seed_all(s):
    random.seed(s)
    np.random.seed(s)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sample_yesno(theta, pos=True):
    p = 0.15 + 0.7 * theta
    p = p if pos else 1 - p
    return int(np.random.rand() < p)


def sample_ord3(theta, pos=True):
    p = 0.15 + 0.7 * theta
    p = p if pos else 1 - p
    return int(np.random.binomial(3, p))


def to_effect(v, t, d):
    x = float(v) if t == "yesno" else float(v) / 3.0
    return x if d > 0 else 1.0 - x


def gen_row(cat):
    a, b = cat["beta"]
    theta = np.clip(np.random.beta(a, b), 0, 1)
    vals = []
    for q in cat["questions"]:
        if q["type"] == "yesno":
            vals.append(sample_yesno(theta, pos=(q["dir"] > 0)))
        else:
            vals.append(sample_ord3(theta, pos=(q["dir"] > 0)))
    effects = [
        to_effect(vals[i], cat["questions"][i]["type"], cat["questions"][i]["dir"])
        for i in range(5)
    ]
    w = np.array([q["w"] for q in cat["questions"]], dtype=float)
    w = w / np.sum(np.abs(w))
    s = float(np.dot(w, effects))
    for sy in cat.get("synergy", []):
        prod = 1.0
        for idx in sy["idx"]:
            prod *= effects[idx]
        s += sy["coef"] * prod
    s = np.clip(s, 0, 1)
    s = sigmoid(cat["scale"]["a"] * (s - cat["scale"]["b"]))
    s = np.clip(s + np.random.normal(0, cat["noise"]), 0, 1)
    label = int(round(100 * s))
    return vals, label


def catalog():
    C = []
    C.append(
        dict(
            key="vitc",
            beta=(1.4, 1.4),
            noise=0.04,
            scale=dict(a=6.0, b=0.5),
            questions=[
                dict(qid="C.VITC.1", type="likert4", dir=+1, w=0.30),
                dict(qid="C.VITC.2", type="likert4", dir=+1, w=0.22),
                dict(qid="C.VITC.3", type="likert4", dir=+1, w=0.20),
                dict(qid="C.VITC.4", type="servings_day4", dir=-1, w=0.18),
                dict(qid="C.VITC.5", type="likert4", dir=+1, w=0.10),
            ],
            synergy=[dict(idx=[0, 2], coef=0.12), dict(idx=[1, 4], coef=0.08)],
        )
    )
    C.append(
        dict(
            key="omega3",
            beta=(1.5, 1.5),
            noise=0.04,
            scale=dict(a=6.0, b=0.5),
            questions=[
                dict(qid="C.OME3.1", type="freq_wk4", dir=-1, w=0.28),
                dict(qid="C.OME3.2", type="yesno", dir=+1, w=0.22),
                dict(qid="C.OME3.3", type="likert4", dir=+1, w=0.20),
                dict(qid="C.OME3.4", type="likert4", dir=+1, w=0.18),
                dict(qid="C.OME3.5", type="likert4", dir=+1, w=0.12),
            ],
            synergy=[dict(idx=[2, 3], coef=0.12), dict(idx=[1, 4], coef=0.10)],
        )
    )
    C.append(
        dict(
            key="ca",
            beta=(1.6, 1.4),
            noise=0.04,
            scale=dict(a=6.0, b=0.52),
            questions=[
                dict(qid="C.CA.1", type="servings_day4", dir=-1, w=0.25),
                dict(qid="C.CA.2", type="yesno", dir=+1, w=0.25),
                dict(qid="C.CA.3", type="freq_wk4", dir=-1, w=0.18),
                dict(qid="C.CA.4", type="yesno", dir=+1, w=0.18),
                dict(qid="C.CA.5", type="yesno", dir=+1, w=0.14),
            ],
            synergy=[dict(idx=[1, 4], coef=0.15), dict(idx=[1, 3], coef=0.10)],
        )
    )
    C.append(
        dict(
            key="lutein",
            beta=(1.5, 1.5),
            noise=0.04,
            scale=dict(a=6.0, b=0.5),
            questions=[
                dict(qid="C.LUT.1", type="likert4", dir=+1, w=0.28),
                dict(qid="C.LUT.2", type="likert4", dir=+1, w=0.22),
                dict(qid="C.LUT.3", type="servings_day4", dir=-1, w=0.18),
                dict(qid="C.LUT.4", type="likert4", dir=+1, w=0.18),
                dict(qid="C.LUT.5", type="freq_wk4", dir=-1, w=0.14),
            ],
            synergy=[dict(idx=[0, 1], coef=0.12), dict(idx=[1, 3], coef=0.10)],
        )
    )
    C.append(
        dict(
            key="vitd",
            beta=(1.4, 1.6),
            noise=0.04,
            scale=dict(a=6.0, b=0.52),
            questions=[
                dict(qid="C.VITD.1", type="likert4", dir=+1, w=0.26),
                dict(qid="C.VITD.2", type="likert4", dir=+1, w=0.18),
                dict(qid="C.VITD.3", type="yesno", dir=+1, w=0.22),
                dict(qid="C.VITD.4", type="likert4", dir=+1, w=0.18),
                dict(qid="C.VITD.5", type="likert4", dir=+1, w=0.16),
            ],
            synergy=[dict(idx=[0, 4], coef=0.12), dict(idx=[0, 3], coef=0.10)],
        )
    )
    C.append(
        dict(
            key="milkthistle",
            beta=(1.5, 1.5),
            noise=0.05,
            scale=dict(a=6.0, b=0.5),
            questions=[
                dict(qid="C.MLK.1", type="likert4", dir=+1, w=0.26),
                dict(qid="C.MLK.2", type="yesno", dir=+1, w=0.22),
                dict(qid="C.MLK.3", type="likert4", dir=+1, w=0.20),
                dict(qid="C.MLK.4", type="yesno", dir=+1, w=0.16),
                dict(qid="C.MLK.5", type="likert4", dir=+1, w=0.16),
            ],
            synergy=[dict(idx=[0, 2], coef=0.12), dict(idx=[1, 3], coef=0.10)],
        )
    )
    C.append(
        dict(
            key="probiotics",
            beta=(1.5, 1.5),
            noise=0.05,
            scale=dict(a=6.0, b=0.5),
            questions=[
                dict(qid="C.PRO.1", type="likert4", dir=+1, w=0.26),
                dict(qid="C.PRO.2", type="likert4", dir=+1, w=0.22),
                dict(qid="C.PRO.3", type="yesno", dir=+1, w=0.18),
                dict(qid="C.PRO.4", type="freq_wk4", dir=-1, w=0.16),
                dict(qid="C.PRO.5", type="likert4", dir=+1, w=0.18),
            ],
            synergy=[dict(idx=[0, 2], coef=0.12), dict(idx=[1, 4], coef=0.10)],
        )
    )
    C.append(
        dict(
            key="vitb",
            beta=(1.6, 1.4),
            noise=0.05,
            scale=dict(a=6.0, b=0.5),
            questions=[
                dict(qid="C.VITB.1", type="likert4", dir=+1, w=0.30),
                dict(qid="C.VITB.2", type="likert4", dir=+1, w=0.18),
                dict(qid="C.VITB.3", type="likert4", dir=+1, w=0.24),
                dict(qid="C.VITB.4", type="likert4", dir=+1, w=0.14),
                dict(qid="C.VITB.5", type="likert4", dir=+1, w=0.14),
            ],
            synergy=[dict(idx=[0, 2], coef=0.12)],
        )
    )
    C.append(
        dict(
            key="mg",
            beta=(1.5, 1.5),
            noise=0.05,
            scale=dict(a=6.0, b=0.5),
            questions=[
                dict(qid="C.MG.1", type="likert4", dir=+1, w=0.28),
                dict(qid="C.MG.2", type="likert4", dir=+1, w=0.20),
                dict(qid="C.MG.3", type="likert4", dir=+1, w=0.16),
                dict(qid="C.MG.4", type="likert4", dir=+1, w=0.20),
                dict(qid="C.MG.5", type="likert4", dir=+1, w=0.16),
            ],
            synergy=[dict(idx=[0, 1], coef=0.12), dict(idx=[0, 3], coef=0.10)],
        )
    )
    C.append(
        dict(
            key="garcinia",
            beta=(1.5, 1.5),
            noise=0.05,
            scale=dict(a=6.5, b=0.5),
            questions=[
                dict(qid="C.GAR.1", type="likert4", dir=+1, w=0.26),
                dict(qid="C.GAR.2", type="likert4", dir=+1, w=0.22),
                dict(qid="C.GAR.3", type="likert4", dir=+1, w=0.20),
                dict(qid="C.GAR.4", type="yesno", dir=+1, w=0.14),
                dict(qid="C.GAR.5", type="likert4", dir=+1, w=0.18),
            ],
            synergy=[dict(idx=[0, 2], coef=0.12), dict(idx=[1, 4], coef=0.10)],
        )
    )
    C.append(
        dict(
            key="multivitamin",
            beta=(1.4, 1.6),
            noise=0.05,
            scale=dict(a=6.0, b=0.52),
            questions=[
                dict(qid="C.MV.1", type="likert4", dir=+1, w=0.26),
                dict(qid="C.MV.2", type="likert4", dir=+1, w=0.18),
                dict(qid="C.MV.3", type="likert4", dir=+1, w=0.22),
                dict(qid="C.MV.4", type="likert4", dir=+1, w=0.18),
                dict(qid="C.MV.5", type="likert4", dir=+1, w=0.16),
            ],
            synergy=[dict(idx=[0, 2], coef=0.10), dict(idx=[1, 3], coef=0.10)],
        )
    )
    C.append(
        dict(
            key="zn",
            beta=(1.5, 1.5),
            noise=0.05,
            scale=dict(a=6.0, b=0.5),
            questions=[
                dict(qid="C.ZN.1", type="likert4", dir=+1, w=0.24),
                dict(qid="C.ZN.2", type="likert4", dir=+1, w=0.22),
                dict(qid="C.ZN.3", type="likert4", dir=+1, w=0.20),
                dict(qid="C.ZN.4", type="yesno", dir=+1, w=0.18),
                dict(qid="C.ZN.5", type="freq_wk4", dir=-1, w=0.16),
            ],
            synergy=[dict(idx=[1, 3], coef=0.10), dict(idx=[0, 2], coef=0.10)],
        )
    )
    C.append(
        dict(
            key="psyllium",
            beta=(1.5, 1.5),
            noise=0.05,
            scale=dict(a=6.0, b=0.5),
            questions=[
                dict(qid="C.PSY.1", type="likert4", dir=+1, w=0.26),
                dict(qid="C.PSY.2", type="likert4", dir=+1, w=0.18),
                dict(qid="C.PSY.3", type="likert4", dir=+1, w=0.20),
                dict(qid="C.PSY.4", type="water_cups_day4", dir=+1, w=0.18),
                dict(qid="C.PSY.5", type="likert4", dir=-1, w=0.18),
            ],
            synergy=[dict(idx=[0, 3], coef=0.12), dict(idx=[1, 2], coef=0.10)],
        )
    )
    C.append(
        dict(
            key="minerals",
            beta=(1.5, 1.5),
            noise=0.05,
            scale=dict(a=6.0, b=0.5),
            questions=[
                dict(qid="C.MIN.1", type="freq_wk4", dir=+1, w=0.24),
                dict(qid="C.MIN.2", type="likert4", dir=+1, w=0.22),
                dict(qid="C.MIN.3", type="likert4", dir=+1, w=0.20),
                dict(qid="C.MIN.4", type="yesno", dir=+1, w=0.16),
                dict(qid="C.MIN.5", type="freq_wk4", dir=-1, w=0.18),
            ],
            synergy=[dict(idx=[0, 2], coef=0.10), dict(idx=[1, 3], coef=0.10)],
        )
    )
    C.append(
        dict(
            key="vita",
            beta=(1.5, 1.5),
            noise=0.05,
            scale=dict(a=6.2, b=0.5),
            questions=[
                dict(qid="C.VITA.1", type="likert4", dir=+1, w=0.26),
                dict(qid="C.VITA.2", type="likert4", dir=+1, w=0.22),
                dict(qid="C.VITA.3", type="likert4", dir=+1, w=0.16),
                dict(qid="C.VITA.4", type="freq_wk4", dir=-1, w=0.18),
                dict(qid="C.VITA.5", type="yesno", dir=+1, w=0.18),
            ],
            synergy=[dict(idx=[0, 1], coef=0.12)],
        )
    )
    C.append(
        dict(
            key="fe",
            beta=(1.5, 1.5),
            noise=0.05,
            scale=dict(a=6.4, b=0.5),
            questions=[
                dict(qid="C.FE.1", type="likert4", dir=+1, w=0.24),
                dict(qid="C.FE.2", type="yesno", dir=+1, w=0.22),
                dict(qid="C.FE.3", type="yesno", dir=+1, w=0.18),
                dict(qid="C.FE.4", type="freq_wk4", dir=-1, w=0.18),
                dict(qid="C.FE.5", type="likert4", dir=-1, w=0.18),
            ],
            synergy=[dict(idx=[1, 2], coef=0.14), dict(idx=[0, 1], coef=0.10)],
        )
    )
    C.append(
        dict(
            key="ps",
            beta=(1.5, 1.5),
            noise=0.05,
            scale=dict(a=6.0, b=0.5),
            questions=[
                dict(qid="C.PS.1", type="likert4", dir=+1, w=0.26),
                dict(qid="C.PS.2", type="likert4", dir=+1, w=0.22),
                dict(qid="C.PS.3", type="likert4", dir=+1, w=0.18),
                dict(qid="C.PS.4", type="likert4", dir=+1, w=0.16),
                dict(qid="C.PS.5", type="likert4", dir=+1, w=0.18),
            ],
            synergy=[dict(idx=[0, 3], coef=0.10), dict(idx=[1, 2], coef=0.10)],
        )
    )
    C.append(
        dict(
            key="folate",
            beta=(1.4, 1.6),
            noise=0.05,
            scale=dict(a=6.0, b=0.5),
            questions=[
                dict(qid="C.FOL.1", type="yesno", dir=+1, w=0.30),
                dict(qid="C.FOL.2", type="yesno", dir=+1, w=0.24),
                dict(qid="C.FOL.3", type="freq_wk4", dir=-1, w=0.18),
                dict(qid="C.FOL.4", type="likert4", dir=+1, w=0.16),
                dict(qid="C.FOL.5", type="yesno", dir=+1, w=0.12),
            ],
            synergy=[dict(idx=[0, 1], coef=0.14)],
        )
    )
    C.append(
        dict(
            key="arginine",
            beta=(1.5, 1.5),
            noise=0.05,
            scale=dict(a=6.2, b=0.5),
            questions=[
                dict(qid="C.ARG.1", type="likert4", dir=+1, w=0.26),
                dict(qid="C.ARG.2", type="freq_wk4", dir=+1, w=0.20),
                dict(qid="C.ARG.3", type="likert4", dir=+1, w=0.18),
                dict(qid="C.ARG.4", type="likert4", dir=+1, w=0.18),
                dict(qid="C.ARG.5", type="likert4", dir=-1, w=0.18),
            ],
            synergy=[dict(idx=[0, 1], coef=0.12), dict(idx=[2, 3], coef=0.10)],
        )
    )
    C.append(
        dict(
            key="chondroitin",
            beta=(1.5, 1.5),
            noise=0.05,
            scale=dict(a=6.2, b=0.5),
            questions=[
                dict(qid="C.CHO.1", type="likert4", dir=+1, w=0.28),
                dict(qid="C.CHO.2", type="likert4", dir=+1, w=0.20),
                dict(qid="C.CHO.3", type="likert4", dir=+1, w=0.18),
                dict(qid="C.CHO.4", type="likert4", dir=+1, w=0.18),
                dict(qid="C.CHO.5", type="likert4", dir=+1, w=0.16),
            ],
            synergy=[dict(idx=[0, 2], coef=0.12)],
        )
    )
    C.append(
        dict(
            key="coq10",
            beta=(1.5, 1.5),
            noise=0.05,
            scale=dict(a=6.1, b=0.5),
            questions=[
                dict(qid="C.COQ10.1", type="yesno", dir=+1, w=0.26),
                dict(qid="C.COQ10.2", type="likert4", dir=+1, w=0.22),
                dict(qid="C.COQ10.3", type="likert4", dir=+1, w=0.18),
                dict(qid="C.COQ10.4", type="likert4", dir=+1, w=0.16),
                dict(qid="C.COQ10.5", type="likert4", dir=+1, w=0.18),
            ],
            synergy=[dict(idx=[0, 1], coef=0.12)],
        )
    )
    C.append(
        dict(
            key="collagen",
            beta=(1.5, 1.5),
            noise=0.05,
            scale=dict(a=6.0, b=0.5),
            questions=[
                dict(qid="C.COL.1", type="likert4", dir=+1, w=0.28),
                dict(qid="C.COL.2", type="likert4", dir=+1, w=0.16),
                dict(qid="C.COL.3", type="likert4", dir=+1, w=0.18),
                dict(qid="C.COL.4", type="likert4", dir=+1, w=0.18),
                dict(qid="C.COL.5", type="likert4", dir=+1, w=0.20),
            ],
            synergy=[dict(idx=[0, 4], coef=0.12), dict(idx=[0, 3], coef=0.10)],
        )
    )
    return C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="./data/c-section-v1")
    ap.add_argument("--rows", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--version", type=str, default="c-1.0")
    ap.add_argument("--schema", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    seed_all(args.seed)
    cats = catalog()
    all_rows = []
    for cat in cats:
        rows = []
        for i in range(args.rows):
            vals, label = gen_row(cat)
            rec = {
                "dataset_version": args.version,
                "scenario_id": f"S{args.seed:04d}-{cat['key']}-{i:04d}",
                "cat_key": cat["key"],
                "label_0_100": label,
            }
            for j, q in enumerate(cat["questions"]):
                rec[q["qid"]] = vals[j]
                rec[f"v{j+1}"] = (
                    float(vals[j]) if q["type"] == "yesno" else float(vals[j]) / 3.0
                )
            rows.append(rec)
        df = pd.DataFrame(rows)
        df.to_csv(
            os.path.join(args.out, f"C-{cat['key']}-{args.version}.csv"),
            index=False,
            encoding="utf-8",
        )
        all_rows.extend(rows)
    pd.DataFrame(all_rows).to_csv(
        os.path.join(args.out, f"C-section-{args.version}-all.csv"),
        index=False,
        encoding="utf-8",
    )
    if args.schema:
        meta = {
            c["key"]: [
                {"qid": q["qid"], "type": q["type"], "dir": q["dir"], "w": q["w"]}
                for q in c["questions"]
            ]
            for c in cats
        }
        with open(
            os.path.join(args.out, f"C-section-{args.version}-schema.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
