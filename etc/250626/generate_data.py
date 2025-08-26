import pandas as pd
import numpy as np

NUM_SAMPLES = 1000
QUESTIONS = [f"Q{i}" for i in range(1, 11)]
LABELS = [
    "비타민C",
    "칼슘",
    "마그네슘",
    "비타민D",
    "아연",
    "프로바이오틱스",
    "밀크씨슬",
    "오메가3",
    "멀티비타민",
    "차전자피 식이섬유",
    "철분",
    "엽산",
    "가르시니아",
    "콜라겐",
    "셀레늄",
    "루테인",
    "비타민A",
]

mapping = {
    1: ["비타민C", "멀티비타민", "철분", "오메가3", "마그네슘"],
    2: ["칼슘", "비타민D", "마그네슘", "콜라겐"],
    3: ["마그네슘", "오메가3", "비타민D", "프로바이오틱스"],
    4: ["프로바이오틱스", "차전자피 식이섬유", "밀크씨슬"],
    5: ["비타민C", "아연", "셀레늄", "비타민D"],
    6: ["콜라겐", "비타민A", "비타민C", "셀레늄"],
    7: ["루테인", "오메가3", "비타민A"],
    8: ["엽산", "철분", "멀티비타민"],
    9: ["오메가3", "멀티비타민", "셀레늄"],
    10: ["비타민C", "셀레늄", "루테인", "밀크씨슬"],
}


half = NUM_SAMPLES // 2
responses = np.zeros((NUM_SAMPLES, len(QUESTIONS)), dtype=int)

responses[:half] = np.random.randint(1, 6, size=(half, len(QUESTIONS)))

for i in range(half, NUM_SAMPLES):
    high_qs = np.random.choice(
        len(QUESTIONS), size=np.random.randint(1, 4), replace=False
    )
    resp = np.random.randint(1, 6, size=len(QUESTIONS))
    resp[high_qs] = np.random.randint(4, 6, size=len(high_qs))
    responses[i] = resp


label_idx = {lbl: j for j, lbl in enumerate(LABELS)}
labels = np.zeros((NUM_SAMPLES, len(LABELS)), dtype=np.float32)
counts = np.zeros_like(labels)
for i in range(NUM_SAMPLES):
    for q in range(len(QUESTIONS)):
        weight = (responses[i, q] - 1) / 4
        for cat in mapping[q + 1]:
            j = label_idx[cat]
            labels[i, j] += weight
            counts[i, j] += 1

labels = np.divide(labels, counts, out=np.zeros_like(labels), where=counts > 0)

df = pd.DataFrame(responses, columns=QUESTIONS)
for j, lbl in enumerate(LABELS):
    df[lbl] = labels[:, j]

df.to_csv("survey_labels.csv", index=False)
