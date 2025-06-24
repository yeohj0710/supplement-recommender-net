import pandas as pd
import numpy as np
import os


NUM_SAMPLES = 100
THRESHOLD = 4

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


responses = np.random.randint(1, 6, size=(NUM_SAMPLES, len(QUESTIONS)))


labels = np.zeros((NUM_SAMPLES, len(LABELS)), dtype=int)


for i in range(NUM_SAMPLES):
    for q_idx, q_val in enumerate(responses[i]):
        if q_val >= THRESHOLD:
            for cat in mapping[q_idx + 1]:
                label_idx = LABELS.index(cat)
                labels[i, label_idx] = 1


df = pd.DataFrame(responses, columns=QUESTIONS)
for idx, lbl in enumerate(LABELS):
    df[lbl] = labels[:, idx]

csv_path = "survey_labels.csv"
df.to_csv(csv_path, index=False)
