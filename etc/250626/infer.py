import numpy as np
import onnxruntime as ort


QUESTIONS = [
    "1) 최근에 쉽게 피로를 느끼고 에너지가 부족하다",
    "2) 뼈나 관절 건강이 걱정된다",
    "3) 스트레스나 불안, 수면 질이 좋지 않다",
    "4) 소화가 잘 안 되고, 변비·복통이 자주 있다",
    "5) 면역력이 약해 자주 감기에 걸린다",
    "6) 피부·모발·손톱 건강이 고민된다",
    "7) 눈이 쉽게 피로하고, 시야가 뿌옇다",
    "8) 임신 준비 중이거나 임신 중이다",
    "9) 심혈관 건강(고지혈·혈압)이 걱정된다",
    "10) 항산화·노화 방지가 필요하다",
]

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

MODEL_PATH = "survey_model.onnx"
TOP_K = 17
T = 2.0
THRESHOLD = 0.3


def get_user_responses():
    resp = []
    for q in QUESTIONS:
        val = int(input(f"{q} (1-5): "))
        resp.append(val)
    return np.array(resp, dtype=np.float32).reshape(1, -1)


def load_model_session(path):
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    return sess, sess.get_inputs()[0].name, sess.get_outputs()[0].name


def predict_top_categories(session, in_name, out_name, responses, top_k=TOP_K):
    logits = session.run([out_name], {in_name: responses})[0][0]
    probs = 1 / (1 + np.exp(-logits / T))
    idxs = np.where(probs >= THRESHOLD)[0]
    idxs = idxs[np.argsort(probs[idxs])[::-1]][:top_k]
    return [(LABELS[i], probs[i]) for i in idxs]


def main():
    resp = get_user_responses()
    sess, in_name, out_name = load_model_session(MODEL_PATH)
    preds = predict_top_categories(sess, in_name, out_name, resp)
    for cat, score in preds:
        print(f" - {cat}: {score:.2f}")


if __name__ == "__main__":
    main()
