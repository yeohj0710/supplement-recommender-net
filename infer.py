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


def get_user_responses():
    """콘솔에서 1~5 정수 응답 10개를 순차적으로 입력받아 numpy 배열로 반환."""
    resp = []
    print("설문에 답해 주세요 (1~5 사이의 숫자로 입력):")
    for q in QUESTIONS:
        while True:
            try:
                val = int(input(f"{q} (1-5): ").strip())
                if 1 <= val <= 5:
                    resp.append(val)
                    break
                else:
                    print("▶ 1부터 5까지만 입력 가능합니다.")
            except ValueError:
                print("▶ 숫자를 입력해 주세요.")
    return np.array(resp, dtype=np.float32).reshape(1, -1)


def load_model_session(path):
    """ONNX 세션을 한 번만 만들고 재활용하기 위해 분리."""
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    return sess, sess.get_inputs()[0].name, sess.get_outputs()[0].name


def predict_top_categories(session, input_name, output_name, responses, top_k=TOP_K):
    """모델 추론 → sigmoid → 상위 top_k 라벨 반환."""
    logits = session.run([output_name], {input_name: responses})[0][0]
    probs = 1 / (1 + np.exp(-logits))

    top_idxs = np.argsort(probs)[::-1][:top_k]
    return [(LABELS[i], probs[i]) for i in top_idxs]


def main():
    responses = get_user_responses()
    session, in_name, out_name = load_model_session(MODEL_PATH)
    top3 = predict_top_categories(session, in_name, out_name, responses, TOP_K)

    print("\n📋 추천 영양제 카테고리 (중요도 순):")
    for cat, score in top3:
        print(f" - {cat}: {score:.2f}")


if __name__ == "__main__":
    main()
