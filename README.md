# 영양제 추천 AI 모델의 구현

이 repository는 설문 기반 맞춤형 영양제 추천 pipeline을 다룹니다.
사용자의 10개의 문항(Q1~Q10)에 대한 답변(리커트 척도)으로, 17개 영양제 성분 카테고리의 추천 점수를 계산하고, 웹에서 바로 구동할 수 있도록 구성하였습니다.

---

## 개발 과정 및 원리 설명

1. **Data Generation**

   - `generate_data.py`로 1,000개 샘플을 생성하였습니다.
   - 10개의 문항에 대해 1~5 점수를 랜덤 또는 일부 high response 방식으로 할당하였습니다.
   - 문항별 연관 카테고리(mapping)를 활용하여 응답 점수를 0~1 범위의 continuous label로 변환하였습니다.
   - 결과는 `survey_labels.csv`에 저장되어 있습니다.

2. **Model Training**

   - PyTorch 기반 ImprovedMLP (Multi-Layer Perceptron)를 사용하였습니다.
   - Input: 10차원 설문 응답, Output: 17차원 추천 score
   - Loss로 `BCEWithLogitsLoss`, Optimizer로 `AdamW`, learning rate scheduler와 early stopping을 적용하였습니다.
   - 최적화된 모델을 `torch.onnx.export`로 `survey_model.onnx` 형식으로 내보냈습니다.

3. **ONNX Export & Inference**

   - ONNX opset_version=11, dynamic axes 설정으로 유연한 배치 처리를 지원합니다.
   - `infer.py`에서 ONNX Runtime을 이용하여
     - logits → temperature scaling (T=2.0) → sigmoid → threshold(0.3) 순으로 확률을 계산하빈다.
     - 임계값 이상인 top categories를 추출하여 추천 결과를 제공합니다.

4. **Web Integration & Deployment**
   - Next.js 앱과 `onnxruntime-web`을 결합하여 브라우저 환경에서 직접 inference를 수행하였습니다.
   - Vercel에 배포하여 Serverless 환경에서도 딥러닝 모델을 안정적으로 서비스 가능합니다.

---

## 요약 정리

- **설문 → Continuous Label**  
  다중 클래스가 아니라, 응답 강도에 따라 여러 카테고리를 동시에 추천할 수 있는 유연한 구조입니다.
- **경량 MLP + ONNX**  
  모델 크기와 연산량을 최소화하여 브라우저에서도 부담 없이 동작하도록 설계하였습니다.
- **End-to-End Pipeline**  
  데이터 생성 → 학습 → ONNX 변환 → Next.js 통합 → Vercel 배포까지 한 번에 재현 가능한 workflow입니다.

이를 통해 개인화된 설문형 recommendation AI를 구현하고 웹에서 서비스를 구동할 수 있습니다.
