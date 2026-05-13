# Cold Embedding Retraining for LightGCN

> 네이버 부스트캠프 AI Tech 7기 기업 해커톤 (TVING) **1위** 수상 프로젝트
>
> Cold start 문제를 해결하기 위해, 완전히 학습된 LightGCN에서 cold user의 임베딩만 재초기화하여 재학습하는 방법론을 구현합니다.

## Background

### Cold User 문제

MovieLens 20M (138,493 users, 26,744 items, 약 2천만 건의 rating, sparsity 99.52%) 데이터에서 LightGCN 등 여러 모델로 실험한 결과, **상호작용 수가 20 이하인 유저(cold user)에서 성능이 뚜렷하게 저하**됨을 확인했습니다.

### 왜 단순 Early Stopping으로는 부족한가

LightGCN을 전체 유저 성능이 최대가 될 때까지 학습시키면 **학습 후반부로 갈수록 cold user의 성능이 급격하게 하락**합니다. Cold user 성능이 가장 높은 epoch에서 조기 종료하면 이 하락을 피할 수 있지만, 그 시점의 cold user 임베딩 역시 최적이 아닙니다.

Cold user의 임베딩 입장에서, 나머지 warm user와 item의 임베딩은 일종의 **학습 환경**입니다. Cold user 성능이 최고인 시점에는 이 학습 환경(다른 임베딩들)이 아직 충분히 수렴하지 않은 상태이므로, cold user는 최적의 환경에서 학습할 기회를 갖지 못합니다.

### 제안 방법론

모델이 완전히 수렴한 뒤, cold user의 임베딩만 정규분포로 재초기화하여 소수 epoch 재학습합니다. Warm user와 item 임베딩은 이미 최적 상태이므로, cold user는 **최적의 학습 환경에서 처음부터 다시** 표현을 학습하게 됩니다.

```
[1단계] 전체 모델 학습 (LightGCN, early stopping)
         → warm user / item 임베딩이 최적 상태로 수렴
         ↓
[2단계] cold user (상호작용 수 ≤ refresh_threshold) 임베딩만 N(0, 0.1)로 재초기화
         ↓
[3단계] cold user 임베딩만 소수 epoch 재학습
         → 최적화된 학습 환경 위에서 cold user 표현 개선
```

### 평가 방법

실제 cold start 상황을 모사하기 위해 **pseudo-cold user** 기반 평가를 사용합니다. 20-core 필터링 후 유저를 Train 8 : Valid 1 : Test 1로 분할하고, Valid/Test 유저의 시간순 첫 K개(1~15개) 상호작용을 학습 데이터로, 이후 N개(N=5) 상호작용을 ground truth로 활용합니다. 이를 통해 평가 샘플 수를 일정하게 유지하여 지표 신뢰도를 확보합니다.

## Results (ML-20M, NDCG@10)

| User group | Original (Best epoch) | Original (Last epoch) | Retrained (1 epoch) |
|:----------:|:---------------------:|:---------------------:|:-------------------:|
| 1-shot     | 0.1579                | 0.1017                | **0.1733**          |
| 3-shot     | 0.1966                | 0.1749                | **0.2152**          |
| 5-shot     | 0.2174                | 0.2026                | **0.2300**          |
| all        | 0.2154                | 0.2154                | **0.2220**          |

재학습 후 cold user 성능이 기존 모든 epoch의 최고 성능보다도 높아집니다.

## Project Structure

```
.
├── main.py              # 진입점 (학습 / 재학습 모드 선택)
├── requirements.txt
├── data/
│   ├── ml-1m/
│   │   ├── train.csv
│   │   └── val.csv
│   └── ml-20m/
│       ├── train1.csv
│       ├── train2.csv
│       └── val.csv
├── saved/               # 학습된 모델 체크포인트
│   └── ml-1m_best_model.pt
└── src/
    ├── dataset.py       # user-item matrix, normalized adjacency matrix
    ├── model.py         # LightGCN
    ├── sampler.py       # BPR negative sampling (C++ 가속)
    ├── sampler_cpp.cpp  # pybind11 C++ 확장
    ├── trainer.py       # 학습 루프 + refresh_cold_users()
    └── utils.py         # Recall, NDCG, BPR loss
```

## Installation

Python 3.10 이상이 필요합니다.

```bash
pip install -r requirements.txt
```

`sampler_cpp.cpp`는 `cppimport`를 통해 최초 실행 시 자동으로 컴파일됩니다. `pybind11`과 C++ 컴파일러(`g++`)가 설치되어 있어야 합니다.

## Usage

### 1단계: 전체 모델 학습

```bash
python main.py --dataset ml-1m
python main.py --dataset ml-20m
```

학습이 완료되면 `saved/{dataset}_best_model.pt`에 체크포인트가 저장됩니다.

### 2단계: Cold user 임베딩 재학습

```bash
python main.py --dataset ml-1m --retrain
python main.py --dataset ml-20m --retrain
```

저장된 모델을 불러와 cold user의 임베딩을 재초기화한 뒤 재학습합니다.

### 주요 하이퍼파라미터

| 인자 | 기본값 | 설명 |
|------|--------|------|
| `--dataset` | `ml-1m` | 사용할 데이터셋 (`ml-1m` / `ml-20m`) |
| `--epochs` | `100` | 최대 학습 epoch 수 |
| `--batch_size` | `2048` | 배치 크기 |
| `--lr` | `0.001` | 학습률 |
| `--patience` | `3` | early stopping patience |
| `--emb_size` | `64` | 임베딩 차원 수 |
| `--num_layers` | `2` | LightGCN graph convolution 레이어 수 |
| `--refresh_threshold` | `5` | cold user 기준 (상호작용 수 이하) |
| `--retrain_epochs` | `3` | 재학습 epoch 수 |


## Requirements

- Python >= 3.10
- PyTorch 2.5.0
- NumPy 1.26.4
- SciPy 1.14.1
- pandas 2.2.3
- cppimport 22.8.2
- pybind11_global 2.13.6
- tqdm 4.66.5
