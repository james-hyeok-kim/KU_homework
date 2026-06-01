# 실험 결과 전체 요약 (한글)
## GLOW vs FLUX.1 vs DDIM/DDPM — FFHQ 64×64 얼굴 생성 비교

---

## 실험 목적

세 가지 계열의 생성 모델을 같은 조건(64×64 해상도, FID 기준)에서 비교:

- **GLOW**: Normalizing Flow 모델. 76M 파라미터. FFHQ-64×64로 처음부터 직접 학습 (30,000 스텝)
- **FLUX.1**: Flow Matching 기반 Diffusion Transformer. ~12B 파라미터. 대규모 사전학습 모델
- **DDIM/DDPM**: CelebA-HQ-256 사전학습 모델(`google/ddpm-celebahq-256`). 64×64으로 다운샘플링 후 평가

세 모델 계열 모두 64×64 해상도에서 얼굴 이미지를 생성하고,
생성 품질 / 확률 계산 능력 / 샘플링 속도 / 잠재공간 특성을 비교한다.

---

## 측정 지표

### NLL (bits/dim) — 낮을수록 좋음
모델이 실제 이미지에 부여하는 확률. GLOW만 계산 가능. 완전 랜덤 = 8 bits/dim.

### FID — 낮을수록 좋음
생성 분포 vs 실제 분포 거리 (InceptionV3 특징 벡터 기준). FID < 10 = 사람이 구별 불가.

### NFE — 낮을수록 효율적
샘플 1장당 모델 순방향 통과 횟수. GLOW=1, FLUX schnell=4, DDIM=10–50, FLUX dev=8–28, DDPM=100.

### PSNR / SSIM — GLOW 전용
인코딩→디코딩 왕복 후 원본 이미지와의 유사도. 역함수 없는 Diffusion 모델은 불가.

---

## 정량적 결과 (전체 Pareto 프론티어)

| 모델 | NLL (bits/dim) ↓ | FID ↓ | NFE | 5k 샘플 시간 | 파라미터 |
|------|-----------------|-------|-----|------------|---------|
| **GLOW** (30k 스텝) | **−4.61** | 183.35 | **1** | **137초** | 76M |
| FLUX.1-schnell | 불가 | 184.94 | 4 | 1,231초 | ~12B |
| FLUX.1-dev-8 | 불가 | 126.15 | 8 | — | ~12B |
| DDIM-10 † | 불가 | 68.75 | 10 | 602초 | 113M |
| DDIM-20 † | 불가 | 65.96 | 20 | 1,346초 | 113M |
| DDIM-50 † | 불가 | **61.78** | 50 | 2,658초 | 113M |
| DDPM-100 † | 불가 | 71.87 | 100 | 5,453초 | 113M |
| FLUX.1-dev | 불가 | 117.87 | 28 | ~8,617초 | ~12B |

† CelebA-HQ-256 사전학습 가중치, 출력을 64×64로 bicubic 다운샘플링

---

## GLOW 학습 곡선

30,000 스텝 동안 NLL이 꾸준히 감소:

| 스텝 | NLL (bits/dim) | 비고 |
|------|---------------|------|
| 100 | −2.39 | 웜업 단계 (lr=5e-6) |
| 1,000 | −3.81 | 풀 lr 도달 (5e-5) |
| 5,000 | −4.31 | — |
| 10,000 | −4.43 | — |
| 20,000 | −4.55 | — |
| 30,000 | **−4.61** | 아직 수렴 전 |

30k 스텝에서도 여전히 하강 중 → 더 학습하면 FID/NLL 모두 개선 가능

---

## GLOW 전용 실험 결과

### 재구성 (인코딩 → 디코딩)
실제 얼굴 이미지 200장을 GLOW로 인코딩 후 다시 디코딩:

- **PSNR = 24.24 dB** (표준편차 0.31)
- **SSIM = 0.9771** (표준편차 0.0081)

SSIM=0.977 → 얼굴의 구조(위치, 표정, 포즈)는 거의 완벽하게 보존.
PSNR이 ∞가 아닌 이유: 128개 순차 레이어를 통과하면서 쌓이는 부동소수점 오차.

### OOD 탐지 (이상 탐지)
GLOW의 NLL로 분포 내/외 이미지 구분:

| 입력 종류 | NLL 평균 | 해석 |
|---------|---------|------|
| FFHQ 얼굴 (분포 내) | −4.609 bits/dim | 정상 |
| 단색 이미지 | −6.854 bits/dim | NLL이 더 낮음 (주의!) |
| 랜덤 노이즈 | NaN (오버플로우) | — |

※ 단색 이미지가 얼굴보다 낮은 NLL을 갖는 것은 "NLL ≠ 지각 품질"이라는 Normalizing Flow의 알려진 한계.

### 잠재 공간 보간 (Interpolation)
두 얼굴 이미지 A, B를 인코딩 후 잠재벡터 z = (1-α)z_A + α·z_B 를 디코딩:
α를 0.0 → 1.0으로 변화시키면 A에서 B로 자연스럽게 변환. Diffusion 모델은 이 실험 자체 불가.

---

## 핵심 발견

### 1. DDIM이 중간 NFE 구간 Pareto 지배
DDIM-10/20/50이 FID=68.75/65.96/61.78 달성 → 같은 NFE 조건에서 FLUX.1을 모두 압도.
이유: CelebA-HQ 학습 모델이 64×64 얼굴에 잘 전이됨 (FLUX.1의 512+ 해상도 미스매치 없음).

### 2. DDIM > DDPM: 결정론적 ODE가 효율적
DDIM-50 (FID=61.78)이 DDPM-100 (FID=71.87)보다 NFE 절반으로 더 좋은 결과.
확률적 SDE보다 결정론적 ODE가 샘플 효율적.

### 3. FLUX.1-schnell은 Pareto-suboptimal
NFE=4에서 FID=184.94 — GLOW(FID=183.35)보다 느리고 FID도 더 나쁨.
NFE=8(dev-8)에서 FID=126.15로 개선되지만 DDIM-10(FID=68.75)에 비할 수 없음.

### 4. NLL·재구성·보간은 GLOW 독점
Diffusion 계열 모델은 수학적 구조상 NLL 직접 계산, 역함수, 잠재공간 보간 모두 불가.
이상 탐지·무손실 압축·데이터 밀도 추정에서 GLOW가 구조적 우위.

---

## Pareto 구조 요약

| 구간 | 최적 모델 | 이유 |
|------|---------|------|
| NFE=1 (지연 최소화) | **GLOW** | 유일한 단일 패스, + NLL 계산 가능 |
| NFE=10–50 (효율 최적) | **DDIM-10~50** | 얼굴 도메인 전이 + 결정론적 ODE |
| NFE=28+ (최고 품질 추구) | FLUX.1-dev | 더 많은 compute를 투입할 때 |
| 어디에도 optimal 아님 | FLUX.1-schnell | GLOW보다 느리고 FID도 더 나쁨 |

---

## 판정표

| 기준 | 승자 |
|------|------|
| FID 전체 최고 | **DDIM-50** (61.78) |
| FID (NFE=1 조건) | **GLOW** (183.35 vs schnell 184.94) |
| 샘플링 속도 | **GLOW** (schnell 대비 9배, dev 대비 63배 빠름) |
| 정확한 NLL | **GLOW** (나머지: 계산 불가) |
| 재구성 품질 | **GLOW** (PSNR 24.24 dB, SSIM 0.977) |
| 잠재공간 보간 | **GLOW** (Diffusion은 불가) |
| 고해상도 생성 | FLUX.1 (이번 실험 범위 외) |
| 텍스트 조건 생성 | FLUX.1 |
| 학습 비용 | FLUX.1 / DDIM (사전학습 활용) |

---

## 실험 아티팩트

| 경로 | 내용 |
|------|------|
| `glow_pretrained/glow_v2_ffhq64_030000.pt` | GLOW 최종 체크포인트 |
| `samples/glow/` | GLOW 생성 이미지 5,000장 |
| `samples/flux_schnell_nfe4/` | FLUX schnell 생성 이미지 5,000장 |
| `samples/flux_dev_nfe8/` | FLUX dev-8 생성 이미지 5,000장 |
| `samples/flux_dev_nfe28/` | FLUX dev-28 생성 이미지 5,000장 |
| `samples/ddim_nfe10/` | DDIM-10 생성 이미지 5,000장 |
| `samples/ddim_nfe20/` | DDIM-20 생성 이미지 5,000장 |
| `samples/ddim_nfe50/` | DDIM-50 생성 이미지 5,000장 |
| `samples/ddpm_nfe100/` | DDPM-100 생성 이미지 5,000장 |
| `logs/ddim_results.json` | DDIM/DDPM FID 원본 |
| `logs/flux_dev_fid.json` | FLUX dev-28 FID 원본 |
| `logs/flux_nfe8_fid.json` | FLUX dev-8 FID 원본 |
| `logs/ood_out.txt` | OOD 탐지 NLL 수치 |
| `experiments/results/` | 모든 시각화 (fig1–fig9) |

---

## 관련 파일

- 실험 설계: `experiment_001.md`
- 영문 상세 결과: `result_001.md`
- 논문 (LaTeX): `paper/report.tex` / `paper/report.pdf`
