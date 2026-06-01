# 발표 스크립트
## GLOW vs FLUX: Comparing Normalizing Flows and Flow Matching for Face Image Generation at 64×64

---

## Slide 1 — Title

안녕하세요. 저는 Younghyeok Kim입니다.
오늘 발표할 주제는 **"GLOW vs FLUX"** 로,
Normalizing Flow 모델과 Flow Matching 기반 Diffusion 모델을
64×64 해상도 얼굴 생성이라는 동일한 조건에서 비교한 실험입니다.

---

## Slide 2 — Motivation

최근 생성 모델 분야는 Diffusion 모델이 지배하고 있습니다.
하지만 Diffusion 모델은 샘플 하나를 만들 때 모델을 수십 번 반복 실행해야 합니다.
이를 NFE, Number of Function Evaluations라고 부릅니다.

반면 Normalizing Flow 계열의 GLOW는 **단 한 번의 순방향 연산**으로 샘플을 생성합니다.
그리고 **정확한 로그 가능도, 즉 NLL을 계산**할 수 있는데,
이는 Diffusion 모델이 수학적으로 제공할 수 없는 기능입니다.

그래서 저는 이런 질문을 던졌습니다.
*"같은 해상도에서 평가했을 때, 어떤 모델이 compute 대비 더 나은 품질을 제공하는가?"*

특히 FLUX.1은 512×512 이상의 고해상도 합성을 위해 설계된 모델입니다.
이 모델을 64×64로 강제했을 때 어떤 일이 벌어지는지가 핵심 질문입니다.

이 실험에서는 세 계열의 모델을 비교합니다.
GLOW, FLUX.1, 그리고 DDPM/DDIM입니다.
NFE가 1부터 100까지 8개 operating point에서 완전한 Pareto 프론티어를 구성했습니다.

---

## Slide 3 — Three Generative Paradigms

이 그림은 세 모델 계열이 noise를 data로 변환하는 방식을 시각화한 것입니다.

왼쪽부터 보면,
**Normalizing Flow**는 데이터 공간과 잠재 공간 사이에 완전한 역함수를 가진 bijection을 정의합니다.
격자가 변형되는 방식으로 표현되는데, 이 역함수가 존재하기 때문에 NLL 계산과 재구성이 가능합니다.

**Flow Matching**은 직선에 가까운 ODE 경로로 noise에서 data로 이동합니다.
FLUX.1이 이 방식을 사용하며, 경로가 직선에 가깝기 때문에 적은 NFE로도 어느 정도 작동합니다.

**DDPM**은 확률적 SDE로 지그재그 경로를 따르며,
**DDIM**은 같은 score function을 결정론적 ODE로 변환해 더 효율적입니다.

하단의 속성 비교표를 보면,
NLL 계산, 재구성, 보간은 오직 GLOW만 가능하다는 것을 알 수 있습니다.

---

## Slide 4 — Background: GLOW

GLOW는 2018년 NeurIPS에서 Kingma와 Dhariwal이 발표한 Normalizing Flow 모델입니다.

핵심 수식은 **변수 변환 공식**입니다.
데이터의 로그 가능도는 잠재 공간의 로그 가능도에 log-determinant 항을 더한 값으로 정확히 계산됩니다.

각 flow step은 세 가지 층으로 구성됩니다.
ActNorm은 데이터 기반 정규화,
Invertible 1×1 Convolution은 채널 순열을 학습하는 층,
Affine Coupling은 채널 절반을 이용해 나머지 절반의 scale과 shift를 예측하는 구조입니다.

저는 4 blocks × 32 flows, hidden dimension 512의 구성으로 약 76M 파라미터를 사용했습니다.

오른쪽 그래프는 학습 곡선입니다.
30,000 스텝 동안 NLL이 꾸준히 감소했으며, **30k 스텝에서도 아직 수렴하지 않았습니다.**
더 학습하면 추가 개선이 가능하다는 의미입니다.

학습 중 두 가지 안정화 조치가 필요했는데,
log-scale을 [-3, 3]으로 제한하고,
NaN/Inf 그래디언트가 발생하면 해당 스텝을 건너뛰는 guard를 추가했습니다.
이 없이는 약 300~500 스텝에서 발산이 발생했습니다.

---

## Slide 5 — Background: FLUX.1 and DDPM/DDIM

왼쪽은 FLUX.1입니다.
Flow Matching 기반으로 시간 의존 벡터 필드를 학습해 ODE를 통해 noise를 data로 변환합니다.
파라미터 수는 약 12B로 대규모 인터넷 이미지로 사전 학습되었으며,
schnell 버전은 NFE=4, dev 버전은 NFE=8 또는 28을 사용합니다.
중요한 점은 **설계 해상도가 512 이상**이라는 것입니다.
64×64에서는 VAE가 이미지를 8×8 feature map으로 줄이고, attention이 256 token만 처리합니다.

오른쪽은 DDPM과 DDIM입니다.
DDPM은 1000 스텝의 Markov chain을 학습합니다.
DDIM은 같은 score function으로부터 **재학습 없이 결정론적 ODE**를 유도해
NFE를 10~50으로 크게 줄일 수 있습니다.
저는 `google/ddpm-celebahq-256` 체크포인트를 사용했습니다.
**CelebA-HQ 얼굴 이미지로 학습된 모델**이기 때문에 64×64 얼굴 생성에 잘 전이됩니다.

---

## Slide 6 — Main Results: Full Pareto Frontier

이것이 이번 실험의 핵심 결과 테이블입니다.
8개 operating point 전체의 FID를 비교했습니다.

먼저 눈에 띄는 점은 **DDIM-50이 FID 61.78로 전체 최고 성능**입니다. 노란색으로 표시했습니다.

초록색으로 표시된 GLOW는 NLL -4.61 bits/dim을 기록한 유일한 모델입니다.
다른 모델들은 NLL을 계산할 수 없습니다.

분홍색으로 표시된 **FLUX.1-schnell이 주목할 만한데,
FID 184.94로 GLOW의 183.35보다 나쁘면서 9배 느립니다.**
즉, 어떤 기준으로도 GLOW에게 지는 모델입니다.

DDIM 계열은 NFE 10, 20, 50에서 일관되게 FLUX.1을 압도합니다.
FLUX.1-dev는 28 NFE에서 FID 117.87이지만,
DDIM-10이 10 NFE에서 이미 68.75를 달성한다는 점에서 경쟁력이 약합니다.

---

## Slide 7 — Compute-Quality Pareto Frontier

이 그래프가 Pareto 프론티어를 시각화한 것입니다.
x축이 NFE(로그 스케일), y축이 FID입니다.

오른쪽에 네 가지 구간을 정리했습니다.

첫 번째는 **지연 최소화 구간**입니다.
NFE=1인 GLOW가 유일한 선택지이며, 동시에 정확한 NLL과 재구성이 가능합니다.

두 번째는 **효율 최적 구간**입니다.
DDIM-10에서 50까지가 이 구간에 해당하며,
같은 NFE에서 FLUX.1보다 훨씬 낮은 FID를 달성합니다.
DDIM-50의 FID 61.78이 전체 최고입니다.

세 번째는 **고compute 품질 구간**입니다.
FLUX.1-dev가 NFE 28에서 117.87을 달성합니다.
텍스트 조건 생성이나 더 높은 NFE 예산이 있을 때 의미 있습니다.

그리고 마지막으로 **Pareto-suboptimal 모델**이 FLUX.1-schnell입니다.
GLOW와 DDIM-10 양쪽에 모두 지배당하는, 어떤 tradeoff에서도 최적이 아닌 모델입니다.

---

## Slide 8 — Generated Samples

실제 생성된 샘플을 보겠습니다.
위부터 GLOW, FLUX.1-schnell, FLUX.1-dev 순입니다.

GLOW 샘플은 전체적으로 일관된 얼굴 구조를 보여주지만,
64×64 저해상도 특성상 texture가 다소 부드럽습니다.

FLUX.1-schnell은 다양한 포즈와 조명을 생성하지만,
64×64에서는 간혹 얼굴이 아닌 구조가 나타나는 hallucination이 관찰됩니다.
설계 해상도가 아닌 환경에서 발생하는 현상입니다.

FLUX.1-dev는 28 NFE를 투입한 만큼 시각적 품질이 가장 좋습니다.
FID 117.87이 이를 수치로 뒷받침합니다.

---

## Slide 9 — GLOW-Exclusive (1): Likelihood & Reconstruction

이 슬라이드에서는 Diffusion 모델이 구조적으로 제공할 수 없는 GLOW의 독점 기능 두 가지를 소개합니다.

먼저 **정확한 로그 가능도**입니다.
GLOW는 검증 세트에서 NLL -4.61 bits/dim을 달성했습니다.
FLUX.1이나 DDPM/DDIM은 정확한 NLL 계산이 불가능합니다.
ELBO 근사를 쓰더라도 계산 비용이 매우 크고 부정확합니다.

이 능력은 **이상 탐지, 무손실 압축, 밀도 추정** 같은 응용에서 결정적입니다.
Diffusion 모델은 아무리 FID가 좋아도 이 용도에는 쓸 수 없습니다.

오른쪽은 **인코딩 후 디코딩 재구성** 실험입니다.
실제 얼굴 200장을 GLOW로 인코딩한 뒤 다시 디코딩했습니다.
PSNR 24.24 dB, SSIM 0.977을 달성했습니다.
SSIM이 0.977이라는 것은 얼굴의 구조, 즉 표정, 포즈, 위치가 거의 완벽하게 보존된다는 의미입니다.
PSNR이 무한대가 아닌 이유는 128개 층을 순차적으로 통과하면서 쌓이는 부동소수점 오차 때문입니다.

---

## Slide 10 — GLOW-Exclusive (2): Interpolation & OOD Detection

두 번째로 **잠재 공간 보간**입니다.
두 이미지 A와 B를 각각 잠재 벡터 z_A, z_B로 인코딩한 뒤
alpha를 0에서 1로 선형 증가시키면서 중간 벡터를 디코딩합니다.

보이시는 것처럼 A에서 B로 자연스럽게 semantic이 전환됩니다.
이 실험은 역함수가 정확히 존재해야만 가능합니다.
FLUX.1이나 DDPM은 역함수가 없으므로 이 실험 자체를 수행할 수 없습니다.

오른쪽은 **OOD 탐지** 실험입니다.
GLOW의 NLL로 분포 내 이미지와 외부 이미지를 구분할 수 있는지 테스트했습니다.

결과를 보면,
FFHQ 얼굴은 NLL -4.61,
단색 이미지는 -6.85,
랜덤 노이즈는 NaN, 즉 수치 오버플로우가 발생했습니다.

흥미로운 점은 단색 이미지가 얼굴보다 **더 낮은 NLL**을 보인다는 것입니다.
이는 NF의 알려진 한계인 "likelihood != 지각 품질" 현상입니다.
단순한 구조의 이미지에 모델이 더 높은 확률을 부여하는 경우가 발생합니다.

---

## Slide 11 — 2D Transport Visualization

이 슬라이드는 실제 64×64 이미지가 아니라
**2D two-moons 데이터셋**에서 각 모델이 점을 어떻게 이동시키는지 시각화한 것입니다.

왼쪽 그림에서,
NF는 격자 변형, 즉 비선형적이지만 bijective한 재배열을 보여주고,
Flow Matching은 거의 직선에 가까운 경로로 이동하며,
DDPM은 지그재그 확률적 경로를,
DDIM은 결정론적이고 부드러운 ODE 경로를 따릅니다.

오른쪽은 각 모델의 중간 분포를 6개 스냅샷으로 보여줍니다.
색상은 noise에서 data까지 각 점의 identity를 추적합니다.
NF는 처음부터 구조가 잡히며, DDPM은 점진적으로 형성됩니다.

---

## Slide 12 — Discussion

실험 결과를 해석하겠습니다.

**왜 FLUX.1-schnell이 GLOW를 이기지 못했는가?**
FLUX.1의 VAE는 64×64 이미지를 8×8 feature map으로 압축합니다.
즉 attention이 처리하는 token이 256개밖에 안 됩니다.
FLUX.1의 positional encoding은 512 이상 해상도에 맞춰져 있어,
이 해상도에서는 spatial reasoning이 크게 저하됩니다.

**왜 DDIM이 중간 구간을 지배하는가?**
`google/ddpm-celebahq-256`은 얼굴 이미지로만 학습된 도메인 특화 모델입니다.
256에서 생성하고 bicubic downsampling으로 64로 줄이면 품질이 잘 보존됩니다.
결정론적 DDIM ODE가 확률적 DDPM보다 샘플 효율적이라는 것도 확인됩니다.

**실용적 선택 가이드:**
NFE=1이 필요하거나 정확한 밀도가 필요하면 GLOW,
최고 FID를 moderate compute로 얻으려면 DDIM-50,
텍스트 조건 생성이나 고해상도가 필요하면 FLUX.1-dev입니다.

---

## Slide 13 — Conclusion

마지막으로 핵심 결론 네 가지를 정리하겠습니다.

첫째, **GLOW는 NFE=1에서 FLUX.1-schnell과 동등한 FID**를 달성합니다.
76M 파라미터 모델이 160배 큰 12B 모델과 대등하다는 것은,
저해상도에서는 거대 사전학습 모델의 이점이 사라진다는 것을 보여줍니다.

둘째, **DDIM-50이 전체 최고 FID 61.78**을 달성합니다.
CelebA-HQ 사전학습 모델이 64×64 얼굴 이미지에 잘 전이되었고,
결정론적 ODE가 효율적으로 작동한 결과입니다.

셋째, **FLUX.1-schnell은 Pareto-suboptimal**입니다.
GLOW와 DDIM 양쪽 모두에게 지배당하며,
어떤 operating point에서도 이 둘 중 하나보다 나은 tradeoff를 제공하지 못합니다.

넷째, **GLOW는 독점적 기능을 유지합니다.**
정확한 NLL, 재구성, 잠재 공간 보간은 Diffusion 모델이 구조적으로 제공할 수 없는 기능입니다.
밀도가 필요한 응용에서 GLOW의 가치는 FID 숫자만으로 평가할 수 없습니다.

결론적으로, **모델 선택은 application-dependent**합니다.
거대 사전학습 모델의 우위는 해상도에 따라 달라지며,
저해상도 도메인 특화 태스크에서는 작은 task-specific 모델이 충분히 경쟁력을 가집니다.

이상으로 발표를 마치겠습니다. 감사합니다.

---

## 예상 질문 & 답변

**Q1. GLOW FID가 183인데, 실용적으로 사용 가능한 수준인가?**
> FID 183은 사람이 보기에 인식 가능한 아티팩트가 있는 수준입니다. 하지만 본 실험의 목적은 최고 품질의 생성보다는 두 패러다임의 상대적 비교이며, GLOW는 30k 스텝에서 아직 수렴 전이라 더 학습하면 개선 가능합니다. 실제로 학습 곡선이 30k에서도 하강 중입니다.

**Q2. DDIM이 FLUX.1을 이기는 것이 공평한 비교인가? 학습 데이터가 다르다.**
> 맞습니다. DDIM은 CelebA-HQ 얼굴 이미지로, FLUX.1은 범용 대규모 데이터로 학습됐습니다. 이 차이 자체가 핵심 발견 중 하나입니다. 동일 해상도 도메인에 특화된 사전학습 모델이 범용 거대 모델보다 해당 도메인에서 우위를 가집니다. 실무 관점에서 모델 선택 시 반드시 고려해야 할 요소입니다.

**Q3. GLOW의 NLL이 음수인데, 어떻게 해석하는가?**
> NLL은 bits/dim 단위로, 균일 분포(uniform)의 경우 0 bits/dim입니다. 음수 NLL은 모델이 실제 이미지에 균일 분포보다 더 높은 확률을 부여한다는 의미이며, 이는 학습이 잘 된 것을 나타냅니다. 완벽한 모델은 데이터 엔트로피와 같은 값에 수렴합니다.

**Q4. FLUX.1-schnell이 왜 Pareto-suboptimal인가?**
> NFE=4에서 FID 184.94인데, NFE=1인 GLOW가 FID 183.35로 더 좋고 9배 빠릅니다. 또한 NFE=10인 DDIM-10이 FID 68.75로 훨씬 낮습니다. schnell은 속도도 품질도 어느 쪽으로도 최적이 아니라는 것이 Pareto-suboptimal의 의미입니다.
