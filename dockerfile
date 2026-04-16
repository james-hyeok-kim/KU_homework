# 1. 베이스 이미지 (중복 제거)
FROM nvcr.io/nvidia/pytorch:24.12-py3

# 2. 사용자 설정 (UID/GID는 빌드 시 스크립트에서 주입됨)
ARG UID=1000
ARG GID=1000
ARG MYNAME=jameskimh

USER root

# [필살기] 기존 NVM 삭제
RUN rm -rf /usr/local/nvm

# 3. 필수 도구 설치 (Node.js 20, Git, wget)
RUN apt-get update && apt-get install -y curl gnupg graphviz git wget \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# 4. Claude Code 설치
RUN npm install -g @anthropic-ai/claude-code

# 🔥 4-1. [드라이버 호환성] Host CUDA 12.2 대응 (cu121)
RUN pip uninstall -y torch torchvision torchaudio && \
    pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 🔥 5. [긴급 수리] NumPy 1.x 고정 및 OpenCV 설치
# NumPy 2.x 충돌을 막기 위해 싹 지우고 순서대로 설치합니다.
RUN pip uninstall -y numpy opencv-python opencv-python-headless opencv-contrib-python

# 1) NumPy 1.x 선설치
RUN pip install --no-cache-dir "numpy<2.0" "pyarrow>=14.0.1,<15.0.0" "datasets>=2.14.0,<3.0.0" requests

# 2) OpenCV 및 기타 라이브러리 설치
RUN pip install --no-cache-dir opencv-python-headless \
    ipykernel jupyter transformers>=4.40.0 diffusers>=0.29.0 \
    seaborn torch_fidelity torchinfo torchviz torchview torchmetrics[multimodal] \
    accelerate tqdm tiktoken ftfy scikit-image lpips sentencepiece fairscale image-reward

# # 3) LeRobot 설치 (의존성 충돌 방지를 위해 마지막 즈음에 설치)
# RUN pip install --no-cache-dir --upgrade "lerobot[pi] @ git+https://github.com/huggingface/lerobot.git"

# 4) 버전 확인 사살
RUN pip install --no-cache-dir "huggingface_hub==0.25.2" "numpy<2.0"

# 6. 환경 설정
ENV PATH="/usr/bin:/usr/local/bin:$PATH"
ENV TORCH_CUDNN_V8_API_ENABLED=1
RUN mkdir -p /app && chown -R $UID:$GID /app
ENV HOME="/home/${MYNAME}"

# 사용자 생성 및 전환
RUN groupadd -g $GID $MYNAME || true
RUN useradd -u $UID -m $MYNAME -g $GID || true
USER $MYNAME

# Git 설정
ENV GIT_AUTHOR_NAME="james-hyeok-kim"
ENV GIT_AUTHOR_EMAIL="younghyeok25@gmail.com"
ENV GIT_COMMITTER_NAME="james-hyeok-kim"
ENV GIT_COMMITTER_EMAIL="younghyeok25@gmail.com"

# 🔥 GitHub 토큰 연동 (기본값 삭제하여 보안 강화)
ARG GITHUB_TOKEN
RUN git config --global url."https://james-hyeok-kim:${GITHUB_TOKEN}@github.com/".insteadOf "https://github.com/"

WORKDIR /app