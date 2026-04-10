#!/bin/bash

# 0. 환경 변수 파일(~/.env) 불러오기
# GITHUB_TOKEN, HF_TOKEN, WANDB_TOKEN 등을 안전하게 로드합니다.
if [ -f ~/.env ]; then
  source ~/.env
  echo "[INFO] ~/.env 파일을 성공적으로 불러왔습니다."
else
  echo "[ERROR] ~/.env 파일을 찾을 수 없습니다. 토큰 설정 확인이 필요합니다."
  exit 1
fi

# 1. 빌드 찌꺼기 정리
# 빌드 캐시와 이름 없는 이미지들을 정리하여 디스크 용량을 확보합니다.
docker image prune --filter "label=maintainer=jameskimh" -f

# 2. 도커 이미지 빌드
# --build-arg를 통해 GITHUB_TOKEN 등을 Dockerfile 내부로 전달합니다.
echo "[BUILD] 이미지를 빌드하는 중입니다: $(whoami)/doc_ku"
docker build --no-cache . -t $(whoami)/doc_ku \
  --build-arg UID=$(id -u) \
  --build-arg GID=$(id -g) \
  --build-arg MYNAME=$(whoami) \
  --build-arg GITHUB_TOKEN="${GITHUB_TOKEN}"

# 3. VS Code 서버 영구 저장을 위한 호스트 폴더 생성
# 이 폴더 덕분에 컨테이너를 껐다 켜도 VS Code 무한 로딩이 생기지 않습니다.
mkdir -p ~/.vscode-server-container

# 4. 컨테이너 실행
echo "[RUN] 컨테이너를 실행합니다. 작업 디렉토리: /app"
docker run --rm -it \
  --gpus "device=1" \
  --shm-size=16g \
  --env-file ~/.env \
  --mount type=bind,source=/dataset,target=/dataset \
  -v /data:/data \
  -v "$(pwd)":/app \
  -v ~/.vscode-server-container:/home/$(whoami)/.vscode-server \
  --workdir /app \
  $(whoami)/doc_ku:latest bash