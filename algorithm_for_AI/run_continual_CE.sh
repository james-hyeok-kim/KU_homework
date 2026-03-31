#!/bin/bash

# 1. 디렉토리 생성 (없을 경우)
mkdir -p logs
mkdir -p results

# 2. 경로 설정 (괄호 대신 중괄호 ${} 사용)
BASE_DIR=$(pwd)
# 로그를 '폴더'가 아닌 '파일'로 지정 (날짜 포함 권장)
LOG_FILE="${BASE_DIR}/logs/experiment.log"

echo "------------------------------------------------"
echo "Starting Continual Concept Erasure Experiment..."
echo "Date: $(date)"
echo "Log file: $LOG_FILE"
echo "Results will be saved in ./results/"
echo "------------------------------------------------"

# 3. 파이썬 스크립트 실행
# 2>&1 : 에러 메시지까지 모두 로그에 포함
# tee -a "$LOG_FILE" : 화면 출력과 동시에 파일에 기록
python continual_CE.py 2>&1 | tee "$LOG_FILE"

echo "------------------------------------------------"
echo "Experiment Finished at $(date)"
echo "Log file saved to: $LOG_FILE"
echo "------------------------------------------------"