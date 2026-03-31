#!/bin/bash

# 디렉토리 생성 (없을 경우)
mkdir -p logs
mkdir -p results

echo "------------------------------------------------"
echo "Starting Continual Concept Erasure Experiment..."
echo "Date: $(date)"
echo "Logs will be saved in ./logs/"
echo "Results will be saved in ./results/"
echo "------------------------------------------------"

# 파이썬 스크립트 실행
python continual_CE.py

echo "------------------------------------------------"
echo "Experiment Finished at $(date)"
echo "------------------------------------------------"