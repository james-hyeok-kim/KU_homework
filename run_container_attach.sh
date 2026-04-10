#!/bin/bash
COMMIT_ID="e7fb5e96c0730b9deb70b33781f98e2f35975036"
mkdir -p ~/.vscode-server/bin/$COMMIT_ID
echo "다운로드 중..."
curl -sSL "https://update.code.visualstudio.com/commit:${COMMIT_ID}/server-linux-x64/stable" -o vscode-server.tar.gz
echo "압축 푸는 중..."
tar -zxf vscode-server.tar.gz -C ~/.vscode-server/bin/$COMMIT_ID --strip-components 1
rm vscode-server.tar.gz
echo "수동 설치 완료! 이제 VS Code에서 다시 Attach를 시도해 보세요."