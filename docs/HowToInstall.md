## Installation
### 1. Docker compose

```shell
cd ${WORKSPACE}  # git clone할 경로
git clone https://${GITHUB_PERSONAL_TOKEN}@github.com/jaechanjo/SCA-SignMatching.git
cd SCA-SignMatching
docker-compose up -d  # build docker container
```

### 2. Packages

```shell
cat requirements.txt | while read PACKAGE; do pip install "$PACKAGE"; done  # 설치 시 버전 오류를 무시합니다.
```
