## Installation
```shell
$ cd ${WORKSPACE}
$ git clone -b superglue --single-branch https://${PERSONAL_ACCESS_TOKEN}@github.com/gyusik19/SCA-SignMatching
$ cd SCA-SignMatching
$ vim docker-compose.yml
  # volume 경로, ports 수정
$ docker-compose up -d --build
$ docker attach ${CONTAINER NAME}
$ pip install -r requirements.txt
```