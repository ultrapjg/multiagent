# 실행 방법

## 로컬에서 직접 실행
1. backend 구동
   ```bash
   cd backend
   python main.py
   ```
2. admin-frontend 구동
   ```bash
   cd admin-frontend
   streamlit run main.py --server.port=8501
   ```
3. user-frontend 구동
   ```bash
   cd user-frontend
   streamlit run main.py --server.port=8502
   ```

## Docker 이미지 빌드 및 실행
Docker를 사용해 각 서비스를 컨테이너로 실행할 수 있습니다.

1. 이미지 빌드
   ```bash
   docker compose -f docker/docker-compose.yml build
   ```
2. 컨테이너 실행
   ```bash
   docker compose -f docker/docker-compose.yml up
   ```

각 서비스는 다음 포트로 노출됩니다.
- backend: `8000`
- admin-frontend: `8501`
- user-frontend: `8502`

## Paketo Buildpack으로 이미지 빌드
`pack` CLI가 설치되어 있다면 Paketo Python Buildpack을 사용해 이미지를 생성할 수 있습니다.

```bash
pack build backend --path backend --builder paketobuildpacks/builder-jammy-base
pack build admin-frontend --path admin-frontend --builder paketobuildpacks/builder-jammy-base
pack build user-frontend --path user-frontend --builder paketobuildpacks/builder-jammy-base
```

## Skaffold를 이용한 자동 재배포
[Skaffold](https://skaffold.dev/)를 사용하면 소스 코드 변경 시 자동으로 빌드와 배포가 이루어집니다.

`skaffold.yaml` 파일에 정의된 대로 Paketo Buildpack으로 이미지를 빌드하고,
`k8s` 폴더의 매니페스트를 이용해 Kubernetes 클러스터에 배포합니다. 로컬 클러스
터를 사용한다면 다음과 같이 실행합니다.

```bash
skaffold dev --port-forward
```
