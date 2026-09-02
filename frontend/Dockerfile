# syntax=docker/dockerfile:1

# VLVerifier 프론트엔드 이미지 — node 로 SPA 정적 빌드 후 nginx 로 서빙
# 개발용 dev 서버는 docker-compose.dev.yml 이 build 스테이지에서 멈춰 재사용

# --- build 스테이지: 의존성 설치 + 정적 자산 빌드 -------------------------------
FROM node:20-slim AS build
WORKDIR /app

# package.json 이 안 바뀌면 npm ci 레이어를 캐시하도록 먼저 복사
COPY package.json package-lock.json ./
RUN npm ci

COPY . .
RUN npm run build

# --- serve 스테이지: 빌드 산출물만 담은 경량 nginx --------------------------------
FROM nginx:1.27-alpine AS serve

# SPA 폴백 + /api·/files 프록시 설정
COPY nginx.conf /etc/nginx/conf.d/default.conf

# build 스테이지의 정적 산출물만 반입 (node_modules 제외)
COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
