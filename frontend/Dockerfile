# syntax=docker/dockerfile:1

# --- build stage -------------------------------------------------------------
FROM node:20-slim AS build
WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci

COPY . .
RUN npm run build

# --- serve stage -----------------------------------------------------------
FROM nginx:1.27-alpine AS serve

# SPA + /api·/files 프록시 설정
COPY nginx.conf /etc/nginx/conf.d/default.conf

# 정적 빌드 산출물
COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
