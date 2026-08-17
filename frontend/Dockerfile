# frontend/Dockerfile
# Multi-stage build: compile the Vite app, then serve the static output with nginx
# (which also reverse-proxies /api/* to the backend container - see nginx.conf).
FROM node:22-alpine AS build
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build

FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
