FROM nginx:1.27-alpine

# wget for HEALTHCHECK (busybox wget shipped with alpine)
RUN apk add --no-cache wget

COPY . /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD wget -qO- http://localhost/ || exit 1
