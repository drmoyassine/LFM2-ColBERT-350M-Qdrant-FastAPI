#!/bin/sh

# Substitute environment variables in nginx config template
envsubst '${PRIMARY_DOMAIN}' < /etc/nginx/conf.d/nginx.conf.template > /etc/nginx/conf.d/default.conf

# Run Nginx
exec nginx -g 'daemon off;'