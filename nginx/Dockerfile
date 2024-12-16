FROM nginx:latest

# Copy the custom Nginx configuration
COPY nginx.conf /etc/nginx/nginx.conf

# Copy the certificate files into the container (after generating them via Certbot)
COPY ./nginx-certs/ /etc/letsencrypt/

# Expose ports
EXPOSE 80 443