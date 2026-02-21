FROM nginx

COPY nginx.conf /etc/nginx/nginx.conf
COPY fullchain.pem /etc/letsencrypt/live/api.delineate-data.com/fullchain.pem
COPY privkey.pem /etc/letsencrypt/live/api.delineate-data.com/privkey.pem
COPY options-ssl-nginx.conf /etc/letsencrypt/options-ssl-nginx.conf
COPY ssl-dhparams.pem /etc/letsencrypt/ssl-dhparams.pem
COPY chain.pem /etc/letsencrypt/live/api.delineate-data.com/chain.pem
