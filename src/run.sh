#!/bin/sh
virtualenv --system-site-packages -q -p /usr/bin/python3.4 .env
source .env/bin/activate
pip3 install --quiet -r src/requirements.txt
.env/bin/python3 src/main.py $@
