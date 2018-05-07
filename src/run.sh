#!/bin/sh
python3 -m venv .env
source .env/bin/activate
pip3 install --quiet -r requirements.txt
.env/bin/python3 -m spacy download en
.env/bin/python3 src/main.py $@
