#!/bin/sh
python3 -m venv .env
source .env/bin/activate
pip3 install -r requirements.txt && pip3 install --upgrade pip &&
.env/bin/python3 -m spacy download en &&
.env/bin/python3 src/main.py $@
