#!/bin/bash

mkdir -p data
echo "Downloading AIFB..."
curl -L "https://www.dropbox.com/sh/ldjd70yvnu9akxi/AAAam7SBr5KXLfjk-NVGQNWRa?dl=1" -O "./data/aifb.zip"
unzip "data/aifb.zip" -d "data/aifb"
echo -e "Done. \n\n"


echo "Downloading MUTAG..."
curl -L "https://www.dropbox.com/sh/tburaaxij0a1vmy/AAAlD5ORzcMbF3YpoynOLGqwa?dl=1" -O "data/mutag.zip"
unzip "data/mutag.zip" -d "data/mutag"
echo -e "Done. \n\n"
