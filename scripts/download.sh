#!/usr/bin/bash
echo "Download start Evaluation Dataset"
mkdir -p /workspace/data/eval
wget ftp://mldisk.sogang.ac.kr/ganpan_matching/eval.tar.gz -O /workspace/data/eval.tar.gz
tar -zxvf /workspace/data/eval.tar.gz -C /workspace/data/