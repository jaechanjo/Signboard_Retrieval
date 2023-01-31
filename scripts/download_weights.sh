#!/usr/bin/bash
echo "Download start superglue model weights"
wget ftp://mldisk.sogang.ac.kr/ganpan_matching/superglue/weights/superglue_outdoor.pth -O /workspace/models/weights/superglue_outdoor.pth
echo "Download start superpoint model weights"
wget ftp://mldisk.sogang.ac.kr/ganpan_matching/superglue/weights/superpoint_v1.pth -O /workspace/models/weights/superpoint_v1.pth
