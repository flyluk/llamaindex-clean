#!/bin/bash
sudo cp docling.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable docling
sudo systemctl start docling
echo "Docling service installed with GPU support (CUDA_VISIBLE_DEVICES=0)"
sudo systemctl status docling