# Video Indexing at Speed

Results cover 30 of 50 files (train-00000 to 00024 and 00040 to 00044).
The other 20 files were not run because I ran out of free GPU quota.

## Install (Ubuntu 22.04, Python 3.11)
sudo apt-get install -y libgl1 libglib2.0-0
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

## Preparation command (needs internet, run once from the repo folder)
python -c "from ultralytics import YOLO; YOLO('yolo11l.pt')"

## Run
python run.py --input data --output results.jsonl
Add --resume to continue an interrupted run.

Note: requirements.txt pins torch 2.7.1 for CUDA 12 drivers. My results
were produced with torch 2.11.0 (CUDA 13 build) on Kaggle.