# Hardware

Results cover 30 of the 50 files: train-00000 to train-00024 (Kaggle) and
train-00040 to train-00044 (Colab). The other 20 files were not run
because I ran out of free GPU quota.

## Kaggle (files 0 to 24)
- GPU: 2x NVIDIA Tesla T4, 15 GB each
- CPU: Intel Xeon @ 2.00GHz, 4 cores
- RAM: 31 GB
- Disk: SSD (lsblk shows non-rotational disks)
- OS: Ubuntu 24.04.4 LTS, Python 3.13
- Run time: about 9 hours 57 minutes (35,795 seconds) for 25 files,
  run as a Kaggle background job (Save and Run All)
- The script used both GPUs, one worker per GPU

## Colab (files 40 to 44)
- GPU: 1x NVIDIA T4 (free Colab)


## Notes
- I ran on Python 3.13 and a CUDA 13 build of PyTorch (2.11.0).
- Speed on one T4 was about 100 frames per second with FP16 and batch size 16.