## Q1: timings

my first version was only tested on 4 videos, not the full 50 files. it took 603.5 seconds to process 21,619 frames. my final version took 205.1 seconds on the same videos. so it was almost 2.9 times faster,, the final version also processed 25 parquet files on 2 T4 GPUs in 35,795 seconds (about 9 hours and 57 minutes).

the biggest improvement came from using FP16 instead of FP32. batching also helped, but FP16 made the biggest difference

## Q2: pipeline

first, my script opens a parquet file and reads the video data from it. then it sends the videos to GPU workers for processing. Each video is opened and decoded into frames, i also handle timestamps and frame rotation when needed. the frames are collected into batches of 16.then the model detects objects in each batch using confidence 0.4, image size 640 and IOU 0.7. i use FP16 for faster inference,, and after detection, the script counts the detected objects for each second of the video, saves the results in a temporary file. Once processing is complete, the temporary file is renamed to the final output file.

## Q3: where the time goes

the model takes most of the time. i tested this on row 0 using a T4 GPU. In the baseline version, decoding took 1.8 seconds while the model took 135.5 seconds. in my final version, decoding took 3 seconds and the model took 45.5 seconds.

i measured both parts using `time.perf_counter()`. i also used `torch.cuda.synchronize()` before and after model inference so the GPU timing was accurate. This showed that decoding was not the main problem. almost all the processing time was spent on model inference

## Q4: checking correctness

i used my original FP32 version as the reference and compared every faster version against it. i made a comparison script to check object F1 score, exact people count match and people count within 1%. I tested batching and FP16 with different batch sizes. on row 0, all versions had an F1 score of 1.0 and 100% exact people count. i also tested the final FP16 batch 16 version on 4 videos. it achieved 1.0 F1, 99.89% exact people count and 100% within 1%.

Tho, after changing OpenCV decoding to PyAV, i repeated the same test. the results stayed the same and there were no error lines.

## Q5: working with AI

One suggestion was to use a separate thread for video decoding to make processing faster. i tested this instead of just assuming it would help. the baseline gave around 39.4 FPS with batching, while adding a separate decode thread only increased it to 39.6 FPS. the improvement was very small. decoding was also taking only around 1 to 2% of the total time. so i decided not to keep this change. it added more complexity without giving any real speed improvement.

## Q6: next steps

if i had one more week, i would test the remaining parquet files and check how the script performs on different types of videos.i would especially test variable frame rate videos, rotated videos and videos with missing or incorrect timestamps,,, i would also test the pipeline on a single GPU to understand how much time it needs without using two GPUs.

another thing i would check is compatibility with the required Python and PyTorch versions. my current tests were done using Python 3.13 and Torch 2.11, while my requirements file specifies Torch 2.7.1.

I think this would help me make sure the pipeline works reliably in different environments.