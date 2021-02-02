# trt_issue
# how to run test?
python issue.py

then you will see results as below, as you can see, tensorrt model with extra tensor operations make the whole inference time increase

```100%|█████████████████████████████████████████████████████████████████████████████████████████████| 10/10 [00:00<00:00, 238.90it/s]
with no gpu 2 cpu operation test result: >>>>>>>>>>>>>>>>> 
average time of raw model is 0.003212761878967285
average time of trt model is 0.0009625434875488281
acceleration of trt model is 3.3377836124046367
100%|█████████████████████████████████████████████████████████████████████████████████████████████| 10/10 [00:00<00:00, 106.91it/s]
with gpu 2 cpu operation test result: >>>>>>>>>>>>>>>>> 
average time of raw model is 0.0022385120391845703
average time of trt model is 0.007107257843017578
acceleration of trt model is 0.31496142234149616```
