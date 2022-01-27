# Image Aesthetic Quality Assesment (IAQA) Research Repository

## contains:
>> training files for AVA benchmarking datase as binary classificaiton
training is from directories in the following format:

```
   cwd\
       ⌊_train\
       |       ⌊_low\
       |       |     ⌊_id.jpg,...
       |       ⌊_high\
       |              ⌊_id.jpg,... 
       ⌊_test\...
       ⌊_val\...
```

>> data dictionary of IAQA datasets with a file structure correspoinding to classes of each IAQA dataset

>> A terminal runnable version --including args --parse is to be completed


>> running  colab notebook here:
>> Colab notebook which evaluates Defocus Deblurring: [![Open In Colab](https://colab.research.google.com/drive/1cwhu6qsGy0Pc3tnXWo82hq3uhV5-fTu-?usp=sharing)
 



Me.self
**Training:** 
- [ ] Select alternative iaqa dataset
- [ ] train all models on second database:
>> - [ ] Resnet
>> - [ ] CvT
>> - [ ] ConViT (T,S,B)
>> - [ ] CaiT
>> - [ ] BeiT
- [ ] Evaluate all models
**code:**
- [ ] terminal runnalbe code with args parsere
- [ ] sepparate python files (resolving import of data augmentation class) 
- [ ] create notebook with public trained models and 

compile results:


[Mawady](https://github.com/mawady)

- [ ] Train on AVA CaiT
- [ ] Train on AVA BeiT
