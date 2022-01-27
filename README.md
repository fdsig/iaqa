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
>> Colab notebook which evaluates Defocus Deblurring: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1cwhu6qsGy0Pc3tnXWo82hq3uhV5-fTu-?usp=sharing)
 


Frida
**Training:**
- [ ] Select alternative IAQA dataset 28/01
- [ ] train all models on second database: 
>> - [ ] Resnets {18,50,152}  29/01
>> - [ ] CvT 29/01
>> - [ ] ConViT (T,S,B)
>> - [ ] CaiT
>> - [ ] BeiT
- [ ] Evaluate all models
**code:**
- [ ] write guidance cells for colab nb. 27/01
- [ ] terminal runnable code with args parsers feb
- [ ] separate python files (resolving import of data augmentation class) feb
- [ ] create notebook with public trained models feb

**EVAL**
- [ ]compile results 

[Mawady](https://github.com/mawady)
**Training:**
- [ ] Train on AVA CaiT locally 28/01-30/01
- [ ] Train on AVA BeiT locally 28/01-30/01
