# Image Aesthetic Quality Assesment (IAQA) Research Repository

## contains:
>> training files for AVA benchmarking datase as binary classificaiton
training is from directories in the following format:

```bash
   cwd\
       ⌊_train\
       |       ⌊_low\
       |       |     ⌊_id.jpg,...
       |       ⌊_high\
       |              ⌊_id.jpg,... 
       ⌊_test\...
       ⌊_val\...
```

>> Data dictionary of IAQA datasets with a file structure correspoinding to classes of each IAQA dataset

>> A terminal runnable version --including args --parse is to be completed (almost there)


**Training Notebook:**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1cwhu6qsGy0Pc3tnXWo82hq3uhV5-fTu-?usp=sharing)

**Evaluation Notebook:**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1f4N2tefnfAWme2ro2O8LU_FwuOfDo_fC?usp=sharing)

notebooks currently have all code if first cell

## DATA:

|Dataset|n Images|Size (compressed)|
|-------|--------|-----------------|
|[AVA](https://drive.google.com/drive/folders/1uc-jyzGNndFvhdiHaxAvEkj-jP5e6F1f?usp=sharing)| 255508|  32 GB batched into 44 .zip| 
|[IAD](https://drive.google.com/open?id=1Viswtzb77vqqaaICAQz9iuZ8OEYCu6-_)|10k |2GB| 



Downloads are handled by code however you can obtain files in terminal using gdwon (best for large files):

to get ava:

```bash
gdown https://drive.google.com/drive/folders/1uc-jyzGNndFvhdiHaxAvEkj-jP5e6F1f?usp=sharing
```
to get IAD:

```bash
gdown https://drive.google.com/open?id=1Viswtzb77vqqaaICAQz9iuZ8OEYCu6-_
```

Frida

**Training:**

- [x] Select alternative IAQA dataset 28/01
- [ ] train all models on second database: 
>> - [ ] Resnets {18,50,152}  29/01
>> - [ ] CvT 29/01
>> - [ ] ConViT (T,S,B)
>> - [ ] CaiT
>> - [ ] BeiT
- [ ] Evaluate all models

**code:**
- [ ] update notbooks with git clone (this repository and remove code heavy cells)
- [ ] write guidance cells for colab nb. 27/01
- [ ] terminal runnable code with args parsers feb
- [ ] separate python files (resolving import of data augmentation class) feb
- [ ] create notebook with public trained models feb

**EVAL**

- [ ] compile results 

[Mawady](https://github.com/mawad)y
**Training:**
- [ ] Train on AVA CaiT locally 28/01-30/01
- [ ] Train on AVA BeiT locally 28/01-30/01
