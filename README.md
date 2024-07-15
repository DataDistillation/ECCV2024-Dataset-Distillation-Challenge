# ECCV2024-Dataset-Distillation-Challenge

This repository contains the evaluation code for the Data Compression Challenge. The code evaluates the performance of submitted models on CIFAR-100 and Tiny ImageNet datasets. The evaluation is performed on an NVidia 4090

The sample submission can be downloaded [sample_submission.zip](https://drive.google.com/file/d/12Ntz6LclB7N3oCXRzGiN7eawlqMochhA/view?usp=drive_link).

The Test Set of CIFAR-100 and Tiny-ImageNet can be downloaded [reference_data.zip](https://drive.google.com/file/d/1MZMsEbBHe3gYrq4y4Na3Ogh9sIKecng-/view?usp=drive_link).

## Important Design Notes

Please keep in mind, that the test data is normalized following the standard normalization technqiues for CIFAR100 and TinyImagenet. In particular we assume your distilled data has been learned from a normalized training dataset using:

```python
#* CIFAR100
# mean = [0.5071, 0.4866, 0.4409]
# std = [0.2673, 0.2564, 0.2762]

#* TinyImagenet
# mean = [0.485, 0.456, 0.406]
# std = [0.229, 0.224, 0.225]

```

We do not perform normalization in this evaluation script -- your data must be pre-normalized (i.e normalized before the distillation following other commone distillation works). The sample submission data represents a random selection at IPC10 for both CIFAR100 and TinyImagenet.


## Usage

1. Please follow the same heiarchial structure as our sample_submission.

2. Please unzip the reference testing data "reference_data.zip" and create the folder structure "./reference_data/{cifar100|tinyimagenet}_test.pt"

3. Please unzip the sample submission data "sample_submission.zip" and create the folder structure "./sample_submission/{cifar100|tinyimagenet}.pt". Note: "sample_submission" contains 2 files: "cifar100.pt" and "tinyimagenet.pt". Both files contain randomly samples images from the respective datasets at IPC 10. Please follow the same structure when creating and saving your distilled data. 

3. To evaluate your data, please set the "--submit_dir {your_path}"
```bash
python evaluate.py --submit_dir {path-to-your-data}
```

Alternatively, to evaluate the sample use:

```bash
python evaluate.py --submit_dir ./sample_submission/
```


## Script Explanation
- `evaluate.py`:
    - Loads the distilled train data from the submission file.
    - Loads the test data and labels from the reference files.
    - Defines a simple Convolutional Neural Network (CNN) for classification.
    - Trains the CNN on the distilled data.
    - Evaluates the trained model on the test data.
    - Computes and outputs the average accuracy over three runs.
 


## Troubleshooting
- Ensure that the input directory structure matches the expected format.
- Verify that the .pt files contain the expected data and are not corrupted.
- Make sure you have a compatible version of PyTorch installed.
- Ensure data normalization
