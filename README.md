# Chinese News Summarization
This repository contains the implementation of Homework 2 for the Applied Deep Learning course in the CSIE department.
# Environment
Install the packages for training and testing:
```
pip install -r config/requirements.txt
```
Install the packages for evaluation:
```
git clone https://github.com/deankuo/ADL24-HW2.git
cd ADL24-HW2
pip install -e tw_rouge
```
# Training
```
bash ./run_train.sh
```
# Testing
```
bash ./run_test.sh <data file> <output file>
```
# Acknowledgement
Hugging Face repo:<br>
[https://github.com/huggingface/transformers](https://github.com/huggingface/transformers)<br>
ADL TA Team:<br>
[https://github.com/deankuo/ADL24-HW2](https://github.com/deankuo/ADL24-HW2)
