# SVM_GARCH_MODEL

Before running, use 
```
pip install -r requirements.txt
```
Run
```
python baseline.py
```
It will automatically download the data from web and run 2 baselines and print out the result.
After the data is downloaded, Run
```
python svm_arch.py
```
It will use svm with different kernels to estimate the parameter, print out the Result and draw predited volatility.
