# LEO
This is the official code of paper "Learning Only What Matters Most: Efficient Cross-Modal Attention Training". The complete version of our paper is presented in:
```bash
LEO_complete.pdf
```


## Setup
install requirement:
```bash
pip install -r requirements.txt
```

## Run an example
To run an example:
```bash
python TimeMMD/run_longExp.py 
```

We use the TimeMMD archive, refers to https://github.com/AdityaLab/Time-MMD
One dataset Health is provided for running an example.

Put data in dir:
```bash
TimeMMD/dataset
```

## For other models
If you would like to use LEO to train your own models, use codes in following files to replace the vanilla attention codes:
```bash
TimeMMD/aurora/sparse_attention.py
TimeMMD/aurora/OurTransformer.py
TimeMMD/aurora/CrossAttnBudget.py
TimeMMD/aurora/QueryBudgetController.py
```






