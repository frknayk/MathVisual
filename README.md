# PhotoMath
Perform math operations from image

# Installation
## 1. Create the conda environment
- ```conda create -n photomath python=3.7```
- ```conda activate photomath```

## 2. Install The Tesseract Project

### First you should install binary:

```sudo apt-get update```

```sudo apt-get install libleptonica-dev ```

```sudo apt-get install tesseract-ocr tesseract-ocr-dev```

```sudo apt-get install libtesseract-dev```

### Then you should install python package using pip:

```pip install tesseract```

```pip install tesseract-ocr```


## 3. Install the project
- ```python setup.py install --user```