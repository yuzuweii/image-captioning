# image-captioning

## 50.038 CDS project final code

1003683 Wang Wei
1003689 Pan Feng
1004447 Wei Wen Chin

# Setup training environment: (all in AWS environment)

## 1. Required libraries

Python 3.6 above
Pytorch 1.0.1.post2
Nltk 3.6.5
Cython version 0.29.24

## 2. Clone COCO API

$ git clone https://github.com/pdollar/coco.git
$ cd coco/PythonAPI/
$ make
$ python setup.py build
$ python setup.py install

Download image and annotations from https://cocodataset.org/#download
Save to ./data2/train2014 and ./data2/val2014
Annotations save to ./data2/annotations/captions_train2014.json

## 3. Build vocabulary

$ python3 build_vocab.py
$ python3 resize.py
$ python3 resize.py --image_dir ‘./data2/val2014’ --output_dir ‘./data2/val_resized2014’

## 4. Training

$ python3 train.py --num_epochs 3

## 5. Validation

$ python3 validate.py

## 6. Prediction

You can download a single image and name is as 'test-image.jpg'.
Place the iamge inside LSTM folder and run following command.

$ python3 sample.py

The prediction will be printed on the screen.
