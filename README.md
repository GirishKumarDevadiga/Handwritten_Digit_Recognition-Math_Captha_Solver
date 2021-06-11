** PROBLEM DEFINITION

To implement a classification algorithm to recognize handwritten digits (0‐ 9). This project presents our implementation of the Neural networks to recognize the handwritten numeral digits.
Deployment of the trained model as a web app using flask.

** MNIST AND MATH SYMBOL DATASETS

The MNIST dataset, a subset of a larger set NIST, is a database of 70,000 handwritten digits, divided into 60,000 training examples and 10,000 testing samples. The images in the MNIST dataset are present in form of an array consisting of 28x28 values representing an image along with their labels. Also, we have used dataset from math symbol containing 7000 samples each for ‘-’ and ‘+’, 3251 samples for ‘*’ and 868 samples for ‘/’. Merging of data resulted in 76307 samples for training and 11812 samples for testing dataset.

** LIBRARY USED

Python 3 and above 
TensorFlow library for training and inference of deep neural networks.
Keras library for interface of TensorFlow library
Flask framework to deploy model

** CNN MODEL SUMMARY

![image](https://user-images.githubusercontent.com/73153277/121680664-6ff46880-cad7-11eb-9e82-420e254dd0f9.png)

** RESULTS
** Training  and Validation Accuracy:

![image](https://user-images.githubusercontent.com/73153277/121680795-961a0880-cad7-11eb-9bc4-079b1d049cb4.png)

** Testing Accuracy:

![image](https://user-images.githubusercontent.com/73153277/121680862-af22b980-cad7-11eb-951b-7dcb5b56e147.png)

** Model Deployment using Flask Demo:

![image](https://user-images.githubusercontent.com/73153277/121680954-cfeb0f00-cad7-11eb-8fe9-a6470cb23bb4.png)

** PROBLEMS FACED AND LESSONS LEARNT
* Faced issue of overfitting and shuffled the dataset. 
* Math dataset and MNIST dataset compatibility issue resolved by converting the math dataset samples to binary images.
* Faced many issues related to Trained model deployment using Flask library, fixed the issues by referring multiple online sources and got exposure to flask functionalities.


** REFERENCES
* https://www.mdpi.com/1424-8220/20/12/3344/htm
* https://core.ac.uk/download/pdf/231148505.pdf
* https://www.freecodecamp.org/news/an-intuitive-guide-to-convolutional-neural-networks-260c2de0a050/
* https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53
* https://towardsdatascience.com/a-laymans-guide-to-building-your-first-image-classification-model-in-r-using-keras-b285deac6572
* MNIST Dataset: http://yann.lecun.com/exdb/mnist/
* Math Symbol Dataset: https://www.kaggle.com/c/digit-recognizer/data
* https://www.ijircst.org/DOC/16-handwritten-digit-recognition-using-various-machine-learning-algorithms-and-models.pdf
* https://www.researchgate.net/figure/Block-Diagram-Proposed-Handwritten-Digit-Recognition-System_fig1_339106193.pdf
* https://data-flair.training/blogs/python-deep-learning-project-handwritten-digit-recognition/
* http://www.diva-portal.org/smash/get/diva2:1293077/FULLTEXT02.pdf
* https://machinelearningmastery.com/handwritten-digit-recognition-using-convolutional-neural-networks-python-keras/
* https://rstudio-conf-2020.github.io/dl-keras-tf/04-computer-vision-cnns.html#7
