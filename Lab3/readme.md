

# Introduction

## Goal

Using algorithms and compute methods to identify clothing items.

## Data

The data was sourced from the Zalando Research fashion MNIST GitHub repository. No additional data was gathered for the model and testing that was done.

Depending on the model used, certain methods were used to transform and clean the data. The most heavily used is the SciKit learn standard scaler function. The X data is reduced to a decimal by dividing it by 255, this makes the data more regular and easier to manipulate as a float32 or float64

## Test Methodology

To find the model parameter combination that provided the highest level of accuracy several tests were run. Each iteration started with a fresh data set, changed one parameter from the last run, and captured several metrics into a Pandas data frame. The output of these tests is captured in the Tests\_All.xlsx file included in the report.

# System/Test Setup

The primary system the tests were conducted on is a home workstation with the below hardware and software running. GPU processing was utilized as much as possible to reduce run time.

## Hardware

- OS Name: Microsoft Windows 10 Pro
- OS Version: 10.0.17134 17134
- CPU: Intel(R) Core (TM) i7-4790K CPU @ 4.00GHz
- RAM: 16 GB
- Graphics Card: NVIDIA GeForce GTX 980
- Python version: 3.6.6
- Anaconda version: 1.8
- CUDA Toolkit version: 9.2

## Python Packages

A full list of packages installed is included in the installed\_packages.txt file. The models tested make heavy use of Scikit Learn version 0.19.1 and Keras version 2.2.2.

# Keras using Theano Backend

Keras was used as a neural network model that allows the utilization of GPU for processing and stacking of various layers to form a model. Keras can be used with multiple different GPU backends, in this series of tests the Theano backend was used.

The X data was transformed into a three-dimensional array while the Y data was transformed into categorical data using Keras&#39; built in np\_utils to\_categorical. Since the Y data is transformed it was necessary to flatten it back to a normal integer when comparing predicted values to the actual test values. This was done with the Numpy argmax fuction.

![Total Accuracy Across Multiple Parameters](https://github.com/mencarellic/IST718/blob/master/Lab3/images/Keras_all_runs.png)

While attempting to find the most effective model parameters 25 iterations of each configuration was used. After identifying the most effective, this epoch value was expanded to 50. The best configuration was found to be using the categorical cross-entropy loss function with the RMSProp optimizer. With 25 iterations, the best accuracy attained was 86.8% in 3.7 minutes. When expanding that to 50 iterations the accuracy increased to 87.83% and runtime increased to 7.2 minutes.

![Accuracy per Iteration, Best Parameter Settings](https://github.com/mencarellic/IST718/blob/master/Lab3/images/Keras_best_run.png)

# K Nearest Neighbor

Using the K Neighbor Classifier function in the SciKit Learn package, a max accuracy of 85.42% was attained. The KNN accuracy was grouped tightly regardless of parameters specified for the model. The total range of the accuracy across all k values was 1.29%. The KNN classifier function was the longest running model attempted. This is probably because GPU processing could not be used to speed up the model, there is a parameter to use parallel processing with the CPU that improved run time by approximately 15 to 20%. On average each run of the model took approximately 22 minutes.

Each test varied the k value, algorithm, or weight function used. The best results were achieved using K = 7, the distance weight method, and with no preference being given to algorithm. The final choice was to use the KD tree algorithm since the compute time was significantly less than calculating with the ball tree algorithm.

The data was left in a two-dimensional Numpy array since the function does not work when inputting three-dimension arrays.

![Comparison of K values, Weight Functions, and Algorithms](https://github.com/mencarellic/IST718/blob/master/Lab3/images/KNN_parameter_comparison.png)

# Logistic Regression

The logistic regression model varied only the solver method from run to run. This appears to only affect the efficiency of the model and not the overall accuracy since all four solvers tested resulted in the same accuracy. The LBFGS solver ran the quickest – 26 seconds. The accuracy was 76.29%. Given the low number of configurable parameters and low initial accuracy continued investigation into this model was not done.

# RandomForest

A RandomForest model was created varying the number of trees from 1 to 100 while testing. Predictably, accuracy increased as the number of trees increased, though there were minimal gains beyond 25 trees. Compute time for each forest was only a few seconds so for the final version of the model, 500 trees were used. With 500 trees, an accuracy of 87.73% was reached in 2.1 minutes. The SciKit Learn implementation of RandomForest has an easy method to allow for parallel processing using the n\_jobs parameter. This was utilized during processing to speed compute times up.

Like the KNN and Logistic Regression models, no data preprocessing was conducted beyond the SciKit Learn preprocessing and the reduction of the X values to floats.

![Random Forest Accuracy per N Trees](https://github.com/mencarellic/IST718/blob/master/Lab3/images/RF_all_acc.png)

# Conclusions and Future Analysis

| Model | Accuracy | Run Time | Comments |
| --- | --- | --- | --- |
| Keras (Theano Backend) | 87.83% | 7.2 Minutes | Used GPU processing. |
| KNN | 85.42% | 21.7 Minutes | Used multi-threaded CPU processing. |
| Logistic Regression | 76.29% | 0.5 Minutes |   |
| Random Forest | 87.73% | 2.1 Minutes | Used multi-threaded CPU processing. |

Both the Keras and RandomForest models performed extraordinarily well and offer the best solution depending on the workstation hardware available. There are potential improvements to be done with the KNN classifier such as possible using the OpenCV3 package to utilize the GPU for processing. The only model that took advantage of a three-dimension array was Keras all the remaining models were limited to two-dimension arrays which means finding additional methods for models to utilize that third dimension could improve accuracy.

There are countless different potential layer combinations with the Keras model that were not explored due to time constraints. Continued tuning and experimentation could probably improve the accuracy of the model. Having a GPU for processing is necessary for this model, running it on CPU only meant a completion time of over four hours for one iteration. To run the final test configuration with 50 iterations, it&#39;d take weeks to complete. Even using the IBM cloud platform one iteration was estimated to take nearly an hour and a half with the free offering.

![Random Sample of 15 Articles, Predicted Values, and Actual Values from the RandomForest Model](https://github.com/mencarellic/IST718/blob/master/Lab3/images/RF_check.png)

The configuration of the environment for GPU processing can be difficult and time consuming as well if the user is not familiar with the process. Use of Anaconda makes it easier, but the installation of the CUDA toolkit and DLib were troublesome.

RandomForest is an excellent alternative if Keras and GPU processing is not a possibility. With little configuration and tuning, the RandomForest was reaching high levels of accuracy. Additional exploration of the configurable parameters could improve accuracy.

# References and Sources

[https://keras.io](https://keras.io)

[http://scikit-learn.org/stable/index.html](http://scikit-learn.org/stable/index.html)

[https://github.com/zalandoresearch/fashion-mnist](https://github.com/zalandoresearch/fashion-mnist)

