# DiabeticRetinopathyDetection-with-Deep-Learning
Detecting the presence of diabetic retinopathy in image scans using Python's Lasagne package

## Background
As a Phd student, intereseted in both enriching my toolbox (especially getting familiar with Python) 
and getting to know deep learning techniques, I looked for a fast and interesting occasion to start from.

I found kaggle's Diabetic Retinopathy Detection competition (https://www.kaggle.com/c/diabetic-retinopathy-detection)
as a motivating event that will push me towards getting to know new areas and tools.

Though I did not manage to submit the results on time, this experience was indeed everything I looked for and more. 
It was actually my first experience in Python, my first time using AWS (Amazone Web Services) and working with ubuntu servers, 
my first use of Github and my first time trying deep learning techniques.

It takes time to find the right guides, here are the excellent links that led me step by step through this exercise:
- Using convolutional neural nets to detect facial keypoints tutorial (http://danielnouri.org/notes/2014/12/17/using-convolutional-neural-nets-to-detect-facial-keypoints-tutorial/)
- Compatible AWS instructions (https://www.kaggle.com/c/facial-keypoints-detection/details/deep-learning-tutorial)

## Getting Started

1. Download files from https://www.kaggle.com/c/diabetic-retinopathy-detection/data, then unzip using 7zip 
(note: it is only possible to unzip after downloading all the partitioned files).
2. Use the /code/PlayingWithImage.ipynb if you want to play a bit with an image or to and watch the steps of its processing,
3. Execute /code/processImages.py (after setting train and test paths) in order to go over the images, process them and save the processed data in a txt file. (two files actually: for train and test)
4. Execute /code/classifyData.ipynb step by step in order to try two optional NeuralNets to solve this task. 

**Bad news:** The results only put me on the 270 place in the competition, but this is only a starting point, 
changing the parameters might lead to improvements.

**Good news:** No need to be upset that I ended one week late..

