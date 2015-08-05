#Deep Learning using Python:
I found some great starting points that I've decided to follow:


Using convolutional neural nets to detect facial keypoints tutorial: http://danielnouri.org/notes/2014/12/17/using-convolutional-neural-nets-to-detect-facial-keypoints-tutorial/


Since a GPU is required for using these technique, the following link guides how to use an AWS (amazon web service) and set the image for using the running packages.
https://www.kaggle.com/c/facial-keypoints-detection/details/deep-learning-tutorial


#Working with AWS:
This was my first experience with using amazon's web services, so I will mention the main obstacles on the way:

1. The image in the link is now called ami-b141a2f5.
2. In order to find it make sure to set the region in the above ribbon to N. California.
3. Following the instructions from the link above, I still had to install/update some packages myself, so here is a what I eventually ran in order to set the environment:

- git clone https://github.com/wendykan/AWSGPU_DeepLearning.git 
- chmod 777 -R AWSGPU_DeepLearning/
- sudo apt-get install libfreetype6-dev
- sudo apt-get install libpng12-dev
- wget https://bootstrap.pypa.io/ez_setup.py -O - | sudo python
- ./AWSGPU_DeepLearning/setup.sh
- pip install -r https://raw.githubusercontent.com/dnouri/kfkd-tutorial/master/requirements.txt

4. From the home directory, add the following ~/.theanorc file that configures Theano to use the machine's GPU"

- vi .theanorc
- Type I for insert, then paste the following text:

[global]

floatX = float32

device = gpu0

[nvcc]

fastmath = True

[mode]

optimizer_excluding=conv_gemm

c. Type esc+: wq to exit and save vi
d. sudo ldconfig /usr/local/cuda/lib64

#Working with Unix:
1. Most installations only succeeded from root directory using sudo prefix
2. To search for packages:

apt-cache search <string>

e.g:

apt-cache search png | grep dev

libpng12-dev 

libpng3

3. Running python (from root directory):
a. Run a command: sudo python -c "X, y = load()"
b. Run a file: sudo python /home/ubuntu/AWSGPU_DeepLearning/ex_load.py
c. Run using Ipython: Ipython

then type %autoindent to set off indentation and be able to paste blocks of code

4.	Split large files (to 10000 lines per file, for example): 

split -l/N 10000 orig_file.txt new
