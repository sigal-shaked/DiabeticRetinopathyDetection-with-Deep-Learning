import os
import numpy 
import Image

samplesize = 100000
#inputdirectory = "D:\\sigal\\doctorat\\kaggle\\test\\extracted\\test\\"
#outputDataPath = "D:\\sigal\\doctorat\\kaggle\\test_images_128_128_data.txt"
#outputNamesPath = "D:\\sigal\\doctorat\\kaggle\\test_images_128_128_names.txt"
inputdirectory = "D:\\sigal\\doctorat\\kaggle\\test\\extracted\\toprocess\\"
outputDataPath = "D:\\sigal\\doctorat\\kaggle\\test_images_128_128_data_remain.txt"
outputNamesPath = "D:\\sigal\\doctorat\\kaggle\\test_images_128_128_names_remain1.txt"
def convert_image(path):
    im = Image.open(path)
    s = im.resize((128, 128),Image.ANTIALIAS)
    s1= s.convert('L') # convert image to monochrome
    data = numpy.asarray(s1)
    return data.flatten()

i = 0
outfile = open(outputDataPath, "w")
namesfile = open(outputNamesPath, "w")
for filename in os.listdir(inputdirectory):
    i+=1
    if i >= samplesize: break
    else:
        fullname= inputdirectory +filename
        X = convert_image(fullname)
        if "right" in filename: 
            X = X.reshape(128,128)
            X = numpy.fliplr(X)
            X = X.reshape(128*128)
        outfile.write( str(X.tolist()).strip('[]'))
        outfile.write("\n")
        namesfile.write( filename[0:filename.find('.')])
        namesfile.write("\n")
namesfile.close
outfile.close

#check validity of output file
with open("D:\\sigal\\doctorat\\kaggle\\test_images_128_128_data.txt",'r') as f:
    row_count = sum(1 for row in f)