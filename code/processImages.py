import os
import Image
import numpy
#import csv

samplesize = 100000
inputdirectory = "D:\\sigal\\doctorat\\kaggle\\train\\extracted\\train\\"
outputpath = "D:\\sigal\\doctorat\\kaggle\\train_images_128_128_all.txt"
def convert_image(path):
    im = Image.open(path)
#    s = im.resize((330, 220),Image.ANTIALIAS)
#    s = im.resize((180, 120),Image.ANTIALIAS)
#   s = im.resize((120, 90),Image.ANTIALIAS)
    s = im.resize((128, 128),Image.ANTIALIAS)

    s1= s.convert('L') # convert image to monochrome
    data = numpy.asarray(s1)
    return data.flatten()

i = -18
# Creates a list containing 5 lists initialized to 0
#Matrix = [[0 for x in range(166501)] for x in range(samplesize)] 
images = [[]]
labels = numpy.recfromcsv('D:\\sigal\\doctorat\\kaggle\\trainLabels.csv', delimiter=',')
#with file(outputpath, 'w') as outfile:    
with open(outputpath, "w") as outfile:
    for filename in os.listdir(inputdirectory):
        i+=1
        if i >= samplesize: break
        else:
            label = labels[numpy.where(labels["image"]==filename[0:filename.find('.')])[0][0]][1]
            if "right" in filename: left = 0
            else: left=1
            fullname= inputdirectory +filename
            #images.append(numpy.append(filename,convert_image(fullname)))
            #numpy.savetxt(outfile, numpy.append(filename,convert_image(fullname)))
            ##outfile.write(",".join(numpy.append(label,numpy.append(filename[0:filename.find('.')],numpy.append(left,convert_image(fullname)))).tolist()))
            outfile.write( str(numpy.append(label,numpy.append(left,convert_image(fullname))).tolist()).strip('[]'))
            outfile.write("\n")
outfile.close
#with open("D:\\sigal\\doctorat\\kaggle\\train_images.csv", "wb") as f:
 #   writer = csv.writer(f)
  #  writer.writerows(images)
#numpy.savetxt('D:\\sigal\\doctorat\\kaggle\\train_images.txt', images, delimiter=',')
#numpy.savetxt('D:\\sigal\\doctorat\\kaggle\\train_images.txt',(images[0],images[1][1:]),delimiter='\t')


