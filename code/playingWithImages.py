import Image
im = Image.open(r"D:\sigal\doctorat\kaggle\train\extracted\train\10_left.jpeg")
print im.format, im.size, im.mode
#from IPython.display import Image 
pylab.imshow(im)

im.resize((667,1000 ), filter)

import numpy as num
imarray=num.array(im)


s = im.resize((500, 333),Image.ANTIALIAS)
pylab.imshow(s)
import ImageOps

s1= s.convert('L') # convert image to monochrome - this works
pylab.imshow(s1)

s2= s.convert('1') # convert image to black and white
pylab.imshow(s2)

data = numpy.asarray(s1)
data1 = data.flatten()


x_str = s1.tostring('raw', s1.mode)
x = np.fromstring(x_str, dtype=np.uint8)
x.shape = s1.size[1], s1.size[0], 3












imdata = s.tostring("raw", "RGB", 0, -1)
st = s1.tostring()

imsave('result_col.png', image_file)

import glob
for infile in glob.glob("*.jpg"):
file, ext = os.splitext(infile)
im = Image.open(infile)
im.thumbnail((128, 128), Image.ANTIALIAS)
im.resize(size, filter)
im.save(file + ".thumbnail", "JPEG")

#to 3 colors
im.split()

im.tostring() => string
#Returns a string containing pixel data, using the standard "raw"encoder.
im.tostring(encoder, parameters) => string
#Returns a string containing pixel data, using the given data encoding

im.transform(size, AFFINE, data) => image
im.transform(size, AFFINE, data, filter) => image
Applies an affine transform to the image, and places the result in a
new image with the given size.
Data is a 6-tuple (a, b, c, d, e, f) which contain the first two rows from
an affine transform matrix. For each pixel (x, y) in the output image,
the new value is taken from a position (a x + b y + c, d x + e y + f) in
the input image, rounded to nearest pixel.
This function can be used to scale, translate, rotate, and shear the
original image.




   1311     def toarray(im, dtype=np.uint8):

   1312         """Teturn a 1D array of dtype."""

-> 1313         x_str = im.tostring('raw', im.mode)

   1314         x = np.fromstring(x_str, dtype)

   1315         return x
   
   