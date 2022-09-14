from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

list1 = []
list2 = []
rank = []
k=0

# Load the model
model = load_model('./model/keras_model.h5')

# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1.
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
# Replace this with the path to your image
image = Image.open('./picture/test1.jpg')
#resize the image to a 224x224 with the same strategy as in TM2:
#resizing the image to be at least 224x224 and then cropping from the center
size = (224, 224)
image = ImageOps.fit(image, size, Image.ANTIALIAS)

#turn the image into a numpy array
image_array = np.asarray(image)
# Normalize the image
normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
# Load the image into the array
data[0] = normalized_image_array

# run the inference
prediction = model.predict(data)
print(prediction)

for i in range(0,len(prediction[0])):
    list1.append(prediction[0,i])
    list2.append(prediction[0,i])

list2.sort(reverse=True)
list2 = list2[0:3]
for i in range(1,len(prediction[0])):
    if(list1[i]==list2[0]):
        rank.append(i)
for i in range(1,len(prediction[0])):
    if(list1[i]==list2[1]):
        rank.append(i)
for i in range(1,len(prediction[0])):
    if(list1[i]==list2[2]):
        rank.append(i)

great_dic = { name:value for name, value in zip(rank, list2) }

print(list1)
print(list2)
print(rank)
print(great_dic)