from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

# make a great list
with open("./model/number.txt", "r", encoding="utf8") as ins:
    great_list = []
    for line in ins:
        great_list.append(line.rstrip('\n'))

# Load the model
model = load_model('./model/keras_model.h5')

# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1.
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
# Replace this with the path to your image
image = Image.open('./picture/test5.jpg')
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

# for x in range(0, 9):
#     if(great[ind[x]] == randword):
#         result[class_names[ind[x]]] = round(pred[ind[x]]*100, 2)

# for i in range(0, 3):
#     a=prediction[0,i]
#     print(a)

print(great_list)

great_dic = {string:0 for string in great_list}

# for i in range(0,9):
#     great_dic.update(i=prediction[0,i]

for i in range(0,10):
    great_dic[str(i)] = prediction[0,i]

print(great_dic)

great_dic = sorted(great_dic.items(), key=lambda x: x[1], reverse=True)


print(great_dic)

print(type(great_dic))