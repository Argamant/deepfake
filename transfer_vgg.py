# example of loading the vgg16 model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input

# load model
model = VGG16()
# summarize the model
model.summary()


# load an image from file
image = load_img('dog.jpg', target_size=(224, 224))
# convert the image pixels to a numpy array
image = img_to_array(image)
# reshape data for the model
image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
# prepare the image for the VGG model
image = preprocess_input(image)