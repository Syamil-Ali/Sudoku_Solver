import tensorflow as tf
import cv2
import numpy as np

# get the model
interpreter = tf.lite.Interpreter(model_path='core/model.tflite')


# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Allocate tensors
interpreter.allocate_tensors()



# doing prediction
def predict_img(img):

    img = img.astype('float32',casting='same_kind') # CONVERSION MUST
    
    img = cv2.resize(img, (64, 64)) / 255 #resize the image
    img = tf.expand_dims(img, axis=2, name=None) #expand the third dimension (for color)
    img = tf.expand_dims(img, axis=0, name=None) #expand the first dimension (batch = 1)
    
    #do prediction
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])

    class_index = np.argmax(prediction, axis = -1)
    probability_val = np.amax(prediction)
    #print(prediction)
    
    if probability_val > 0.90:
        return class_index[0]
    else:
        return 0
    

