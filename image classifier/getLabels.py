import tensorflow as tf

model = tf.lite.Interpreter(
    model_path='model.tflite')

print(model.get_signature_list())