import numpy.random as random
import coremltools
from keras.models import load_model
random.seed(1337)  # for reproducibility


print('Loading Keras TensorFlow model from disk');
model = load_model('./cifar-weights.h5');
print('Model loaded, executing conversion')

coreml_model = coremltools.converters.keras.convert(model, input_names="image", image_input_names="image", output_names="prediction", predicted_feature_name="character", image_scale=1.0/255.0);
coreml_model.save("tensormodel.mlmodel")
