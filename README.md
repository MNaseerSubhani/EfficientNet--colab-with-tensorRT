The code creators are Rawan and Faris.

# Classifier report

## Model types

There are 3 types, keras (no optimization): .h5

### .h5 model:

- Faster to save and load
- Slower inference time
- Can access input shape information with model.input_shape
- Inference command: `model.predict(image)`

### .trt model:

- Takes a very long time to save and load (could be more than 30 minutes to save)
    - the reason is that protobuf uses python implementation by default, this could be sped up by compiling the cpp implementation
- Fastest inference time for jetsons
- Don’t know how to access model input shape
- Inference command:

    ```python
    func = model.signatures['serving_default']
    func = func(tf.dtypes.cast(input, tf.float32))
    key = list(func.keys())[0]
    results = func[key].numpy()
    ```

### .tflite model:

- Quantization is not done yet.
- Faster inference time
- Faster loading and saving time
- Don’t know how to access model input shape
- Inference command: `model.predict(image)`
- Can create a tflite model using `image_classifier.create()`
    - Only works on tf 2.4
    - Needs to generate data using `ImageClassifierDataLoader.from_folder(data_path)`
    - Very hard to access data using the data loader. To make it work I created this function:

      ```python
      def lite_data_generation(self, data_path):
        data = ImageClassifierDataLoader.from_folder(data_path)
        train_data, test_data = data.split(0.8)
        test_data, val_data = test_data.split(0.5)
        
        num_of_trianing_images = 0
        num_of_testing_images = 0
        for i, (image, label) in enumerate(train_data.gen_dataset().unbatch().take(-1)):
          num_of_trianing_images =+ 1
          if i == 0:
              self.image_input_tensor = tf.convert_to_tensor(image.numpy(), np.float32)
        self.num_of_trianing_images = num_of_trianing_images
        for i, (image, label) in enumerate(test_data.gen_dataset().unbatch().take(-1)):
          num_of_testing_images =+ 1
        self.num_of_testing_images = num_of_testing_images
        
        train_ds = list(train_data.gen_dataset())
        
        self.training_images = tf.expand_dims(train_ds[0][0][0][:][:],0)
        for i in range(1, len(train_ds)):
          self.training_images = tf.concat([self.training_images, tf.expand_dims(train_ds[i][0][0][:][:],0)] , 0)
        test_ds = list(train_data.gen_dataset())
        
        self.testing_images = tf.expand_dims(test_ds[0][0][0][:][:],0)
        labels = []
        labels.append(np.array(test_ds[0][:][:][:][:][:][1])[0])
        for i in range(1, len(test_ds)):
          self.testing_images = tf.concat([self.testing_images, tf.expand_dims(test_ds[i][0][0][:][:],0)] , 0)
          labels.append(np.array(test_ds[i][:][:][:][:][:][1])[0])
        self.test_labels = labels
        return train_data, test_data, val_data
      ```

## Augmentation

- Some defects appear on the edges and therefore we must be careful with data augmentation. One solution is to apply zoom out with zero padding before
  applying any augmentation to make sure no edges are being cut.
- Color augmentations include histogram normalization and grayscale together.

## Input preprocessing (EffecientnetB0)

- height, width = 224, 224
- Use belt in: `tf.keras.applications.efficientnet.preprocess_input()`
- Remove images that contain aspect ratio > 0.5
- For directories created by Apple macOS a .DS_Store must be removed from the dataset directory

## Editing threshold

Applying threshold now only works on two classes classification. Further work needs to be done on 3 or more classes classification. How to use `Classifier` class

- `classifier()`: a single function call that trains a new model and saves it as .h5 or .tflite or .trt
- `evaluate()`: Evaluate the model on the testing data set and show accuracy and loss results and a confusion matrix
- `predict()`: returns prediction results of an input. Could be a single image or a batch of images
- `load_model()`: load an existing trained model from a path
    - With `opt_mode="h5"` a .h5 file path must be passed
    - With `opt_mode="trt"` or .trt/ a directory path that contains a file named `saved_model.pb` must be passed

## How to use `classifier_interactive.py` (autoTrainer class)

- the auto trainer class takes 8 arguments:
    - path to data source
    - path to a trained model to load
    - path to save a newly trained model
    - Boolean value (show_dir_loading_model) to whether show the widgets of choosing a directory to load the model
    - Boolean value (show_dir_saving_model) to whether show the widgets of choosing a directory to save the model
    - Boolean value (show_dir_dataset) to whether show the widgets of choosing a directory to load the datset
    - a classifier object
    - path to save the threshold value
- `display_auto_train_interface()`: this function shows the interface. It allows the user to choose whether to lead a trained model or train a new
  model. After this selection some more options might appear depending on the Boolean arguments passed to the autotrainer class. The user then
  loads/trains a model
- `display_prediction_making()`: allows the user to evaluate the model on the testing dataset. There are two options:
    - take the maximum probability: No further action needed from the user. A confusion matrix will appear
    - edit a threshold: the user will be able to edit the threshold depending on a plot that shows the distribution of the probabilities. Then the
      user can test the modified threshold on the testing dataset. A confusion matrix will appear
