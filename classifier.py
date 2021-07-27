import logging
import os
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm.auto import tqdm

try:
    from sklearn.metrics import confusion_matrix, classification_report
except ImportError:
    confusion_matrix, classification_report = None, None

logging.getLogger('tensorflow').setLevel(logging.FATAL)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.preprocessing.image import ImageDataGenerator

sns.set_style('darkgrid')

in_notebook = sys.argv[0].endswith('ipykernel_launcher.py') and len(sys.argv) == 3 and sys.argv[-1].endswith('json') and \
              sys.argv[-2] == '-f'

if not in_notebook:
    plt.show = lambda *args, **kwargs: plt.close()


class Classifier:
    """
    ### The classifier function is a general purpose image classifier that can be adapted to most image classification tasks.
    It enables you to select one of 6 pre-trained models. The comments next to the function parameters should  provide
    the information necessary to set the values for your application. Note the function uses a custom calback that
    adjust the learning rate. Initially the learning rate is adjusted by monitoring accuracy. Once the model accuracy
    exceeds the level set by parameter threshold, the learning rate is adjusted based on monitoring validation loss.
    """

    def __init__(self, saving_dir_probability, checkpoint_path, opt_mode='trt'):
        assert opt_mode in ['trt', 'tflite', 'h5']
        self.classification_model = None
        self.decision_threshold = 0.5
        self.saving_dir_probability = saving_dir_probability
        self.checkpoint_path = checkpoint_path
        self.preprocess_input = tf.keras.applications.efficientnet.preprocess_input
        self.opt_mode = opt_mode
        self.train_gen = None
        self.trt_model_func = None
        self.model_path = None

        # tflite stuff
        self.output_details = None
        self.input_details = None

        self.representitive_data_pbar = None

    def classifier(self,
                   my_dir,  # main directory
                   model_type='EfficientNetB0',
                   # select from Mobilenet','MobilenetV2', 'VGG19','InceptionV3', 'ResNet50V2', 'NASNetMobile', 'DenseNet201
                   structure=1,  # 3 for train,test, valid directories, 2 for train, test directories 1 for a single directory dir
                   v_split=.10,  # only valid if structure =1 or 2. Then it is percentage of training images that will be used for validation
                   epochs=100,  # number of epochs to run the model for
                   freeze=False,  # if False all layers of the model are trained for epoch epochs, if True layersof basemodel are not trained
                   fine_tune_epochs=10,  # only used when freeze=True, then all model layers are trained for this value epochs after  initial epochs
                   height=224,  # height of images to be used by the model
                   width=224,  # width of images to be used by the model
                   bands=3,  # bands in image 3 for rgb 1 for grey scale
                   batch_size=32,  # batch size used in generators
                   lr=.001,  # initial learning rate
                   patience=1,  # number of epochs with no performance improvement before learning rate is adjusted
                   stop_patience=3,  # number of times learning rate can be adjusted with performance improvement before stopping training
                   threshold=.90,  # float value is training accuracy<threshold adjust lr on accuracy, above threshold adjust on validation loss
                   dwell=False,  # if True when there is no performance improvement model weights set back to best_weights
                   factor=.5,  # float <1 factor to multiply current learning rate by if performance does not improve
                   dropout=.7,  # dropout float<1 defines dropout factor
                   print_code=10,  # Integer if 0 no file classifications are printed, if >0 is the maximum number misclassified
                   # file classifications that will be printed out
                   neurons_a=128,  # number of neurons in Dense layer between base model and top classification layer
                   metrics=None,  # create a list of desired metrics Note 'accuracy' metric is automatically added to the list of metrics
                   visualize=False,  #whether or not to visualine hidden layers
                   model_saving_path="./my_model",
                   weight_saving_path="./model_weights"):
        assert model_type == 'EfficientNetB0', "Currently only EfficientNetB0 is supported, " \
                                               "because the preprocessing code is hardcoded."
        assert structure == 1, 'only structure==1 is supported'

        if metrics is None:
            metrics = ['accuracy']
        if 'accuracy' not in metrics:
            metrics.append('accuracy')

        my_dir = os.path.join(my_dir, 'train')
        train_dir, test_dir, valid_dir = my_dir, my_dir, my_dir

        train_gen, test_gen, valid_gen = self.make_gens(structure, train_dir, test_dir, valid_dir, height, width, bands, batch_size, v_split)
        self.train_gen = train_gen
        # display some training images
        labels = show_training_samples(train_gen)
        #create a list of classes from the sub directory names
        class_list = os.listdir(train_dir)  # list of class directories within the train directory
        #determine the number of classes
        class_count = len(class_list)
        # make the model based on input parameters
        self._model = make_model(model_type, neurons_a, class_count, width, height, bands, lr, freeze, dropout, metrics)
        # determine class weights to handle imbalanced data sets
        # create a custom callback using input parameters
        self._lra = LRA(model=self._model, epochs=epochs, patience=patience, stop_patience=stop_patience, threshold=threshold, factor=factor,
                        dwell=dwell, model_name=model_type, freeze=freeze, end_epoch=epochs - 1, action="Training")

        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=self.checkpoint_path,
            save_weights_only=True,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True
        )
        callbacks = [self._lra, model_checkpoint_callback]
        # determine batch sizes for generators and steps per epoch for model.fit
        test_batch_size = batch_size

        class_weights = get_class_weights(labels)

        results = self._model.fit(x=train_gen,
                                  #steps_per_epoch=int(len(train_gen.labels) / batch_size),
                                  validation_data=valid_gen,
                                  #validation_steps=int( len(valid_gen.labels)  / batch_size),
                                  epochs=epochs,
                                  verbose=0,
                                  callbacks=callbacks,
                                  shuffle=True,
                                  initial_epoch=0,
                                  class_weight=class_weights
                                  )

        tr_plot(results, 0)  # plot the loss and accuracy metrics
        self._model.set_weights(self._lra.best_weights)  # load the best weights saved during training

        #print_info( test_dir, test_gen, preds, print_code )
        if freeze:  # Start Fine Tunning
            msg = 'Starting Fine Tuning of Model data from last epoch is shown below'
            print_in_color(msg, (255, 255, 0), (55, 65, 80))
            for layer in self._model.layers:  # make all layers trainable
                layer.trainable = True
            start_epoch = epochs
            total_epochs = start_epoch + fine_tune_epochs
            self._lra = LRA(model=self._model,
                            epochs=fine_tune_epochs,
                            patience=patience,
                            stop_patience=stop_patience,
                            threshold=threshold,
                            factor=factor,
                            dwell=dwell,
                            model_name=model_type,
                            freeze=freeze,
                            end_epoch=epochs - 1,
                            action="Fine Tuning")
            callbacks = [self._lra, model_checkpoint_callback]
            # train model for fine tunning
            data = self._model.fit(x=train_gen,
                                   #steps_per_epoch=int(len(train_gen.labels) / batch_size),
                                   #validation_steps=int( len(valid_gen.labels)  / batch_size),
                                   epochs=total_epochs,
                                   verbose=0,
                                   callbacks=callbacks,
                                   validation_data=valid_gen,
                                   initial_epoch=start_epoch,
                                   shuffle=True,
                                   class_weight=class_weights
                                   )

            tr_plot(data, start_epoch)  # plot training graph
            self._model.set_weights(self._lra.best_weights)  # load best weights from finetuning

        #tf.saved_model.save(model, model_saving_path)
        e_dict = self._model.evaluate(test_gen, batch_size=test_batch_size, verbose=0, steps=None, return_dict=True)

        acc = display_eval_metrics(e_dict)
        model_name = model_type + '_' + str(acc)[:str(acc).rfind('.') + 3] + '.' + self.opt_mode
        model_path = os.path.join(model_saving_path, model_name)

        self._model.save(model_path.replace('.' + self.opt_mode, '.h5'))  # save .h5 first
        print('saved h5 model in', model_path.replace('.' + self.opt_mode, '.h5'))
        try:
            self.save_model(self._model, model_path)
        except Exception as e:
            import traceback
            traceback.print_exc(e)
            print("Failed to save .trt model but you can load the .h5 model and convert it later")

        msg = 'Training complete'
        print_in_color(msg, (0, 255, 0), (55, 65, 80))

    def save_model(self, model, model_path):
        if self.opt_mode == 'tflite':
            raise NotImplementedError()
            # with open(model_path, "wb") as f:
            #     f.write(model)
            # converter = tf.lite.TFLiteConverter.from_keras_model(model)
            # #converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_LATENCY]
            # model = converter.convert()
            #
            # self.classification_model = tf.lite.Interpreter(model_path=model_path)
            # self.input_details = self.classification_model.get_input_details()
            # self.output_details = self.classification_model.get_output_details()
        elif self.opt_mode == 'trt':
            # https://blog.tensorflow.org/2021/01/leveraging-tensorflow-tensorrt-integration.html
            from tensorflow.python.compiler.tensorrt import trt_convert
            tf.saved_model.save(model, model_path)

            print("converting and quantizing model (may take up to 30minutes)", end=' ... ')
            params = tf.experimental.tensorrt.ConversionParams(precision_mode=trt_convert.TrtPrecisionMode.INT8)
            converter = trt_convert.TrtGraphConverterV2(input_saved_model_dir=model_path,
                                                        conversion_params=params)
            converter.convert(calibration_input_fn=self.representitive_data)
            converter.build(input_fn=self.representitive_data)
            converter.save(model_path)
            print('saved trt model, now loading ...')
            self.model_path = model_path
            self.load_model(model_path, 'trt')
            print('loaded trt model')
        elif self.opt_mode == 'h5':
            self.classification_model = model
            model.save(model_path)

        print("model saved at {}".format(model_path))

    def representitive_data(self):
        if self.representitive_data_pbar is None:
            #TODO: add the total number of images
            self.representitive_data_pbar = tqdm(desc='Quantizing model: iterating over dataset', unit='image')
        images, labels = next(self.train_gen)
        self.representitive_data_pbar.update(1)
        yield tf.expand_dims(images[0], axis=0),

    def evaluate(self, model_loading_path, data_dir, decision_threshold=0.5,
                 bands=3, height=224, width=224, batch_size=32, print_code=10,
                 v_split=.10, edit_threshold=False, is_print_info=True):
        """ function to evaluate a model """

        structure = 1
        train_dir = os.path.join(data_dir, 'train')

        train_gen, test_gen, valid_gen = self.make_gens(structure, train_dir, train_dir, train_dir, height, width, bands, batch_size, v_split)

        if self.opt_mode == 'tflite':
            model = (self.classification_model
                     if self.classification_model is not None else
                     self.load_model(model_loading_path, opt_mode='tflite'))

            model.resize_tensor_input(self.input_details[0]['index'], [len(test_gen.labels), height, width, 3])
            model.allocate_tensors()
            model.set_tensor(self.input_details[0]['index'], test_gen.next()[0])
            model.invoke()
            preds_orginal = model.get_tensor(self.output_details[0]['index'])
        elif self.opt_mode == 'trt':
            model = self.classification_model
            preds_orginal = []
            for i, (images, labels) in zip(tqdm(range(len(test_gen))), test_gen):
                func = self.trt_model_func(tf.dtypes.cast(images, tf.float32))
                key = list(func.keys())[0]
                preds_orginal.append(func[key].numpy())
            preds_orginal = np.concatenate(preds_orginal)
        else:
            raise ValueError(self.opt_mode + ' is not a valid opt_mode')

        preds = preds_orginal.copy()

        if edit_threshold:
            preds[:, 0] = (preds[:, 0] >= decision_threshold).astype(int)
            preds[:, 1] = (np.invert(preds[:, 0].astype(bool))).astype(int)

        if is_print_info:
            print_info(test_gen, preds, print_code)

        return preds_orginal, test_gen.labels

    def predict(self, image, edit_threshold=False, opt_mode=None):
        if opt_mode is None:
            opt_mode = self.opt_mode

        def predict_h5(image, edit_threshold=False):
            preds = self.classification_model.predict(image)

            if edit_threshold:
                preds[:, 0] = (preds[:, 0] >= self.decision_threshold).astype(int)
                preds[:, 1] = (np.invert(preds[:, 0].astype(bool))).astype(int)

            return preds

        def predict_trt(image, edit_threshold=False):
            func = self.trt_model_func(tf.dtypes.cast(image, tf.float32))
            key = list(func.keys())[0]
            preds = func[key].numpy()

            if edit_threshold:
                preds[:, 0] = (preds[:, 0] >= self.decision_threshold).astype(int)
                preds[:, 1] = (np.invert(preds[:, 0].astype(bool))).astype(int)

            return preds

        def predict_tflite(images, edit_threshold=False):
            """ images: batch of images (batch, width, height, channel) """
            preds = self.classification_model.predict(images)

            if edit_threshold:
                preds[:, 0] = (preds[:, 0] >= self.decision_threshold).astype(int)
                item_pred = preds[:, 0] == 1
            else:
                item_pred = max(preds[:, 0], preds[:, 1])
            return item_pred

        if opt_mode == 'tflite':
            return predict_tflite(image, edit_threshold)
        elif opt_mode == 'trt':
            return predict_trt(image, edit_threshold)
        elif opt_mode == 'h5':
            return predict_h5(image, edit_threshold)

    def load_model(self, image, opt_mode=None):
        if opt_mode is None:
            opt_mode = self.opt_mode

        def load_trt_model(loading_path):
            if self.classification_model is None:
                self.classification_model = tf.saved_model.load(loading_path)
            if self.trt_model_func is None:
                self.trt_model_func = self.classification_model.signatures['serving_default']

        def load_h5_model(loading_path):
            self.classification_model = tf.keras.models.load_model(loading_path)
            return self.classification_model

        def load_tflite_model(loading_path):
            if self.classification_model is None:
                model = tf.lite.Interpreter(model_path=loading_path)
            else:
                model = self.classification_model

            input_details = model.get_input_details()
            output_details = model.get_output_details()

            self.output_details = output_details
            self.input_details = input_details

            self.classification_model = model
            return model

        if opt_mode == 'tflite':
            return load_tflite_model(image)
        elif opt_mode == 'trt':
            return load_trt_model(image)
        elif opt_mode == 'h5':
            return load_h5_model(image)
        else:
            raise Exception(f"opt_mode {opt_mode} recognized")

    def make_gens(self, structure, train_dir, test_dir, valid_dir, height, width, bands, batch_size, v_split):
        assert structure == 1, 'only structure==1 is supported'

        total_data = 0

        for directory in os.listdir(train_dir):
            files = os.listdir(os.path.join(train_dir, directory))
            try:
                files.remove(".DS_Store")
            except ValueError:
                pass

            try:
                files.remove(".ipynb_checkpoints")
            except ValueError:
                pass
            total_data += len(files)

        assert total_data > 2, 'more than 2 images are required'

        if bands == 3:
            color_mode = 'rgb'
        else:
            color_mode = 'grayscale'

        train_gen = ImageDataGenerator(
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            fill_mode='constant',
            cval=0,
            preprocessing_function=self.preprocess_input,
            validation_split=v_split,
            horizontal_flip=True,
            zoom_range=.2,
            rotation_range=360.0,
            dtype='float32',
        ).flow_from_directory(
            train_dir, target_size=(width, height),
            batch_size=batch_size, seed=123,
            class_mode='categorical', color_mode=color_mode,
            subset='training')

        while len(train_gen.labels) == total_data:
            v_split += 0.1

            train_gen = ImageDataGenerator(
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                fill_mode='constant',
                cval=0,
                preprocessing_function=self.preprocess_input,
                validation_split=v_split,
                horizontal_flip=True,
                zoom_range=.2,
                rotation_range=360.0,
                dtype='float32',
            ).flow_from_directory(
                train_dir, target_size=(width, height),
                batch_size=batch_size, seed=123,
                class_mode='categorical', color_mode=color_mode,
                subset='training')

        valid_gen = ImageDataGenerator(
            preprocessing_function=self.preprocess_input,
            validation_split=v_split,
            horizontal_flip=False,
            dtype='float32',
        ).flow_from_directory(
            train_dir, target_size=(width, height),
            batch_size=batch_size, seed=123, shuffle=False,
            class_mode='categorical', color_mode=color_mode,
            subset='validation')

        test_gen = valid_gen
        return train_gen, test_gen, valid_gen


class LRA(keras.callbacks.Callback):
    """
    ### Create subclass of callback class as custom callback to adjust learning rate and save best weights

    Parameters:
        patience (int): is an integer that specifies how many consecutive epoch can occur until learning rate is adjusted
        threshold (float): is a float. It specifies that if training accuracy is above this level learning rate will be adjusted based on validation loss
        factor (float): is a float <1 that specifies the factor by which the current learning rate will be multiplied by
            class variable self.best_weights stores the model weights for the epoch with the lowest validation loss
            after train set the model weights with model.load_weights(self.best_weights) then do predictions on the test set
    """

    def __init__(self, model, patience, epochs, stop_patience, threshold, factor, dwell, model_name, freeze, end_epoch, action):
        super(LRA, self).__init__()
        self.model = model
        self.patience = patience  # specifies how many epochs without improvement before learning rate is adjusted
        self.stop_patience = stop_patience
        self.threshold = threshold  # specifies training accuracy threshold when lr will be adjusted based on validation loss
        self.factor = factor  # factor by which to reduce the learning rate
        self.dwell = dwell
        self.lr = float(tf.keras.backend.get_value(model.optimizer.lr))  # get the initiallearning rate and save it in self.lr
        self.highest_tracc = 0.0  # set highest training accuracy to 0
        self.lowest_vloss = np.inf  # set lowest validation loss to infinity
        self.count = 0  # initialize counter that counts epochs with no improvement
        self.stop_count = 0  # initialize counter that counts how manytimes lr has been adjustd with no improvement
        self.end_epoch = end_epoch  # value of the number of epochs to run

        self.pbar = tqdm(desc=action, total=epochs, unit='epoch')

        self.best_weights = self.model.get_weights()  # set a class vaiable so weights can be loaded after training is completed
        if freeze:
            msgs = f' Starting training using  base model {model_name} with weights frozen to imagenet weights initializing LRA callback'
        else:
            msgs = f' Starting training using base model {model_name} training all layers '
        #print_in_color(msgs, (244, 252, 3), (55, 65, 80))

    def on_epoch_end(self, epoch, logs=None):  # method runs on the end of each epoch
        lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))  # get the current learning rate
        v_loss = logs.get('val_loss')  # get the validation loss for this epoch
        acc = logs.get('accuracy')  # get training accuracy

        self.pbar.set_postfix(v_loss=v_loss, v_acc=acc)
        self.pbar.update(1)

        if self.highest_tracc > acc:
            self.highest_tracc = acc
            self.best_weights = self.model.get_weights()
        if acc < self.threshold:  # if training accuracy is below threshold adjust lr based on training accuracy
            if acc > self.highest_tracc:  # training accuracy improved in the epoch
                msg = f' training accuracy improved from  {self.highest_tracc:7.4f} to {acc:7.4f} learning rate held at {lr:10.8f}'
                self.highest_tracc = acc  # set new highest training accuracy
                self.best_weights = self.model.get_weights()  # traing accuracy improved so save the weights
                self.count = 0  # set count to 0 since training accuracy improved
                self.stop_count = 0  # set stop counter to 0
                if v_loss < self.lowest_vloss:
                    self.lowest_vloss = v_loss
            else:
                # training accuracy did not improve check if this has happened for patience number of epochs
                # if so adjust learning rate
                if self.count >= self.patience - 1:
                    self.lr = lr * self.factor  # adjust the learning by factor
                    tf.keras.backend.set_value(self.model.optimizer.lr, self.lr)  # set the learning rate in the optimizer
                    self.count = 0  # reset the count to 0
                    self.stop_count = self.stop_count + 1
                    if self.dwell:
                        self.model.set_weights(self.best_weights)  # return to better point in N space
                    else:
                        if v_loss < self.lowest_vloss:
                            self.lowest_vloss = v_loss
                    msgs = f' training accuracy {acc:7.4f} < highest accuracy of {self.highest_tracc:7.4f} '
                    msg = msgs + f' for {self.patience} epochs, lr adjusted to {self.lr:10.8f}'
                else:
                    self.count = self.count + 1  # increment patience counter
                    msg = f' training accuracy {acc:7.4f} < highest accuracy of {self.highest_tracc:7.4f} '
                    #print_in_color(msg, (255,255,0), (55,65,80))
        else:  # training accuracy is above threshold so adjust learning rate based on validation loss
            if v_loss < self.lowest_vloss:  # check if the validation loss improved
                msgs = f' validation loss improved from {self.lowest_vloss:8.5f} to {v_loss:8.5}, saving best weights'
                if acc > self.highest_tracc:
                    self.highest_tracc = acc
                msg = msgs + f' learning rate held at {self.lr:10.8f}'
                #print_in_color(msg, (0,255,0), (55,65,80))
                self.lowest_vloss = v_loss  # replace lowest validation loss with new validation loss
                self.best_weights = self.model.get_weights()  # validation loss improved so save the weights
                self.count = 0  # reset count since validation loss improved
                self.stop_count = 0
            else:  # validation loss did not improve
                if self.count >= self.patience - 1:
                    self.lr = self.lr * self.factor  # adjust the learning rate
                    self.stop_count = self.stop_count + 1  # increment stop counter because lr was adjusted
                    msgs = f' val_loss of {v_loss:8.5f} > {self.lowest_vloss:8.5f} for {self.patience} epochs'
                    msg = msgs + f', lr adjusted to {self.lr:10.8f}'
                    self.count = 0  # reset counter
                    tf.keras.backend.set_value(self.model.optimizer.lr, self.lr)  # set the learning rate in the optimizer
                    if self.dwell:
                        self.model.set_weights(self.best_weights)  # return to better point in N space
                else:
                    self.count = self.count + 1  # increment the patience counter
                    msg = f' validation loss of {v_loss:8.5f} > {self.lowest_vloss:8.5f}'
                    #print_in_color(msg, (255,255,0), (55,65,80))
                if acc > self.highest_tracc:
                    self.highest_tracc = acc
        if epoch == self.end_epoch:
            print_in_color(msg, (255, 255, 0), (55, 65, 80))  # print out data for the final epoch
        if self.stop_count > self.stop_patience - 1:  # check if learning rate has been adjusted stop_count times with no improvement
            msg = f' training has been halted at epoch {epoch + 1} after {self.stop_patience} adjustments of learning rate with no improvement'
            self.model.stop_training = True  # stop training


def make_model(model_type, neurons_a, class_count, width, height, bands, lr=0.001, freeze=False, dropout=0.5, metrics=('accuracy',)):
    img_shape = (width, height, bands)
    # model_list = ['MobileNet', 'MobilenetV2', 'VGG19', 'InceptionV3', 'ResNet50V2', 'NASNetMobile', 'DenseNet201']
    # if model_type not in model_list:
    #     msg = f'ERROR the model name you specified {model_type} is not an allowed model name'
    #     print_in_color(msg, (255, 0, 0), (55, 65, 80))
    #     return None

    base_model = getattr(tf.keras.applications, model_type)(
        include_top=False, input_shape=img_shape, pooling='max', weights='imagenet', drop_connect_rate=.4)

    if freeze:
        for layer in base_model.layers:  #train top 20 layers of base model
            layer.trainable = False

    # clf = tf.keras.models.Sequential([
    #     tf.keras.layers.Input(base_model.output_shape[1:]),
    #     keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001),
    #     Dense(neurons_a,
    #           kernel_regularizer=regularizers.l2(l=0.016),
    #           activity_regularizer=regularizers.l1(0.006),
    #           bias_regularizer=regularizers.l1(0.006),
    #           activation='relu',
    #           kernel_initializer=tf.keras.initializers.GlorotUniform(seed=123)),
    #     Dropout(rate=dropout, seed=123),
    #     Dense(class_count, activation='softmax', kernel_initializer=tf.keras.initializers.GlorotUniform(seed=123)),
    # ], name='clf')
    # model = tf.keras.models.Sequential([
    #     base_model,
    #     clf
    # ], name='ft_model')

    x = base_model.output
    x = keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(x)
    x = Dense(neurons_a,
              kernel_regularizer=regularizers.l2(l=0.016),
              activity_regularizer=regularizers.l1(0.006),
              bias_regularizer=regularizers.l1(0.006),
              activation='relu',
              kernel_initializer=tf.keras.initializers.GlorotUniform(seed=123))(x)
    x = Dropout(rate=dropout, seed=123)(x)
    output = Dense(class_count, activation='softmax', dtype='float32', kernel_initializer=tf.keras.initializers.GlorotUniform(seed=123))(x)
    model = Model(inputs=base_model.input, outputs=output)

    model.compile(Adamax(lr=lr), loss='categorical_crossentropy', metrics=metrics)
    return model


def show_training_samples(gen):
    class_dict = gen.class_indices
    new_dict = {}
    # make a new dictionary with keys and values reversed
    for key, value in class_dict.items():  # dictionary is now {numeric class label: string of class_name}
        new_dict[value] = key
    images, labels = next(gen)  # get a sample batch from the generator
    plt.figure(figsize=(20, 20))
    length = len(labels)
    if length < 10:  #show maximum of 25 images
        r = length
    else:
        r = 10
    for i in range(r):
        plt.subplot(5, 5, i + 1)
        image = (images[i] + 1) / 2  # scale images between 0 and 1 because pre-processor set them between -1 and +1
        plt.imshow(image.astype('uint8'))
        index = np.argmax(labels[i])
        class_name = new_dict[index]
        plt.title(class_name, color='blue', fontsize=16)
        plt.axis('off')
    plt.show()
    return labels


def print_in_color(txt_msg, fore_tupple, back_tupple, ):
    #prints the text_msg in the foreground color specified by fore_tupple with the background specified by back_tupple
    #text_msg is the text, fore_tupple is foregroud color tupple (r,g,b), back_tupple is background tupple (r,g,b)
    rf, gf, bf = fore_tupple
    rb, gb, bb = back_tupple
    msg = '{0}' + txt_msg
    mat = '\u001B[38;2;{0};{1};{2};48;2;{3};{4};{5}m'.format(rf, gf, bf, rb, gb, bb)
    print(msg.format(mat), flush=True)
    print('\33[0m', flush=True)  # returns default print color to back to black
    return


def my_plot_confusion_matrix(cm,
                             target_names,
                             title='Confusion matrix',
                             cmap=None,
                             normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citation
    ---------
    <http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html>

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    # misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.grid(b=None)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')  #\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()


def get_class_weights(y):
    num_of_classes = len(y[0])
    total_data = len(y)
    weights = {}

    for i in range(num_of_classes):
        y_ = y[:, i].tolist()
        class_weight = (1.0 / (y_.count(1))) * ((total_data) / 2.0)
        weights.update({i: class_weight})
    return weights


def display_eval_metrics(e_data):
    """function to print metrics from model.evaluate"""
    msg = 'Model Metrics after Training'
    print_in_color(msg, (255, 255, 0), (55, 65, 80))
    msg = '{0:^24s}{1:^24s}'.format('Metric', 'Value')
    print_in_color(msg, (255, 255, 0), (55, 65, 80))
    for key, value in e_data.items():
        print(f'{key:^24s}{value:^24.5f}')
    acc = e_data['accuracy'] * 100
    return acc


def print_info(data_gen, preds, print_code=10):
    new_dict = {v: k for (k, v) in data_gen.class_indices.items()}
    labels = data_gen.labels
    error_list = []
    true_class = []
    pred_class = []
    prob_list = []
    error_indices = []
    y_pred = []
    classes = list(new_dict.values())  # list of string of class names
    errors = 0
    for i, p in enumerate(preds):
        pred_index = np.argmax(p)
        true_index = labels[i]  # labels are integer values
        if pred_index != true_index:  # a misclassification has occurred
            error_list.append(data_gen.filenames[i])
            true_class.append(new_dict[true_index])
            pred_class.append(new_dict[pred_index])
            prob_list.append(p[pred_index])
            error_indices.append(true_index)
            errors = errors + 1
        y_pred.append(pred_index)

    if print_code != 0:
        if errors > 0:
            if print_code > errors:
                r = errors
            else:
                r = print_code
        else:
            msg = 'With accuracy of 100 % there are no errors to print'
            print_in_color(msg, (0, 255, 0), (55, 65, 80))

    if errors > 0:
        plot_bar = []
        plot_class = []
        for key, value in new_dict.items():
            count = error_indices.count(key)
            if count != 0:
                plot_bar.append(count)  # list containg how many times a class c had an error
                plot_class.append(value)  # stores the class
        fig = plt.figure()
        fig.set_figheight(len(plot_class) / 3)
        fig.set_figwidth(10)
        plt.style.use('fivethirtyeight')
        for i in range(0, len(plot_class)):
            c = plot_class[i]
            x = plot_bar[i]
            plt.barh(c, x, )
            plt.title(' Errors by Class on Test Set')

    if len(classes) <= 20:
        # create a confusion matrix and a test report

        class_names = [new_dict[k] for k in range(len(new_dict))]

        y_true = np.array(labels)
        y_pred = np.array(y_pred)
        if confusion_matrix is not None and classification_report is not None:
            cm = confusion_matrix(y_true, y_pred)
            clr = classification_report(y_true, y_pred)
            print("Classification Report:\n----------------------\n", clr)
        else:
            warnings.warn("Couldn't import sklearn confusion matrix")
            cm = np.zeros((len(classes), len(classes)))
            for a, p in zip(y_true, y_pred):
                cm[a][p] += 1

        my_plot_confusion_matrix(cm, normalize=False, target_names=class_names, title="Confusion Matrix")


def tr_plot(tr_data, start_epoch):
    """Plot the training and validation data"""
    tacc = tr_data.history['accuracy']
    tloss = tr_data.history['loss']
    vacc = tr_data.history['val_accuracy']
    vloss = tr_data.history['val_loss']
    Epoch_count = len(tacc) + start_epoch
    Epochs = []
    for i in range(start_epoch, Epoch_count):
        Epochs.append(i + 1)
    index_loss = np.argmin(vloss)  #  this is the epoch with the lowest validation loss
    val_lowest = vloss[index_loss]
    index_acc = np.argmax(vacc)
    acc_highest = vacc[index_acc]
    plt.style.use('fivethirtyeight')
    sc_label = 'best epoch= ' + str(index_loss + 1 + start_epoch)
    vc_label = 'best epoch= ' + str(index_acc + 1 + start_epoch)
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 8))
    axes[0].plot(Epochs, tloss, 'r', label='Training loss')
    axes[0].plot(Epochs, vloss, 'g', label='Validation loss')
    axes[0].scatter(index_loss + 1 + start_epoch, val_lowest, s=150, c='blue', label=sc_label)
    axes[0].set_title('Training and Validation Loss')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[1].plot(Epochs, tacc, 'r', label='Training Accuracy')
    axes[1].plot(Epochs, vacc, 'g', label='Validation Accuracy')
    axes[1].scatter(index_acc + 1 + start_epoch, acc_highest, s=150, c='blue', label=vc_label)
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    plt.tight_layout()
    # plt.style.use('fivethirtyeight')
    plt.show()


def get_bs(dir, b_max):
    # ### determine batch size and steps per epoch
    # dir is the directory containing the samples, b_max is maximum batch size to allow based on your memory capacity
    # you only want to go through test and validation set once per epoch this function determines needed batch size ans steps per epoch
    length = 0
    #get_ipython().system('rm dir/.DS_Store')
    dir_list = os.listdir(dir)

    for d in dir_list:
        d_path = os.path.join(dir, d)
        length = length + len(os.listdir(d_path))
        #print(length)
    batch_size = sorted([int(length / n) for n in range(1, length + 1) if length % n == 0 and length / n <= b_max], reverse=True)[0]

    return batch_size, int(length / batch_size)
