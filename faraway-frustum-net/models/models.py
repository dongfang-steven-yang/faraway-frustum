from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, Activation, GlobalAveragePooling2D, Dropout, Flatten
from tensorflow import keras


class BEV2dCentroid:

    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.model = self._build_model_3()

    def _build_model(self):

        inputs = keras.Input(shape=self.input_shape)
        x = Conv2D(32, 3, strides=2, padding="same")(inputs)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        x = Conv2D(32, 3, padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = GlobalAveragePooling2D()(x)

        x = Dropout(0.5)(x)

        outputs = Dense(2, activation='relu')(x)

        return keras.Model(inputs, outputs)

    def _build_model_2(self):

        inputs = keras.Input(shape=self.input_shape)
        x = Conv2D(32, 3, strides=2, padding="same")(inputs)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        x = Conv2D(32, 3, padding="same")(x)
        x = Conv2D(16,3,activation='relu')(x)
        x = Flatten()(x)

        x = Dropout(0.5)(x)

        outputs = Dense(2, activation='relu')(x)

        return keras.Model(inputs, outputs)

    def _build_model_3(self):
        
        #base_model = keras.applications.resnet50.ResNet50(weights=None,include_top=False,input_shape=self.input_shape)
        base_model = keras.applications.MobileNet(weights=None,include_top=False,input_shape=self.input_shape)

        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.5)(x)

        outputs = Dense(7, activation='relu')(x)

        return keras.Model(base_model.input, outputs)

    def _build_model_4(self):
        base_model = keras.applications.resnet50.ResNet50(weights=None,include_top=False,input_shape=self.input_shape)
        # base_model = keras.applications.MobileNet(weights=None, include_top=False, input_shape=self.input_shape)

        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.5)(x)

        outputs = Dense(7, activation='relu')(x)

        return keras.Model(base_model.input, outputs)

    def _build_model_centroid(self):
        base_model = keras.applications.resnet50.ResNet50(weights=None,include_top=False,input_shape=self.input_shape)
        # base_model = keras.applications.MobileNet(weights=None, include_top=False, input_shape=self.input_shape)

        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.5)(x)

        outputs = Dense(3, activation='relu')(x)

        return keras.Model(base_model.input, outputs)