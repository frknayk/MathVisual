import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
import matplotlib.pyplot as plt

class OccModel:
    def __init__(self) -> None:
        self.model = None
        self.history = None

    def create_model(self):
        # using sequential model for training
        model = Sequential()

        # 1st layer and taking input in this of shape 100x100x3 ->  100 x 100 pixles and 3 channels
        model.add(Conv2D(32, (3, 3), padding='same', input_shape=(100, 100, 3), activation="relu"))
        model.add(Conv2D(32, (3, 3), activation="relu"))

        # maxpooling will take highest value from a filter of 2*2 shape
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # it will prevent overfitting by making it hard for the model to idenify the images
        model.add(Dropout(0.25))

        model.add(Conv2D(64, (3, 3), padding='same', activation="relu"))
        model.add(Conv2D(64, (3, 3), activation="relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(512, activation="relu"))
        model.add(Dropout(0.5))

        # last layer predicts 16 labels
        model.add(Dense(16, activation="softmax"))

        # Compile the model
        model.compile(
            loss='categorical_crossentropy',
            optimizer="adam",
            metrics=['accuracy']
        )

        model.summary()
        self.model = model
    
    def train(self,X_train,y_train):
        # training the model
        self.history = self.model.fit(
            X_train,
            y_train,
            batch_size=50,
            epochs=200,
            validation_split=0.2,
            shuffle=True
        )

    def plot_acc(self):
        """Plot model accuracy"""
        # displaying the model accuracy
        plt.plot(self.history.history['accuracy'], label='train', color="red")
        plt.plot(self.history.history['val_accuracy'], label='validation', color="blue")
        plt.title('Model accuracy')
        plt.legend(loc='upper left')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.show()

    def plot_loss(self):
        # displaying the model loss
        plt.plot(self.history.history['loss'], label='train', color="red")
        plt.plot(self.history.history['val_loss'], label='validation', color="blue")
        plt.title('Model loss')
        plt.legend(loc='upper left')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.show()
    
    def show_performance_test(self,X_test,y_test):
        score, acc = self.model.evaluate(X_test, y_test)
        print('Test score:', score)
        print('Test accuracy:', acc)

    def show_performance_train(self,X_train,y_train):
        score, acc = self.model.evaluate(X_train, y_train)
        print('Train score:', score)
        print('Train accuracy:', acc)

    def inference(self,sample):
        return self.model.predict(sample)
        
