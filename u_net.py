import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D,MaxPool2D,Dropout,UpSampling2D,Concatenate
from tensorflow.keras import Model

class Unet_Model(Model):
    def __init__(self, u_layers=3,min_feature=16,increment=2,shape=(3,3),dropout=0.1):
        super(Unet_Model, self).__init__()
            
        self.layer=u_layers
        self.convs1=[]
        self.convs2=[]
        self.convs3=[]
        self.convs4=[]
        self.pool_up=[]
        self.drop=[]
        feature=min_feature
        
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.MeanSquaredError(name='train_accuracy')
        self.train_IoU_accuracy = tf.keras.metrics.MeanIoU(num_classes=2,name='IoU')

        self.test_loss = tf.keras.metrics.Mean(name='test_loss')
        self.test_accuracy = tf.keras.metrics.MeanSquaredError(name='test_accuracy')
        self.test_IoU_accuracy = tf.keras.metrics.MeanIoU(num_classes=2,name='IoU')

        self.loss_object = tf.keras.losses.LogCosh()
        self.optimizer = tf.keras.optimizers.Adam()
        for i in range(self.layer):
            self.pool_up.append(MaxPool2D((2, 2), padding='same'))
            self.convs1.append(Conv2D(feature, shape, padding='same', activation='relu'))
            self.drop.append(Dropout(dropout))
            self.convs2.append(Conv2D(feature, shape, padding='same', activation='relu'))
            feature*=increment
            
        for i in range(self.layer):
            feature//=increment

            self.convs3.append(Conv2D(feature, shape, padding='same', activation='relu'))
            self.drop.append(Dropout(dropout))
            self.convs4.append(Conv2D(feature, shape, padding='same', activation='relu'))
            self.pool_up.append(UpSampling2D())
            
        self.out = Conv2D(1, (3, 3), padding='same', activation='sigmoid')

    def call(self, x):
        
        keep=[]
         
        for i in range(self.layer):
            if(i>0):
                x=self.pool_up[i](x)
                x=self.convs1[i](x)
                x=self.drop[i](x)
                x=self.convs2[i](x)
                keep.append(x)
            else:
                x=self.convs1[i](x)
                x=self.convs2[i](x)
                keep.append(x)
           
        for i in range(self.layer):
            if(i>0):
                x=self.pool_up[self.layer+i](x)
                x=Concatenate(axis=3)([keep[self.layer-i-1], x])
                x=self.convs3[i](x)
                x=self.drop[self.layer+i](x)
                x=self.convs4[i](x)
            else:

                x=self.convs3[i](x)
                x=self.convs4[i](x)                

                
        x = self.out(x)
 

        return x

    @tf.function
    def train_step(self,images, labels):
        with tf.GradientTape() as tape:
            # training=True is only needed if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            predictions = self(images, training=True)

            loss = self.loss_object(labels, predictions)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        predictions_bin=tf.round(predictions)
        self.train_loss(loss)
        self.train_accuracy(labels, predictions)
        self.train_IoU_accuracy(labels, predictions_bin)
        
    @tf.function
    def test_step(self,images, labels):
        # training=False is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = self(images, training=False)
        predictions=tf.keras.backend.round(predictions)
        t_loss = self.loss_object(labels, predictions)

        self.test_loss(t_loss)
        self.test_accuracy(labels, predictions)
        self.test_IoU_accuracy(labels, predictions)

        
    def train(self,X_np_5, y_np,EPOCHS=10):
        self.train_ds = tf.data.Dataset.from_tensor_slices((X_np_5, y_np)).shuffle(10000).batch(32)

        self.test_ds = tf.data.Dataset.from_tensor_slices((X_np_5, y_np)).batch(32)
        
        for epoch in range(EPOCHS):
            # Reset the metrics at the start of the next epoch
            self.train_loss.reset_states()
            self.train_accuracy.reset_states()
            self.train_IoU_accuracy.reset_states()
            self.test_loss.reset_states()
            self.test_accuracy.reset_states()
            self.test_IoU_accuracy.reset_states()


            for images, labels in self.train_ds:
                self.train_step(images, labels)

            for test_images, test_labels in self.test_ds:
                self.test_step(test_images, test_labels)

            print(
            f'Epoch {epoch + 1}, '
            f'Loss: {self.train_loss.result():.2f}, '
            f'MSE: {self.train_accuracy.result() * 100:.2f}, '
            f'Accuracy IoU: {self.train_IoU_accuracy.result() * 100:.2f}, '
            f'Test Loss: {self.test_loss.result():.2f}, '
            f'Test MSE: {self.test_accuracy.result() * 100:.2f}'
            f'Test IoU Accuracy: {self.test_IoU_accuracy.result() * 100:.2f}'
            )