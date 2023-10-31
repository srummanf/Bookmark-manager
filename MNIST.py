import tensorflow as tf
from tensorflow.keras import Sequential

from tensorflow.keras.layers import Dense

model = Sequential([
    Dense(units=25, activation='relu'),
    Dense(units=15, activation='relu'),
    Dense(units=10, activation='linear')
])


from tensorflow.keras.losses import SparseCategoricalCrossentropy
model.compile(..., loss=SparseCategoricalCrossentropy(from_logits=True))

model.fit(X,Y,epochs=100)

logits = model(X,Y)
f_x = tf.nn.softmax(logits)
f_x = tf.nn.sigmoid(logits)