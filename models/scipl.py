import tensorflow as tf

def build_scipl(num_classes):

    l2 = tf.keras.regularizers.l2(0.0001)
    model = tf.keras.Sequential(name="scipl")

    model.add(tf.keras.layers.Conv2D(
        8, (3, 3),
        activation=None,
        padding="same",
        kernel_regularizer=l2,
        input_shape=(28, 28, 1)
    ))

    model.add(tf.keras.layers.ReLU())
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))

    model.add(tf.keras.layers.Conv2D(
        16, (3, 3),
        activation=None,
        padding="same",
        kernel_regularizer=l2
    ))

    model.add(tf.keras.layers.ReLU())
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))

    model.add(tf.keras.layers.Flatten())

    model.add(tf.keras.layers.Dense(
        32,
        activation="relu",
        kernel_regularizer=l2
    ))

    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Dense(
        num_classes,
        activation="softmax"
    ))

    return model