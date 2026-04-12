import tensorflow as tf

def build_chipl(
    num_classes,
    input_shape=(28, 28, 1),
    conv_layers=2,
    base_filters=8,
    dense_units=32,
    kernel_size=3,
    dropout=0.3,
    batch_norm="Yes",
    weight_decay=0.0001
):

    l2 = tf.keras.regularizers.l2(weight_decay)

    model = tf.keras.Sequential(name='chipl')

    for i in range(conv_layers):
        filters = base_filters * (2 ** i)

        if i == 0:
            model.add(tf.keras.layers.Conv2D(
                filters,
                (kernel_size, kernel_size),
                activation=None,
                padding='same',
                kernel_regularizer=l2,
                input_shape=input_shape
            ))
        else:
            model.add(tf.keras.layers.Conv2D(
                filters,
                (kernel_size, kernel_size),
                activation=None,
                padding='same',
                kernel_regularizer=l2
            ))

        if batch_norm == "Yes":
            model.add(tf.keras.layers.BatchNormalization())

        model.add(tf.keras.layers.ReLU())
        model.add(tf.keras.layers.MaxPooling2D((2, 2)))

    model.add(tf.keras.layers.Flatten())

    model.add(tf.keras.layers.Dense(
        dense_units,
        activation='relu',
        kernel_regularizer=l2
    ))

    if dropout > 0:
        model.add(tf.keras.layers.Dropout(dropout))

    model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))

    return model