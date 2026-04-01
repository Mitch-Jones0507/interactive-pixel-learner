import tensorflow as tf

def build_scipl_old(num_classes, input_shape, conv_layers, base_filters, dense_units, dropout, batch_norm):
    return tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D((2, 2)),

        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),

        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ],name='scipl')

def build_scipl(num_classes, input_shape,
                conv_layers, base_filters,
                dense_units, dropout, batch_norm):

    model = tf.keras.models.Sequential(name='scipl')
    for i in range(conv_layers):
        filters = base_filters * (2 ** i)
        if i == 0:
            model.add(tf.keras.layers.Conv2D(
                filters,
                (3, 3),
                activation='relu',
                padding='same',
                input_shape=input_shape
            ))
        else:
            model.add(tf.keras.layers.Conv2D(
                filters,
                (3, 3),
                activation='relu',
                padding='same'
            ))
        if batch_norm:
            model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.MaxPooling2D((2, 2)))
        if dropout > 0:
            model.add(tf.keras.layers.Dropout(dropout * 0.5))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(dense_units, activation='relu'))
    if dropout > 0:
        model.add(tf.keras.layers.Dropout(dropout))
    model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))

    return model