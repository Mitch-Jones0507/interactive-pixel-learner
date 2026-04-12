import tensorflow as tf

class PatchEmbedding(tf.keras.layers.Layer):
    def __init__(self, patch_size, embed_dim):
        super().__init__()
        self.patch_size = patch_size
        self.projection = tf.keras.layers.Dense(embed_dim)

    def call(self, images):
        batch_size = tf.shape(images)[0]

        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )

        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])

        return self.projection(patches)


class ViTBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.att = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(ff_dim, activation="relu"),
            tf.keras.layers.Dense(embed_dim),
        ])
        self.norm1 = tf.keras.layers.LayerNormalization()
        self.norm2 = tf.keras.layers.LayerNormalization()
        self.dropout = tf.keras.layers.Dropout(dropout)

    def call(self, x, training=False):
        attn = self.att(x, x)
        x = self.norm1(x + self.dropout(attn, training=training))

        ffn = self.ffn(x)
        return self.norm2(x + self.dropout(ffn, training=training))

def build_tripl(num_classes, input_shape, num_layers,
                embed_dim, num_heads, ff_dim, patch_size,
                dropout_rate):

    inputs = tf.keras.Input(shape=input_shape)

    x = PatchEmbedding(patch_size, embed_dim)(inputs)

    num_patches = (input_shape[0] // patch_size) ** 2
    positions = tf.range(start=0, limit=num_patches, delta=1)
    pos_embed = tf.keras.layers.Embedding(
        input_dim=num_patches, output_dim=embed_dim
    )(positions)

    x = x + pos_embed

    for layer in range(num_layers):
        x = ViTBlock(embed_dim, num_heads, ff_dim, dropout)(x)

    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)

    if dropout > 0:
        x = tf.keras.layers.Dropout(dropout)(x)

    x = tf.keras.layers.Dense(ff_dim, activation="relu")(x)

    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    return tf.keras.Model(inputs, outputs, name="vit")

