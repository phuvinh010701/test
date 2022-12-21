import tensorflow as tf

def create_model(shape):
    model = tf.keras.applications.EfficientNetV2B0(
    include_top=True,
    weights=None,
    input_shape=(shape, shape, 3),
    pooling=None,
    classes=9,
    classifier_activation="softmax",
    include_preprocessing=True,
    )

    model.summary()

    return model

