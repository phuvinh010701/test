from model import *
import argparse
from data import *


def parser():
    parser = argparse.ArgumentParser(description='multi task')
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--epochs", default=200, type=int)
    parser.add_argument("--shape", default=96, type=int)
    return parser.parse_args()

def main():
    args = parser()
    model = create_model(args.shape)

    early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    checkpoint = tf.keras.callbacks.ModelCheckpoint('mobilenetv3small{epoch:02d}.h5', monitor = "val_loss",save_best_only = True)
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'],
    )
    train_data = DataGenerator('original_images/', 'new_train.txt', dim=(args.shape, args.shape, 3))
    valid_data = DataGenerator('original_images/', 'new_val.txt', dim=(args.shape, args.shape, 3), train=False)
    history = model.fit(train_data, validation_data=valid_data, epochs=args.epochs, callbacks=[early, checkpoint], workers=2)
    model.save('final.h5')

if __name__ == "__main__":
    main()