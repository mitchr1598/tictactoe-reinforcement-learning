import tensorflow as tf
import numpy as np


class Network:
    def __init__(self):
        self.model = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=(3, 6)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(9)
        ])
        self.model.compile(optimizer='adam',
                           loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                           metrics=['accuracy'])
        self.move_maker = tf.keras.Sequential([self.model,
                                                tf.keras.layers.Softmax()])

    def choose_move(self, p1_board: np.array, p2_board: np.array, valid_moves: set) -> (int, np.array):
        merged = np.concatenate((p1_board, p2_board), axis=1)
        naive_predictions = self.move_maker.predict(np.array([merged]))  # Not considering valid moves
        predictions = naive_predictions * valid_moves
        predictions = predictions / predictions.sum()  # Removes invalid moves and renormalizes
        return np.argmax(predictions), predictions  # A flattend array of the best moves in order


def main():
    p1 = np.array([[False, False, False],
                   [False, False, False],
                   [True, False, False]])

    p2 = np.array([[False, False, False],
                   [False, False, False],
                   [False, False, False]])
    net = Network()
    move = net.choose_move(p1, p2)
    print(move)


if __name__ == '__main__':
    main()

