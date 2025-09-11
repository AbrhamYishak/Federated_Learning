
import numpy as np
import tensorflow as tf
import random

def build_model():
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Reshape((28, 28, 1), input_shape=(28*28,)),
            tf.keras.layers.Conv2D(32, 3, activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, 3, activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(10, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def load_mnist_flat():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape((-1, 28*28)).astype("float32") / 255.0
    x_test = x_test.reshape((-1, 28*28)).astype("float32") / 255.0
    return (x_train, y_train), (x_test, y_test)


def create_iid_partitions(x, y, num_clients):
    n = len(x)
    idxs = np.random.permutation(n)
    split = np.array_split(idxs, num_clients)
    return [(x[s], y[s]) for s in split]


def create_noniid_partitions(x, y, num_clients, shards_per_client=2):
    num_shards = num_clients * shards_per_client
    shard_size = len(x) // num_shards

    idxs = np.argsort(y)
    x_sorted, y_sorted = x[idxs], y[idxs]

    shards = [(x_sorted[i*shard_size:(i+1)*shard_size],
               y_sorted[i*shard_size:(i+1)*shard_size])
              for i in range(num_shards)]

    random.shuffle(shards)
    client_partitions = []
    for i in range(num_clients):
        client_shards = shards[i*shards_per_client:(i+1)*shards_per_client]
        xs = np.vstack([s[0] for s in client_shards])
        ys = np.hstack([s[1] for s in client_shards])
        client_partitions.append((xs, ys))
    return client_partitions

def fedavg_train(num_clients=10, frac_clients=0.5, rounds=5, local_epochs=1,
                batch_size=32, non_iid=False):
    (x_train, y_train), (x_test, y_test) = load_mnist_flat()

    if non_iid:
        partitions = create_noniid_partitions(x_train, y_train, num_clients)
    else:
        partitions = create_iid_partitions(x_train, y_train, num_clients)

    # Global model
    global_model = build_model()

    # History
    round_acc = []

    for r in range(1, rounds+1):
        print(f"\n--- Round {r} ---")

        # Select fraction of clients
        m = max(1, int(frac_clients * num_clients))
        selected = random.sample(range(num_clients), m)

        client_weights = []
        client_sizes = []

        # Train each selected client
        for cid in selected:
            model = build_model()
            model.set_weights(global_model.get_weights())
            x_c, y_c = partitions[cid]
            model.fit(x_c, y_c,
                      batch_size=batch_size,
                      epochs=local_epochs,
                      verbose=0)
            client_weights.append(model.get_weights())
            client_sizes.append(len(x_c))

        # Aggregate updates (weighted average)
        new_weights = []
        total_size = sum(client_sizes)
        for weights_tuple in zip(*client_weights):
            avg_w = np.sum(
                [w * (n/total_size) for w, n in zip(weights_tuple, client_sizes)],
                axis=0
            )
            new_weights.append(avg_w)
        global_model.set_weights(new_weights)

        # Evaluate global model
        loss, acc = global_model.evaluate(x_test, y_test, verbose=0)
        print(f"Round {r} Test accuracy: {acc:.4f}")
        round_acc.append(acc)

    return global_model, round_acc
global_model, acc_history = fedavg_train(
    num_clients=10,
    frac_clients=0.5,
    rounds=5,
    local_epochs=1,
    batch_size=32,
    non_iid=True   # set True for non-IID
)
