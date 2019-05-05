import numpy as np
import codecs

TRAIN_DATA = './simple-examples/data/train_data_index.txt'
TRAIN_BATCH_SIZE = 20
TRAIN_NUM_STEP = 35


def generate_trian_data(TRAIN_DATA=TRAIN_DATA, TRAIN_BATCH_SIZE=TRAIN_BATCH_SIZE, TRAIN_NUM_STEP=TRAIN_NUM_STEP):
    train_data_org = []
    with codecs.open(TRAIN_DATA, 'r', 'utf-8') as f:
        for line in f:
            train_data_org.extend([int(i) for i in line.split()])
    train_data = np.array(train_data_org)

    num_batches = len(train_data) // (TRAIN_BATCH_SIZE * TRAIN_NUM_STEP)
    train_data = train_data[: num_batches * TRAIN_NUM_STEP * TRAIN_BATCH_SIZE]
    train_data = np.reshape(train_data, [TRAIN_BATCH_SIZE, -1])

    train_data = np.split(train_data, num_batches, axis=1)

    label = np.array(train_data_org[1: num_batches * TRAIN_NUM_STEP * TRAIN_BATCH_SIZE+1])
    label = np.reshape(label, [TRAIN_BATCH_SIZE, -1])
    label = np.split(label, num_batches, axis=1)
    return list(zip(train_data, label))


def main():
    train_batches = generate_trian_data(TRAIN_DATA=TRAIN_DATA, TRAIN_BATCH_SIZE=TRAIN_BATCH_SIZE, TRAIN_NUM_STEP=TRAIN_NUM_STEP)
    print(train_batches)


if __name__ == '__main__':
    main()