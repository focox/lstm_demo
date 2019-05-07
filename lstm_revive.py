import tensorflow as tf
import numpy as np
import process_batch
import config
from config import Config

config = Config(config.file('./config.cfg'))

TRAIN_DATA_PATH = config.getByPath('train_data_path')
TEST_DATA_PATH = config.getByPath('test_data_path')
VALID_DATA_PATH = config.getByPath('valid_data_path')
HIDDEN_SIZE = config.getByPath('hidden_size')

NUM_LAYERS = config.getByPath('num_layers')
VOCAB_SIZE = config.getByPath('vocab_size')
TRAIN_BATCH_SIZE = config.getByPath('train_batch_size')
TRAIN_NUM_STEPS = config.getByPath('train_num_steps')

VALID_BATCH_SIZE = config.getByPath('valid_batch_size')
VALID_NUM_STEPS = config.getByPath('valid_num_steps')
NUM_EPOCHS = config.getByPath('num_epochs')
LSTM_KEEP_OUTPUT_PROB = config.getByPath('lstm_keep_output_prob')
EMBEDDING_KEEP_PROB = config.getByPath('embedding_keep_prob')
MAX_GRAD_NORM = config.getByPath('max_gradient_norm')
SHARE_EMB_AND_SOFTMAX = config.getByPath('share_emb_and_softmax')


class PTBModel:
    def __init__(self, batch_size, num_steps):
        self.x = tf.placeholder(tf.int32, [batch_size, num_steps])
        self.y = tf.placeholder(tf.int32, [batch_size, num_steps])
        self.backward(self.x, self.y, batch_size=batch_size, num_steps=num_steps)

    def forward(self, x, is_training, batch_size, num_steps):
        output_keep_prob = LSTM_KEEP_OUTPUT_PROB if is_training else 1.0
        basic_cell = [tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE), output_keep_prob=output_keep_prob) for _ in range(NUM_LAYERS)]
        lstm_cell = tf.nn.rnn_cell.MultiRNNCell(basic_cell)
        self._initial_state = lstm_cell.zero_state(batch_size, tf.float32)

        with tf.variable_scope('embedding_layer'):
            embedding_weight = tf.get_variable('embedding', [VOCAB_SIZE, HIDDEN_SIZE], tf.float32, tf.truncated_normal_initializer)
            inputs = tf.nn.embedding_lookup(embedding_weight, x, name='embedding_inputs')
        if is_training:
            with tf.variable_scope('embedding_dropout'):
                inputs = tf.nn.dropout(inputs, EMBEDDING_KEEP_PROB, name='embedding_dropout')

        output = []
        state = self._initial_state
        with tf.variable_scope('LSTM'):
            for time_step in range(num_steps):
                if time_step > 0: tf.get_variable_scope().reuse_variables()
                cell_output, state = lstm_cell(inputs[:, time_step, :], state)
                output.append(cell_output)
            self.final_state = state

        # Todo: concat every num_steps, and then reshape into [-1, HIDDEN_SIZE]
        output = tf.reshape(tf.concat(output, axis=1), [-1, HIDDEN_SIZE])

        with tf.variable_scope('softmax'):
            if SHARE_EMB_AND_SOFTMAX:
                weight = tf.transpose(embedding_weight)
            else:
                weight = tf.get_variable('softmax', [HIDDEN_SIZE, VOCAB_SIZE], tf.float32, tf.truncated_normal_initializer)
            bias = tf.get_variable('bias', [VOCAB_SIZE], tf.float32, tf.constant_initializer(0.01))

            output = tf.matmul(output, weight) + bias
        return output

    def backward(self, x, y, batch_size=TRAIN_BATCH_SIZE, num_steps=TRAIN_NUM_STEPS):
        logits = self.forward(x, True, batch_size, num_steps)

        # Todo: notify the shapes of y and losses
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=tf.reshape(y, [-1]))
        self.loss = tf.reduce_sum(losses) / batch_size

        trainable_variables = tf.trainable_variables()
        gradients = tf.gradients(self.loss, trainable_variables)
        cliped_gradients, _ = tf.clip_by_global_norm(gradients, MAX_GRAD_NORM)

        optimizer = tf.train.AdamOptimizer()
        # optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0)
        self.train_op = optimizer.apply_gradients(zip(cliped_gradients, trainable_variables))

    def train(self):
        train_batch = process_batch.make_batches(process_batch.read_data(TRAIN_DATA_PATH), TRAIN_BATCH_SIZE, TRAIN_NUM_STEPS)
        valid_batch = process_batch.make_batches(process_batch.read_data(VALID_DATA_PATH), VALID_BATCH_SIZE, VALID_NUM_STEPS)


        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            state = sess.run(self._initial_state)
            step = 0
            iters = 0
            total_costs = 0.0
            for epoch in range(NUM_EPOCHS):
                print('Epoch', epoch + 1)
                for x, y in train_batch:
                    _, loss, state = sess.run([self.train_op, self.loss, self.final_state], feed_dict={self.x: x, self.y: y, self._initial_state: state})
                    total_costs += loss
                    iters += TRAIN_NUM_STEPS
                    # print(loss)
                    if step % 100 == 0:
                        print('After %d step(s), the loss on train batch is %g' % (step + 1, np.exp(total_costs/iters)))
                    step += 1



PTBModel = PTBModel(TRAIN_BATCH_SIZE, TRAIN_NUM_STEPS)


def main(argv=None):
    PTBModel.train()


if __name__ == '__main__':
    tf.app.run()
