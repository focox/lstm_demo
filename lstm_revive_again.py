import tensorflow as tf
import config
from config import Config

config = Config(config.file('./config.cfg'))

TRAIN_DATA_PATH = config.getByPath('train_data_path')
TEST_DATA_PATH = config.getByPath('test_data_path')

HIDDEN_SIZE = config.getByPath('hidden_size')
NUM_LAYERS = config.getByPath('num_layers')
VOCAB_SIZE = config.getByPath('vocab_size')
TRAIN_BATCH_SIZE = config.getByPath('train_batch_size')
TRAIN_NUM_TIME_STEPS = config.getByPath('train_num_steps')

VALID_DATA_PATH = config.getByPath('valid_data_path')
VALID_BATCH_SIZE = config.getByPath('valid_batch_size')
VALID_NUM_TIME_STEPS = config.getByPath('valid_num_steps')

NUM_EPOCHS = config.getByPath('num_epochs')
SHARE_EMB_AND_SOTFMAX = config.getByPath('share_emb_and_softmax')
OUTPUT_KEEP_PROB = config.getByPath('lstm_keep_output_prob')
EMBEDDING_KEEP_PROB = config.getByPath('embedding_keep_prob')
MAX_CLIPPED_NORM = config.getByPath('max_gradient_norm')


class PTBModle:
    def forward(self, x, is_training):
        batch_size, num_time_steps = int(x.shape[0]), int(x.shape[1])
        with tf.variable_scope('embedding_layer'):
            embedding_weight = tf.get_variable('weight', [VOCAB_SIZE, HIDDEN_SIZE], tf.int32, tf.truncated_normal_initializer)

        inputs = tf.nn.embedding_lookup(embedding_weight, x)

        if is_training:
            inputs = tf.nn.dropout(inputs, EMBEDDING_KEEP_PROB)

        with tf.name_scope('lstm-structure'):
            output_keep_prob = OUTPUT_KEEP_PROB if is_training else 1.0
            cell = [tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE), output_keep_prob=output_keep_prob) for _ in range(NUM_LAYERS)]
            lstm_cell = tf.nn.rnn_cell.MultiRNNCell(cell)
            self.initial_state = lstm_cell.zero_state(batch_size, tf.float32)
            state = self.initial_state
            outputs = []
            for time_step in range(num_time_steps):
                tf.get_variable_scope().reuse_variables() if time_step > 0 else None
                output, state = lstm_cell(inputs[:, -1, :], state)
                outputs.append(outputs)
            self.final_state = state
            outputs = tf.reshape(tf.concat(outputs, axis=1), [-1, HIDDEN_SIZE])

        if SHARE_EMB_AND_SOTFMAX:
            softmax_weight = tf.transpose(embedding_weight)
        else:
            with tf.variable_scope('softmax'):
                softmax_weight = tf.get_variable('weight', [HIDDEN_SIZE, VOCAB_SIZE], tf.float32, tf.truncated_normal_initializer)
        bias = tf.get_variable('softmax_bias', [VOCAB_SIZE], tf.float32, tf.constant_initializer)
        self.logits = tf.matmul(outputs, softmax_weight) + bias
        return self.logits, batch_size, num_time_steps

    def backward(self, x, y):
        with tf.name_scope('losses'):
            logits, batch_size, num_time_steps = self.forward(x, is_training=True)
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=tf.reshape(y, [-1]))
            self.cost = tf.reduce_sum(loss) / batch_size

        trainable_variables = tf.trainable_variables()

        with tf.name_scope('compute_gradient'):
            gradient = tf.gradients(self.cost, trainable_variables)
            clipped_grad, _ = tf.clip_by_global_norm(gradient, MAX_CLIPPED_NORM)

            optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0)
            self.train_op = optimizer.apply_gradients(zip(clipped_grad, trainable_variables))
