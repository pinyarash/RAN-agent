import os
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense

class CriticNetwork(keras.Model):
    def __init__(self, fc1_dims=512, fc2_dims=512,
            name='critic', chkpt_dir='tmp/ddpg'):
        super(CriticNetwork, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims

        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, 
                    self.model_name+'_ddpg.h5')

        self.fc1 = Dense(self.fc1_dims, activation='relu')
        self.fc2 = Dense(self.fc2_dims, activation='relu')
        self.q = Dense(1, activation=None)

    def call(self, state, action):
        action_value = self.fc1(tf.concat([state, action], axis=1))
        action_value = self.fc2(action_value)

        q = self.q(action_value)

        return q

class ActorNetwork(keras.Model):
    def __init__(self, fc1_dims=512, fc2_dims=512, n_actions=2, name='actor',
            chkpt_dir='tmp/ddpg'):
        super(ActorNetwork, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions

        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, 
                    self.model_name+'_ddpg.h5')

        self.fc1 = Dense(self.fc1_dims, activation='relu')
        self.fc2 = Dense(self.fc2_dims, activation='relu')
        self.mu = Dense(self.n_actions, activation='tanh')

    def call(self, state):
        prob = self.fc1(state)
        prob = self.fc2(prob)

        mu = self.mu(prob)

        return mu


class RepNetwork(keras.Model):
    def __init__(self, fc1_dims=512, fc2_dims=512,
            name='lstm', chkpt_dir='tmp/ddpg'):
        super(RepNetwork, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims

        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, 
                    self.model_name+'_ddpg.h5')

        self.fc1 = Dense(self.fc1_dims, activation='relu')
        self.fc2 = Dense(self.fc2_dims, activation='relu')
        self.q = Dense(1, activation=None)

    def call(self, state_size, hidden_size):
        print("Now we build the representation model")
        S = tf.keras.Input(shape=(None, state_size))
        lstm1 = tf.keras.LSTM(hidden_size, return_sequences=False)(S)
        model = tf.keras.Model(inputs=S, outputs=lstm1)
        return model, model.trainable_weights, S


class RepNetwork(keras.Model):
    def __init__(self, sess, state_size, hidden_size, BATCH_SIZE, TAU, LEARNING_RATE):
        self.sess = sess
        self.TAU = TAU
        self.LEARNING_RATE = LEARNING_RATE
        self.hidden_size = hidden_size
        K.set_session(sess)

        # Now create the model
        self.model, self.weights, self.state = self.create_rep_network(state_size,hidden_size)
        self.target_model, self.target_weights, self.target_state = self.create_rep_network(state_size,hidden_size)
        self.state_gradient = tf.placeholder(tf.float32, [None, hidden_size])
        self.unnormalized_gradients = tf.gradients(self.model.output, self.weights, -self.state_gradient)
        self.params_grad = list(
            map(lambda x: tf.math.divide(x, BATCH_SIZE), self.unnormalized_gradients))
        grads = zip(self.params_grad, self.weights)
        self.optimize = tf.train.AdamOptimizer(LEARNING_RATE).apply_gradients(grads)
        self.sess.run(tf.initialize_all_variables())

    def train(self, states, state_grads):
        self.sess.run(self.optimize, feed_dict={
            self.state: states,
            self.state_gradient: state_grads
        })

    def target_train(self):
        rep_weights = self.model.get_weights()
        rep_target_weights = self.target_model.get_weights()
        for i in range(len(rep_weights)):
            rep_target_weights[i] = self.TAU * rep_weights[i] + (1 - self.TAU) * rep_target_weights[i]
        self.target_model.set_weights(rep_target_weights)

    def create_rep_network(self, state_size, hidden_size):
        print("Now we build the representation model")
        S = Input(shape=(None, state_size))
        lstm1 = LSTM(hidden_size, return_sequences=False)(S)
        model = Model(inputs=S, outputs=lstm1)
        return model, model.trainable_weights, S

