import tensorflow.keras as keras
from tensorflow.keras.layers import Dense


class ActorNetwork(keras.Model):
    def __init__(self, n_actions, fc1_dims=128, fc2_dims=128,fc3_dims=64):
        super(ActorNetwork, self).__init__()

        self.fc1 = Dense(fc1_dims, activation='relu')
        self.fc2 = Dense(fc2_dims, activation='relu')
        self.fc3 = Dense(fc3_dims, activation='relu')

        # self.fc3 = Dense(n_actions, activation='softmax')
        self.fc4 = Dense(n_actions, activation='sigmoid')

    def call(self, state):
        x = self.fc1(state)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        # x = x / tf.reduce_sum(x, axis=-1)
        
        return x


class CriticNetwork(keras.Model):
    def __init__(self, fc1_dims=128, fc2_dims=64, fc3_dims=64):
        super(CriticNetwork, self).__init__()
        self.fc1 = Dense(fc1_dims, activation='relu')
        self.fc2 = Dense(fc2_dims, activation='relu')
        self.fc3 = Dense(fc3_dims, activation='relu')

        self.q = Dense(1, activation=None)

    def call(self, state):
        x = self.fc1(state)
        x = self.fc2(x)
        x = self.fc3(x)
        q = self.q(x)

        return q