import tensorflow as tf
import numpy as np

ENCODER_HL1_NODES = 512
ENCODER_OUT_NODES = 512
DECODER_HL1_NODES = 512
DECODER_OUT_NODES = 512

class VAE():
    def __init__(self, beta=1, latent_dim=2, batch_size=100, learning_rate=1e-4):
        """
        Instantiates the VAE, builds placeholder for the data, the network then runs an InteractiveSession.
        :param beta: Coefficient term to weigh how important the KL term is in the loss function.
        :param latent_dim: Latent space dimension.
        :param batch_size: Training batch size.
        :param learning_rate: For the training optimizer.
        """
        self.beta = beta
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.x = tf.placeholder('float', [None, 784], name='x')
        self._build_network()
        self._vae_loss_and_optimizer()
        
        init = tf.global_variables_initializer()
        self.sess = tf.InteractiveSession()
        self.sess.run(init)
        
    def _build_network(self):
        """
        Builds graph for a fully connected neural network.
        """
        self._initialize_weights_fc()
        self._encoder_fc(self.x)
        self._sample_z()
        self._decoder_fc()

    def _initialize_weights_fc(self):
        """
        Uses Xavier initialization to initialize neural network weights.
        """
        initializer = tf.contrib.layers.xavier_initializer()
        
        self.encoder_input = {'weights':tf.Variable(initializer([784, ENCODER_HL1_NODES]), name='encoder_input_w'),
                          'biases':tf.Variable(initializer([ENCODER_HL1_NODES]), name='encoder_input_b')}

        self.encoder_output_mu = {'weights':tf.Variable(initializer([ENCODER_OUT_NODES, self.latent_dim]), \
                                                        name='encoder_out_mu_w'),
                        'biases':tf.Variable(initializer([self.latent_dim]), name='encoder_out_mu_b')}

        self.encoder_output_log_sigma = {'weights':tf.Variable(initializer([ENCODER_OUT_NODES, self.latent_dim]), \
                                                               name='encoder_out_log_sigma_w'),
                        'biases':tf.Variable(initializer([self.latent_dim]), name='encoder_out_log_sigma_b')}
        
        self.decoder_input = {'weights':tf.Variable(initializer([self.latent_dim, DECODER_HL1_NODES]), \
                                                    name='decoder_input_w'),
                          'biases':tf.Variable(initializer([ENCODER_HL1_NODES]), name='decoder_input_b')}

        self.decoder_output = {'weights':tf.Variable(initializer([DECODER_OUT_NODES, 784]), name='decoder_output_w'), \
                        'biases':tf.Variable(initializer([784]), name='decoder_output_b')}
        
    def _encoder_fc(self, data):
        """
        Builds fully connected neural network encoder with one hidden layer.
        :param data: Placeholder for mini-batch of flattened digit data.
        :return: Parameters of Gaussian distribution of z.
        """
        l1 = tf.add(tf.matmul(data,self.encoder_input['weights']), self.encoder_input['biases'])
        l1 = tf.nn.tanh(l1)

        self.z_mu = tf.add(tf.matmul(l1,self.encoder_output_mu['weights']), self.encoder_output_mu['biases'])
        self.z_log_sigma = tf.add(tf.matmul(l1,self.encoder_output_log_sigma['weights']), self.encoder_output_log_sigma['biases'])
        
    def _decoder_fc(self):
        """
        Builds fully connected neural network decoder with one hidden layer.
        :return: Reconstructed flattened digit from sampled z.
        """
        l1 = tf.add(tf.matmul(self.z,self.decoder_input['weights']), self.decoder_input['biases'])
        l1 = tf.nn.tanh(l1)

        out = tf.add(tf.matmul(l1,self.decoder_output['weights']), self.decoder_output['biases'])
        self.x_reconstruction = tf.nn.sigmoid(out)

    def _sample_z(self):
        """
        Sample z for decoder. Reparameterized trick used to allow backpropagation during training.
        :return: Sampled z with shape [batch_size, latent_dim].
        """
        epsilon = tf.random_normal(shape=[self.batch_size, self.latent_dim],mean=0.0, stddev=1.0, name='epsilon', dtype='float32')
        self.z = tf.add(self.z_mu,tf.multiply(epsilon, tf.exp(0.5 * self.z_log_sigma)), name='z')
        
    def _vae_loss_and_optimizer(self):
        """
        Loss function for training. Binary cross entropy used as binarized MNIST data is used.
        :return: Reconstruction, KL loss, total loss and optimizer.
        """
        self.recon = tf.reduce_mean(tf.reduce_sum(tf.keras.backend.binary_crossentropy(self.x, self.x_reconstruction)))
        # KL divergence between q(z|x) and p(z)
        self.kl = tf.reduce_mean(0.5 * tf.reduce_sum(tf.exp(self.z_log_sigma) + tf.square(self.z_mu) - 1 - self.z_log_sigma, 1))

        self.loss = self.recon + self.beta*self.kl
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

    def fit(self, x_batch):
        """
        Used for mini-batch gradient descent.
        :param x_batch: Mini-batch of data for training.
        :return: Optimizer, total loss, reconstruction loss and KL loss.
        """
        return self.sess.run([self.optimizer, self.loss, self.recon, self.kl], feed_dict={self.x: x_batch})
                
    def reconstruct_X(self, x_true):
        """
        Takes in true digit, randomly samples z from laten space distribution then reconstructs the digit.
        :param x_true: flattened mini-batch of binarized MNIST image.
        :return:
        """
        return self.sess.run(self.x_reconstruction, feed_dict={self.x: x_true})
    
    def generate(self, z_mu=None):
        """
        Reconstructs x from specified z in the latent space.
        :param z_mu: Mini-batch of points in the latent space.
        :return: Reconstructed digit.
        """
        if z_mu is None:
            z_mu = np.random.normal(size=[self.batch_size, self.latent_dim])
        return self.sess.run(self.x_reconstruction,
                             feed_dict={self.z: z_mu})
    
    def get_z_mean(self, x_test):
        """
        Finds mean of latent space for given digit.
        :param x_test: Mini-batch of flatten binarized digit.
        :return: Mean of latent space distribution.
        """
        return self.sess.run(self.z_mu, feed_dict={self.x: x_test})