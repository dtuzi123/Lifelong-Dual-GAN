import time
from utils import *
from scipy.misc import imsave as ims
from ops import *
from utils import *
from Utlis2 import *
import random as random
from glob import glob
import os, gzip
from glob import glob
from Basic_structure import *
from mnist_hand import *
from CIFAR10 import *

beta = 1

os.environ['CUDA_VISIBLE_DEVICES']='4'

def file_name(file_dir):
    t1 = []
    file_dir = "F:/Third_Experiment/Multiple_GAN_codes/data/images_background/"
    for root, dirs, files in os.walk(file_dir):
        for a1 in dirs:
            b1 = "F:/Third_Experiment/Multiple_GAN_codes/data/images_background/" + a1 + "/renders/*.png"
            b1 = "F:/Third_Experiment/Multiple_GAN_codes/data/images_background/" + a1
            for root2, dirs2, files2 in os.walk(b1):
                for c1 in dirs2:
                    b2 = b1 + "/" + c1 + "/*.png"
                    img_path = glob(b2)
                    t1.append(img_path)
    cc = []

    for i in range(len(t1)):
        a1 = t1[i]
        for p1 in a1:
            cc.append(p1)
    return cc

def Image_classifier(inputs, scopename, is_training=True, reuse=False):
    with tf.variable_scope(scopename, reuse=reuse):
        batch_norm_params = {'is_training': is_training, 'decay': 0.9, 'updates_collections': None}
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params):
            x = tf.reshape(inputs, [-1, 32, 32, 3])

            # For slim.conv2d, default argument values are like
            # normalizer_fn = None, normalizer_params = None, <== slim.arg_scope changes these arguments
            # padding='SAME', activation_fn=nn.relu,
            # weights_initializer = initializers.xavier_initializer(),
            # biases_initializer = init_ops.zeros_initializer,
            net = slim.conv2d(x, 32, [5, 5], scope='conv1')
            net = slim.max_pool2d(net, [2, 2], scope='pool1')
            net = slim.conv2d(net, 64, [5, 5], scope='conv2')
            net = slim.max_pool2d(net, [2, 2], scope='pool2')
            net = slim.flatten(net, scope='flatten3')

            # For slim.fully_connected, default argument values are like
            # activation_fn = nn.relu,
            # normalizer_fn = None, normalizer_params = None, <== slim.arg_scope changes these arguments
            # weights_initializer = initializers.xavier_initializer(),
            # biases_initializer = init_ops.zeros_initializer,
            net = slim.fully_connected(net, 1024, scope='fc3')
            net = slim.dropout(net, is_training=is_training, scope='dropout3')  # 0.5 by default
            outputs = slim.fully_connected(net, 4, activation_fn=None, normalizer_fn=None, scope='fco')
    return outputs

def CodeImage_classifier(s, scopename, reuse=False):
    with tf.variable_scope(scopename, reuse=reuse):
        input = s

        # initializers
        w_init = tf.contrib.layers.variance_scaling_initializer()
        b_init = tf.constant_initializer(0.)
        n_hidden = 500
        keep_prob = 0.9

        # 1st hidden layer
        w0 = tf.get_variable('w0', [input.get_shape()[1], n_hidden], initializer=w_init)
        b0 = tf.get_variable('b0', [n_hidden], initializer=b_init)
        h0 = tf.matmul(s, w0) + b0
        h0 = tf.nn.tanh(h0)
        h0 = tf.nn.dropout(h0, keep_prob)

        n_output = 4
        # output layer-mean
        wo = tf.get_variable('wo', [h0.get_shape()[1], n_output], initializer=w_init)
        bo = tf.get_variable('bo', [n_output], initializer=b_init)
        y1 = tf.matmul(h0, wo) + bo
        y = tf.nn.softmax(y1)

    return y1, y

def sample_gumbel(shape, eps=1e-20):
    """Sample from Gumbel(0, 1)"""
    U = tf.random_uniform(shape, minval=0, maxval=1)
    return -tf.log(-tf.log(U + eps) + eps)

def my_gumbel_softmax_sample(logits, cats_range, temperature=0.1):
    """ Draw a sample from the Gumbel-Softmax distribution"""
    y = logits + sample_gumbel(tf.shape(logits))
    logits_with_noise = tf.nn.softmax(y / temperature)
    return logits_with_noise


def load_mnist(dataset_name):
    data_dir = os.path.join("./data", dataset_name)

    def extract_data(filename, num_data, head_size, data_size):
        with gzip.open(filename) as bytestream:
            bytestream.read(head_size)
            buf = bytestream.read(data_size * num_data)
            data = np.frombuffer(buf, dtype=np.uint8).astype(np.float)
        return data

    data = extract_data(data_dir + '/train-images-idx3-ubyte.gz', 60000, 16, 28 * 28)
    trX = data.reshape((60000, 28, 28, 1))

    data = extract_data(data_dir + '/train-labels-idx1-ubyte.gz', 60000, 8, 1)
    trY = data.reshape((60000))

    data = extract_data(data_dir + '/t10k-images-idx3-ubyte.gz', 10000, 16, 28 * 28)
    teX = data.reshape((10000, 28, 28, 1))

    data = extract_data(data_dir + '/t10k-labels-idx1-ubyte.gz', 10000, 8, 1)
    teY = data.reshape((10000))

    trY = np.asarray(trY)
    teY = np.asarray(teY)

    X = np.concatenate((trX, teX), axis=0)
    y = np.concatenate((trY, teY), axis=0).astype(np.int)

    seed = 547
    np.random.seed(seed)
    np.random.shuffle(X)
    np.random.seed(seed)
    np.random.shuffle(y)

    y_vec = np.zeros((len(y), 10), dtype=np.float)
    for i, label in enumerate(y):
        y_vec[i, y[i]] = 1.0

    return X / 255., y_vec

def My_Encoder_mnist(image, z_dim, name, batch_size=64, reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()
        len_discrete_code = 4

        is_training = True
        x = image
        net = lrelu(conv2d(x, 64, 4, 4, 2, 2, name='c_conv1'))
        net = lrelu(bn(conv2d(net, 128, 4, 4, 2, 2, name='c_conv2'), is_training=is_training, scope='c_bn2'))
        net = tf.reshape(net, [batch_size, -1])
        net = lrelu(bn(linear(net, 1024, scope='c_fc3'), is_training=is_training, scope='c_bn3'))

        net = lrelu(bn(linear(net, 64, scope='e_fc11'), is_training=is_training, scope='c_bn11'))

        z_mean = linear(net, z_dim, 'e_mean')
        z_log_sigma_sq = linear(net, z_dim, 'e_log_sigma_sq')
        z_log_sigma_sq = tf.nn.softplus(z_log_sigma_sq)

        return z_mean, z_log_sigma_sq

def My_Classifier_mnist(image, z_dim, name, batch_size=64, reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()
        len_discrete_code = 4

        is_training = True
        # z_dim = 32
        x = image
        net = lrelu(conv2d(x, 64, 4, 4, 2, 2, name='c_conv1'))
        net = lrelu(bn(conv2d(net, 128, 4, 4, 2, 2, name='c_conv2'), is_training=is_training, scope='c_bn2'))
        net = tf.reshape(net, [batch_size, -1])
        net = lrelu(bn(linear(net, 1024, scope='c_fc3'), is_training=is_training, scope='c_bn3'))

        net = lrelu(bn(linear(net, 64, scope='e_fc11'), is_training=is_training, scope='c_bn11'))

        out_logit = linear(net, len_discrete_code, scope='e_fc22')
        softmaxValue = tf.nn.softmax(out_logit)

        return out_logit, softmaxValue

def MINI_Classifier(s, scopename, reuse=False):
    keep_prob = 1.0
    with tf.variable_scope(scopename, reuse=reuse):
        input = s
        n_output = 10
        n_hidden = 500
        # initializers
        w_init = tf.contrib.layers.variance_scaling_initializer()
        b_init = tf.constant_initializer(0.)

        # 1st hidden layer
        w0 = tf.get_variable('w0', [input.get_shape()[1], n_hidden], initializer=w_init)
        b0 = tf.get_variable('b0', [n_hidden], initializer=b_init)
        h0 = tf.matmul(s, w0) + b0
        h0 = tf.nn.tanh(h0)
        h0 = tf.nn.dropout(h0, keep_prob)

        n_output = 10
        # output layer-mean
        wo = tf.get_variable('wo', [h0.get_shape()[1], n_output], initializer=w_init)
        bo = tf.get_variable('bo', [n_output], initializer=b_init)
        y1 = tf.matmul(h0, wo) + bo
        y = tf.nn.softmax(y1)

    return y1, y


# Create model of CNN with slim api

class LifeLone_MNIST(object):
    def __init__(self):
        self.batch_size = 64
        self.input_height = 32
        self.input_width = 32
        self.c_dim = 3
        self.z_dim = 100
        self.len_discrete_code = 4
        self.epoch = 20

        self.learning_rate = 0.0002
        self.beta1 = 0.5

        # MNIST dataset
        mnistName = "mnist"
        fashionMnistName = "Fashion"

        mnist_train_x, mnist_train_label, mnist_test, mnist_label_test, x_train, y_train, x_test, y_test = GiveMNIST_SVHN()

        self.mnist_train_x = mnist_train_x
        self.mnist_train_y = np.zeros((np.shape(x_train)[0], 4))
        self.mnist_train_y[:, 0] = 1
        self.mnist_label = mnist_train_label
        self.mnist_label_test = mnist_label_test
        self.mnist_test_x = mnist_test
        self.mnist_test_y = np.zeros((np.shape(mnist_test)[0], 4))
        self.mnist_test_y[:, 0] = 1

        self.svhn_train_x = x_train
        self.svhn_train_y = np.zeros((np.shape(x_train)[0], 4))
        self.svhn_train_y[:, 1] = 1
        self.svhn_label = y_train
        self.svhn_label_test = y_test
        self.svhn_test_x = x_test
        self.svhn_test_y = np.zeros((np.shape(x_test)[0], 4))
        self.svhn_test_y[:, 1] = 1

        self.FashionTrain_x, self.FashionTrain_label, self.FashionTest_x, self.FashionTest_label = GiveFashion32()
        self.FashionTrain_y = np.zeros((np.shape(self.FashionTrain_x)[0], 4))
        self.FashionTrain_y[:, 2] = 1

        self.FashionTest_y = np.zeros((np.shape(self.FashionTest_x)[0], 4))
        self.FashionTest_y[:,2] = 1

        self.InverseFashionTrain_x, self.InverseFashionTrain_label, self.InverseFashionTest_x, self.InverseFashionTest_label = Give_InverseFashion32()
        self.InverseFashionTrain_y = np.zeros((np.shape(self.InverseFashionTrain_x)[0], 4))
        self.InverseFashionTrain_y[:, 3] = 1

        self.InverseFashionTest_y = np.zeros((np.shape(self.InverseFashionTest_x)[0], 4))
        self.InverseFashionTest_y[:, 3] = 1

        cifar_train_x, cifar_train_label, cifar_test_x, cifar_test_label = prepare_data()
        cifar_train_x, cifar_test_x = color_preprocessing(cifar_train_x, cifar_test_x)
        self.cifarTrainX = cifar_train_x
        self.cifarTestX = cifar_test_x
        self.cifarTrainLabels = cifar_train_label
        self.cifarTestLabels = cifar_test_label

        self.cifarTrainY = np.zeros((np.shape(self.cifarTrainX)[0], 4))
        self.cifarTrainY[:, 3] = 1

        self.cifarTestY = np.zeros((np.shape(self.cifarTestX)[0], 4))
        self.cifarTestY[:, 3] = 1

        '''
        self.cifar_train_x, self.cifar_train_label, self.cifar_test_x, self.cifar_test_label = prepare_data()
        self.cifar_train_x, self.cifar_test_x = color_preprocessing(self.cifar_train_x, self.cifar_test_x)
        self.CifarTrain_y = np.zeros((np.shape(self.cifar_train_x)[0], 4))
        self.CifarTrain_y[:, 3] = 1
        '''
    def build_model(self):
        min_value = 1e-10
        # some parameters
        image_dims = [self.input_height, self.input_width, self.c_dim]
        bs = self.batch_size
        self.inputs = tf.placeholder(tf.float32, [bs] + image_dims, name='real_images')
        self.z = tf.placeholder(tf.float32, [self.batch_size, self.z_dim], name='z')
        self.y = tf.placeholder(tf.float32, [self.batch_size, self.len_discrete_code])
        self.labels = tf.placeholder(tf.float32, [self.batch_size, 10])
        self.weights = tf.placeholder(tf.float32, [self.batch_size, 4])
        self.index = tf.placeholder(tf.int32, [self.batch_size])
        self.gan_inputs = tf.placeholder(tf.float32, [bs] + image_dims)
        self.gan_domain = tf.placeholder(tf.float32, [self.batch_size, 4])
        self.gan_domain_labels = tf.placeholder(tf.float32, [self.batch_size, 1])

        domain_labels = tf.argmax(self.gan_domain, 1)

        # GAN networks
        gan_code = self.z
        G1 = Generator_SVHN("GAN_generator1", gan_code, reuse=False)
        G2 = Generator_SVHN("GAN_generator2", gan_code, reuse=False)

        self.GAN_gen1 = G1
        self.GAN_gen2 = G2

        ## 1. GAN Loss
        # output of D for real images
        D_real_logits1 = Discriminator_SVHN_WGAN(self.inputs, "discriminator1", reuse=False)
        D_real_logits2 = Discriminator_SVHN_WGAN(self.inputs, "discriminator1", reuse=True)

        # output of D for fake images
        D_fake_logits1 = Discriminator_SVHN_WGAN(G1, "discriminator1", reuse=True)
        D_fake_logits2 = Discriminator_SVHN_WGAN(G2, "discriminator1", reuse=True)

        self.g_loss1 = -tf.reduce_mean(D_fake_logits1)
        self.g_loss2 = -tf.reduce_mean(D_fake_logits2)

        self.g_totalLoss = self.g_loss1 + self.g_loss2

        self.d_loss1 = -tf.reduce_mean(D_real_logits1) + tf.reduce_mean(D_fake_logits1)
        self.d_loss2 = -tf.reduce_mean(D_real_logits2) + tf.reduce_mean(D_fake_logits2)
        self.d_totalLoss = self.d_loss1 + self.d_loss2


        epsilon = tf.random_uniform([], 0.0, 1.0)
        x_hat1 = epsilon * self.inputs + (1 - epsilon) * G1
        d_hat1 = Discriminator_SVHN_WGAN(x_hat1, "discriminator1", reuse=True)
        scale = 10.0
        ddx1 = tf.gradients(d_hat1, x_hat1)[0]
        ddx1 = tf.sqrt(tf.reduce_sum(tf.square(ddx1), axis=1))
        ddx1 = tf.reduce_mean(tf.square(ddx1 - 1.0) * scale)
        self.d_loss1 = self.d_loss1 + ddx1

        epsilon = tf.random_uniform([], 0.0, 1.0)
        x_hat2 = epsilon * self.inputs + (1 - epsilon) * G2
        d_hat2 = Discriminator_SVHN_WGAN(x_hat2, "discriminator1", reuse=True)
        scale = 10.0
        ddx2 = tf.gradients(d_hat2, x_hat2)[0]
        ddx2 = tf.sqrt(tf.reduce_sum(tf.square(ddx2), axis=1))
        ddx2 = tf.reduce_mean(tf.square(ddx2 - 1.0) * scale)
        self.d_loss2 = self.d_loss2 + ddx2

        G_all = self.gan_inputs

        # encoder continoual information
        z_mean1, z_log_sigma_sq1 = Encoder_SVHN(G_all, "encoder", batch_size=64, reuse=False)
        continous_variables1 = z_mean1 + z_log_sigma_sq1 * tf.random_normal(tf.shape(z_mean1), 0, 1, dtype=tf.float32)

        VAE1 = Generator_SVHN("VAE_Generator", continous_variables1, reuse=False)

        KL_divergence1 = 0.5 * tf.reduce_sum(
            tf.square(z_mean1) + tf.square(z_log_sigma_sq1) - tf.log(1e-8 + tf.square(z_log_sigma_sq1)) - 1, 1)
        KL_divergence1 = tf.reduce_mean(KL_divergence1)

        reconstruction_loss1 = tf.reduce_mean(tf.reduce_sum(tf.square(VAE1 - G_all), [1, 2, 3]))

        vaeloss1 = reconstruction_loss1 + KL_divergence1 * beta
        self.vaeLoss = vaeloss1

        # Get VAE loss

        """ Training """
        # divide trainable variables into a group for D and a group for G
        T_vars = tf.trainable_variables()
        discriminator_vars1 = [var for var in T_vars if var.name.startswith('discriminator1')]

        GAN_generator_vars1 = [var for var in T_vars if var.name.startswith('GAN_generator1')]
        GAN_generator_vars2 = [var for var in T_vars if var.name.startswith('GAN_generator2')]
        VAE_encoder_vars = [var for var in T_vars if var.name.startswith('encoder')]
        VAE_generator_vars = [var for var in T_vars if var.name.startswith('VAE_Generator')]

        vae_vars = VAE_encoder_vars + VAE_generator_vars
        # optimizers
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.d_optim1 = tf.train.RMSPropOptimizer(self.learning_rate) \
                .minimize(self.d_loss1, var_list=discriminator_vars1)
            self.d_optim2 = tf.train.RMSPropOptimizer(self.learning_rate) \
                .minimize(self.d_loss2, var_list=discriminator_vars1)
            self.g_optim1 = tf.train.RMSPropOptimizer(self.learning_rate) \
                .minimize(self.g_loss1, var_list=GAN_generator_vars1)
            self.g_optim2 = tf.train.RMSPropOptimizer(self.learning_rate) \
                .minimize(self.g_loss2, var_list=GAN_generator_vars2)
            self.d_totalOptim = tf.train.RMSPropOptimizer(self.learning_rate) \
                .minimize(self.d_totalLoss, var_list=discriminator_vars1)
            self.g_totalOptim1 = tf.train.RMSPropOptimizer(self.learning_rate) \
                .minimize(self.g_totalLoss, var_list=GAN_generator_vars1+GAN_generator_vars2)

            self.vae_optim = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1) \
                .minimize(self.vaeLoss, var_list=vae_vars)
        b1 = 0

    def predict(self):
        # define the classifier
        label_logits = Image_classifier(self.inputs, "label_classifier", reuse=True)
        label_softmax = tf.nn.softmax(label_logits)
        predictions = tf.argmax(label_softmax, 1, name="predictions")
        return predictions

    def Domain_Predict(self):
        z_mean1, z_log_sigma_sq1 = Encoder_SVHN(self.inputs, "encoder", batch_size=64, reuse=True)
        continous_variables1 = z_mean1 + z_log_sigma_sq1 * tf.random_normal(tf.shape(z_mean1), 0, 1, dtype=tf.float32)

        domain_logit, domain_class = CodeImage_classifier(continous_variables1, "encoder_domain",reuse=True)
        label_softmax = tf.nn.softmax(domain_logit)
        predictions = tf.argmax(label_softmax, 1)
        return predictions

    def Predict_DomainLabels(self,testX,domainPrediction):
        totalN = np.shape(testX)[0]
        myN = int(totalN / self.batch_size)
        myPrediction = domainPrediction
        totalPredictions = []
        myCount = 0
        for i in range(myN):
            my1 = testX[self.batch_size * i:self.batch_size * (i + 1)]
            predictions = self.sess.run(myPrediction, feed_dict={self.inputs: my1})
            for k in range(self.batch_size):
                totalPredictions.append(predictions[k])

        totalPredictions = np.array(totalPredictions)
        totalPredictions = keras.utils.to_categorical(totalPredictions,4)
        return totalPredictions

    def Give_predictedLabels(self, testX):
        totalN = np.shape(testX)[0]
        myN = int(totalN / self.batch_size)
        myPrediction = self.predict()
        totalPredictions = []
        myCount = 0
        for i in range(myN):
            my1 = testX[self.batch_size * i:self.batch_size * (i + 1)]
            predictions = self.sess.run(myPrediction, feed_dict={self.inputs: my1})
            for k in range(self.batch_size):
                totalPredictions.append(predictions[k])

        totalPredictions = np.array(totalPredictions)
        totalPredictions = keras.utils.to_categorical(totalPredictions)
        return totalPredictions

    def Calculate_accuracy(self, testX, testY):
        # testX = self.mnist_test_x
        totalN = np.shape(testX)[0]
        myN = int(totalN / self.batch_size)
        myPrediction = self.predict()
        totalPredictions = []
        myCount = 0
        for i in range(myN):
            my1 = testX[self.batch_size * i:self.batch_size * (i + 1)]
            predictions = self.sess.run(myPrediction, feed_dict={self.inputs: my1})
            for k in range(self.batch_size):
                totalPredictions.append(predictions[k])

        totalPredictions = np.array(totalPredictions)

        testLabels = testY[0:np.shape(totalPredictions)[0]]
        testLabels = np.argmax(testLabels, 1)
        trueCount = 0
        for k in range(np.shape(testLabels)[0]):
            if testLabels[k] == totalPredictions[k]:
                trueCount = trueCount + 1

        accuracy = (float)(trueCount / np.shape(testLabels)[0])

        return accuracy

    def Give_RealReconstruction(self):
        z_mean1, z_log_sigma_sq1 = Encoder_SVHN(self.inputs, "encoder", batch_size=64, reuse=True)
        continous_variables1 = z_mean1 + z_log_sigma_sq1 * tf.random_normal(tf.shape(z_mean1), 0, 1, dtype=tf.float32)

        VAE1 = Generator_SVHN("VAE_Generator", continous_variables1, reuse=True)

        reconstruction_loss1 = tf.reduce_mean(tf.reduce_sum(tf.square(VAE1 - self.inputs), [1, 2, 3]))

        return reconstruction_loss1

    def Give_Elbo(self):
        z_mean1, z_log_sigma_sq1 = Encoder_SVHN(self.inputs, "encoder", batch_size=64, reuse=True)
        continous_variables1 = z_mean1 + z_log_sigma_sq1 * tf.random_normal(tf.shape(z_mean1), 0, 1, dtype=tf.float32)

        VAE1 = Generator_SVHN("VAE_Generator", continous_variables1, reuse=True)

        reconstruction_loss1 = tf.reduce_mean(tf.reduce_sum(tf.square(VAE1 - self.inputs), [1, 2, 3]))

        KL_divergence1 = 0.5 * tf.reduce_sum(
            tf.square(z_mean1) + tf.square(z_log_sigma_sq1) - tf.log(1e-8 + tf.square(z_log_sigma_sq1)) - 1, 1)
        KL_divergence1 = tf.reduce_mean(KL_divergence1)

        loss = reconstruction_loss1 + KL_divergence1

        return loss

    def Calculate_ReconstructionErrors2(self,testX,myPro):
        p1 = int(np.shape(testX)[0] / self.batch_size)
        sumError = 0
        for i in range(p1):
            g = testX[i * self.batch_size:(i + 1) * self.batch_size]
            sumError = sumError + self.sess.run(myPro, feed_dict={self.inputs: g})

        sumError = sumError / p1
        return sumError

    def Calculate_ReconstructionErrors(self,testX):
        p1 = int(np.shape(testX)[0]/self.batch_size)
        myPro = self.Give_RealReconstruction()
        sumError = 0
        for i in range(p1):
            g = testX[i*self.batch_size:(i+1)*self.batch_size]
            sumError = sumError + self.sess.run(myPro,feed_dict={self.inputs:g})

        sumError = sumError / p1
        return sumError

    def Calculate_Elbo(self,testX):
        p1 = int(np.shape(testX)[0]/self.batch_size)
        myPro = self.Give_Elbo()
        sumError = 0
        for i in range(p1):
            g = testX[i*self.batch_size:(i+1)*self.batch_size]
            sumError = sumError + self.sess.run(myPro,feed_dict={self.inputs:g})

        sumError = sumError / p1
        return sumError

    def Calculate_Elbo2(self,testX,myPro):
        p1 = int(np.shape(testX)[0]/self.batch_size)
        sumError = 0
        for i in range(p1):
            g = testX[i*self.batch_size:(i+1)*self.batch_size]
            sumError = sumError + self.sess.run(myPro,feed_dict={self.inputs:g})

        sumError = sumError / p1
        return sumError

    def test(self):

        config = tf.ConfigProto(allow_soft_placement=True)
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
        config.gpu_options.allow_growth = True

        z_dim = 150
        z_mean1, z_log_sigma_sq1 = Encoder_SVHN(self.inputs, "encoder", batch_size=64, reuse=True)
        continous_variables1 = z_mean1 + z_log_sigma_sq1 * tf.random_normal(tf.shape(z_mean1), 0, 1, dtype=tf.float32)

        VAE1 = Generator_SVHN("VAE_Generator", continous_variables1, reuse=True)

        testX = np.concatenate((self.mnist_test_x,self.svhn_test_x,self.FashionTest_x,self.InverseFashionTest_x),axis=0)
        testY = np.concatenate((self.mnist_test_y,self.svhn_test_y,self.FashionTest_y,self.InverseFashionTest_y),axis=0)
        testY = np.concatenate((self.mnist_test_y,self.svhn_test_y,self.FashionTest_y,self.InverseFashionTest_y),axis=0)

        with tf.Session(config=config) as sess:
            self.saver = tf.train.Saver()

            self.sess = sess
            sess.run(tf.global_variables_initializer())
            self.saver.restore(sess, 'models/TwinGenerative_Unsupervised_ThreeData')


            for cIndex in range(10):
                batch_images = np.concatenate((self.mnist_test_x[0:self.batch_size], self.FashionTest_x[0:self.batch_size],
                                               self.InverseFashionTest_x[0:self.batch_size],
                                               self.svhn_test_x[0:self.batch_size]), axis=0)

                index = [i for i in range(np.shape(batch_images)[0])]
                random.shuffle(index)
                batch_images = batch_images[index]
                batch_images = batch_images[0:self.batch_size]
                reconstruction = self.sess.run(
                    VAE1,
                    feed_dict={self.inputs: batch_images})

                mySize = 10
                ims("results/" + "TwinGANs_VAE_recon" + str(cIndex) + ".png", merge(reconstruction[:mySize], [1, mySize]))
                ims("results/" + "TwinGANs_Real" + str(cIndex) + ".png", merge(batch_images[:mySize], [1, mySize]))

                batch_z = np.random.uniform(-1, 1, [self.batch_size, self.z_dim]).astype(np.float32)
                g1, g2 = self.sess.run([self.GAN_gen1, self.GAN_gen2], feed_dict={self.z: batch_z})
                ims("results/" + "TwinGANs_GAN1_1" + str(cIndex) + ".png", merge(g1[:mySize], [1, mySize]))
                ims("results/" + "TwinGANs_GAN2_2" + str(cIndex) + ".png", merge(g2[:mySize], [1, mySize]))

            bc = 0

            mnistError = self.Calculate_ReconstructionErrors(self.mnist_test_x)
            fashionError = self.Calculate_ReconstructionErrors(self.FashionTest_x)
            svhnError = self.Calculate_ReconstructionErrors(self.svhn_test_x)
            IFashionError = self.Calculate_ReconstructionErrors(self.InverseFashionTest_x)

            mnistElbo = self.Calculate_Elbo(self.mnist_test_x)
            fashionElbo = self.Calculate_Elbo(self.FashionTest_x)
            svhnElbo = self.Calculate_Elbo(self.svhn_test_x)
            IFashionElbo = self.Calculate_Elbo(self.InverseFashionTest_x)

            print(mnistError)
            print('\n')
            print(mnistElbo)
            print('\n')
            print(fashionError)
            print('\n')
            print(fashionElbo)
            print('\n')
            print(svhnError)
            print('\n')
            print(svhnElbo)

            myIndex = 2

    def train(self):

        taskCount = 3

        config = tf.ConfigProto(allow_soft_placement=False)
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
        config.gpu_options.allow_growth = True

        myPro = self.Give_RealReconstruction()
        myElbo = self.Give_Elbo()

        isFirstStage = True
        with tf.Session(config=config) as sess:
            self.sess = sess
            sess.run(tf.global_variables_initializer())
            self.saver = tf.train.Saver()

            # self.saver.restore(sess, 'models/TeacherStudent_MNIST_TO_Fashion_invariant')

            # saver to save model
            self.saver = tf.train.Saver()
            ExpertWeights = np.ones((self.batch_size, 4))
            DomainState = np.zeros(4).astype(np.int32)
            DomainState[0] = 0
            DomainState[1] = 1
            DomainState[2] = 2
            DomainState[3] = 3

            elboArr = []
            recoArr = []

            for taskIndex in range(taskCount):

                if taskIndex == 0:
                    currentTrainX = self.mnist_train_x
                    currentTrainY = self.mnist_train_y
                    currentTrain_labels = self.mnist_label
                elif taskIndex == 1:
                    currentTrainX = self.svhn_train_x
                    currentTrainY = self.svhn_train_y
                    currentTrain_labels = self.svhn_label
                elif taskIndex == 2:
                    currentTrainX = self.FashionTrain_x
                    currentTrainY = self.FashionTrain_y
                    currentTrain_labels = self.FashionTrain_label
                elif taskIndex == 3:
                    currentTrainX = self.InverseFashionTrain_x
                    currentTrainY = self.InverseFashionTrain_y
                    currentTrain_labels = self.InverseFashionTrain_label
                elif taskIndex == 4:
                    currentTrainX = self.mnist_train_x
                    currentTrainY = self.mnist_train_y
                    currentTrain_labels = self.mnist_label
                    '''
                    currentTrainX = self.cifar_train_x
                    currentTrainY = self.CifarTrain_y
                    currentTrain_labels = self.cifar_train_label
                    '''

                dataX = currentTrainX
                labelsY = currentTrain_labels
                dataY = currentTrainY
                n_examples = np.shape(dataX)[0]

                start_epoch = 0
                start_batch_id = 0
                self.num_batches = int(n_examples / self.batch_size)

                mnistAccuracy_list = []
                mnistFashionAccuracy_list = []

                # loop for epoch
                start_time = time.time()
                for epoch in range(start_epoch, self.epoch):
                    count = 0
                    # Random shuffling
                    index = [i for i in range(n_examples)]
                    random.shuffle(index)
                    dataX = dataX[index]
                    dataY = dataY[index]
                    labelsY = labelsY[index]
                    counter = 0

                    t = 1
                    if taskIndex != 0:
                        t = 2
                    generatedSamples = 0
                    # get batch data
                    for idx in range(start_batch_id, self.num_batches*t):

                        if taskIndex == 0:
                            batch_images = dataX[idx * self.batch_size:(idx + 1) * self.batch_size]
                            batch_y = dataY[idx * self.batch_size:(idx + 1) * self.batch_size]
                            batch_labels = labelsY[idx * self.batch_size:(idx + 1) * self.batch_size]
                        else:
                            batch_images = dataX[idx * 32:(idx + 1) * 32]
                            batch_y = dataY[idx * 32:(idx + 1) * 32]
                            batch_labels = labelsY[idx * 32:(idx + 1) * 32]

                        # update GAN
                        batch_z = np.random.uniform(-1, 1, [self.batch_size, self.z_dim]).astype(np.float32)

                        dataIndex = [i for i in range(self.batch_size)]
                        random.shuffle(dataIndex)

                        if taskIndex == 0:
                            # update D network
                            _, d_loss = self.sess.run([self.d_totalOptim, self.d_totalLoss],
                                                      feed_dict={self.inputs: batch_images,
                                                                 self.z: batch_z,self.gan_inputs:batch_images})

                            # update G and Q network
                            _, g_loss = self.sess.run(
                                [self.g_totalOptim1, self.g_totalLoss],
                                feed_dict={self.inputs: batch_images, self.z: batch_z,self.gan_inputs:batch_images})

                        else:
                            if taskIndex % 2 == 0:
                                generatedSamples = self.sess.run(self.GAN_gen1,feed_dict={self.inputs: dataX[0:self.batch_size], self.z: batch_z})
                                newSample = np.concatenate((batch_images,generatedSamples[0:32]),axis=0)

                                myIndex = [i for i in range(self.batch_size)]
                                random.shuffle(myIndex)
                                newSample = newSample[myIndex]

                                # update D network
                                _, d_loss = self.sess.run(
                                    [self.d_optim2, self.d_loss2],
                                    feed_dict={self.gan_inputs: newSample,self.inputs: newSample,self.z:batch_z
                                               })

                                # update G and Q network
                                _, g_loss = self.sess.run(
                                    [self.g_optim2, self.g_loss2],
                                    feed_dict={self.gan_inputs: newSample,self.inputs: newSample,self.z:batch_z
                                                })
                            else:
                                generatedSamples = self.sess.run(self.GAN_gen2,
                                                                 feed_dict={self.inputs: dataX[0:self.batch_size], self.z: batch_z})

                                newSample = np.concatenate((batch_images, generatedSamples[0:32]), axis=0)
                                myIndex = [i for i in range(self.batch_size)]
                                random.shuffle(myIndex)
                                newSample = newSample[myIndex]

                                # update D network
                                _, d_loss = self.sess.run(
                                    [self.d_optim1, self.d_loss1],
                                    feed_dict={self.gan_inputs:newSample,self.inputs: newSample, self.z: batch_z
                                               })

                                # update G and Q network
                                _, g_loss = self.sess.run(
                                    [self.g_optim1, self.g_loss1],
                                    feed_dict={self.gan_inputs:newSample,self.inputs: newSample, self.z: batch_z
                                               }
                                )

                        if taskIndex == 0:
                            batch_y = np.zeros((self.batch_size,4))
                            batch_y[:,0] = 1
                            # update G and Q network
                            _, vaeLoss = self.sess.run(
                                [self.vae_optim, self.vaeLoss],
                                feed_dict={self.inputs:batch_images,self.z:batch_z,
                                           self.gan_inputs: batch_images,self.gan_domain:batch_y})
                        else:
                            newSample = np.concatenate((batch_images, generatedSamples[0:32]), axis=0)
                            myIndex = [i for i in range(self.batch_size)]
                            random.shuffle(myIndex)
                            newSample = newSample[myIndex]

                            _, vaeLoss = self.sess.run(
                                [self.vae_optim, self.vaeLoss],
                                feed_dict={self.inputs:newSample,self.z:batch_z,
                                           self.gan_inputs: newSample})

                        # display training status
                        counter += 1
                        print(
                            "Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f, vae_loss:%.8f. c_loss:%.8f" \
                            % (epoch, idx, self.num_batches, time.time() - start_time, d_loss, g_loss, 0, 0))

                    g1, g2= self.sess.run(
                        [self.GAN_gen1, self.GAN_gen2],
                        feed_dict={self.inputs:dataX[0:self.batch_size], self.z: batch_z,self.gan_inputs: dataX[0:self.batch_size]
                                   })
                    elbo1 = self.Calculate_Elbo2(self.mnist_test_x,myElbo)
                    reco1 = self.Calculate_Elbo2(self.mnist_test_x,myPro)

                    recoArr.append(reco1)
                    elboArr.append(elbo1)

                    ims("results/" + "G_1_" + str(epoch) + ".png", merge(g1[:64], [8, 8]))
                    ims("results/" + "G_2_" + str(epoch) + ".png", merge(g2[:64], [8, 8]))

            lossArr1 = np.array(recoArr).astype('str')
            f = open("results/MNIST_reco" + str(beta) + ".txt", "w", encoding="utf-8")
            for i in range(np.shape(lossArr1)[0]):
                f.writelines(lossArr1[i])
                f.writelines('\n')
            f.flush()
            f.close()

            lossArr2 = np.array(elboArr).astype('str')
            f = open("results/MNIST_elbo" + str(beta) + ".txt", "w", encoding="utf-8")
            for i in range(np.shape(lossArr1)[0]):
                f.writelines(lossArr2[i])
                f.writelines('\n')
            f.flush()
            f.close()

            #self.saver.save(self.sess, "models/TwinGenerative_Unsupervised_ThreeData")

infoMultiGAN = LifeLone_MNIST()
infoMultiGAN.build_model()
infoMultiGAN.train()
# infoMultiGAN.train_classifier()
#infoMultiGAN.test()
