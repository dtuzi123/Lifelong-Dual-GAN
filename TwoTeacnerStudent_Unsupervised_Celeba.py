from keras.datasets import mnist
import time
from utils import *
from scipy.misc import imsave as ims
from ops import *
from utils import *
from Utlis2 import *
import random as random
from glob import glob
import os, gzip
import keras as keras
from glob import glob
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from Basic_structure import *

os.environ['CUDA_VISIBLE_DEVICES']='0'

def file_name(file_dir):
    t1 = []
    file_dir = "../LifelongMixtureGAN/data/images_background/"
    for root, dirs, files in os.walk(file_dir):
        for a1 in dirs:
            b1 = "F:/Third_Experiment/Multiple_GAN_codes/data/images_background/" + a1 + "/renders/*.png"
            b1 = "../LifelongMixtureGAN/data/images_background/" + a1
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

def file_name2(file_dir):
    t1 = []
    for root, dirs, files in os.walk(file_dir):
        for a1 in dirs:
            b1 = "../rendered_chairs/" + a1 + "/renders/*.png"
            img_path = glob(b1)
            t1.append(img_path)

    cc = []

    for i in range(len(t1)):
        a1 = t1[i]
        for p1 in a1:
            cc.append(p1)
    return cc

def Generator_Celeba(name, z, batch_size=64, reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()

        kernel = 3
        # fully-connected layers
        h1 = linear(z, 256 * 8 * 8, 'g_h1_lin')
        h1 = tf.reshape(h1, [batch_size, 8, 8, 256])
        h1 = tf.nn.relu(g_bn1(h1))

        # deconv layers
        h2 = deconv2d(h1, [batch_size, 16, 16, 256],
                      kernel, kernel, 2, 2, name='g_h2')
        h2 = tf.nn.relu(g_bn2(h2))

        h3 = deconv2d(h2, [batch_size, 16, 16, 256],
                      kernel, kernel, 1, 1, name='g_h3')
        h3 = tf.nn.relu(g_bn3(h3))

        h4 = deconv2d(h3, [batch_size, 32, 32, 256],
                      kernel, kernel, 2, 2, name='g_h4')
        h4 = tf.nn.relu(g_bn4(h4))

        h5 = deconv2d(h4, [batch_size, 32, 32, 256],
                      kernel, kernel, 1, 1, name='g_h5')
        h5 = tf.nn.relu(g_bn5(h5))

        h6 = deconv2d(h5, [batch_size, 64, 64, 128],
                      kernel, kernel, 2, 2, name='g_h6')
        h6 = tf.nn.relu(g_bn6(h6))

        '''
        h7 = deconv2d(h6, [batch_size, 128, 128, 64],
                      5, 5, 2, 2, name='g_h7')
        h7 = tf.nn.relu(g_bn7(h7))
        '''
        h8 = deconv2d(h6, [batch_size, 64, 64, 3],
                      kernel, kernel, 1, 1, name='g_h8')
        h8 = tf.nn.tanh(h8)

        return h8

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
def Image_classifier(inputs, scopename, is_training=True, reuse=False):
    with tf.variable_scope(scopename, reuse=reuse):
        batch_norm_params = {'is_training': is_training, 'decay': 0.9, 'updates_collections': None}
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params):
            x = tf.reshape(inputs, [-1, 28, 28, 1])

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
            outputs = slim.fully_connected(net, 10, activation_fn=None, normalizer_fn=None, scope='fco')
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


class LifeLone_MNIST(object):
    def __init__(self):
        self.batch_size = 64
        self.input_height = 64
        self.input_width = 64
        self.c_dim = 3
        self.z_dim = 256
        self.len_discrete_code = 4
        self.epoch = 20

        self.learning_rate = 0.00002
        self.beta1 = 0.5

    def Give_loss_penality(self,G):
        epsilon = tf.random_uniform([], 0.0, 1.0)
        x_hat = epsilon * self.inputs + (1 - epsilon) * G
        d_hat = Discriminator_Celeba(x_hat, "discriminator", reuse=True)
        scale = 10.0
        ddx = tf.gradients(d_hat, x_hat)[0]
        ddx = tf.sqrt(tf.reduce_sum(tf.square(ddx), axis=1))
        ddx = tf.reduce_mean(tf.square(ddx - 1.0) * scale)
        return ddx

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

        # GAN networks
        gan_code = self.z
        G1 = Generator_Celeba("GAN_generator1", gan_code, reuse=False)
        G2 = Generator_Celeba("GAN_generator2", gan_code, reuse=False)

        ## 1. GAN Loss
        # output of D for real images
        D_real_logits1 = Discriminator_Celeba(self.inputs, "discriminator1", reuse=False)
        D_real_logits2 = Discriminator_Celeba(self.inputs, "discriminator2", reuse=False)

        # output of D for fake images
        D_fake_logits1 = Discriminator_Celeba(G1, "discriminator1", reuse=True)
        D_fake_logits2 = Discriminator_Celeba(G2, "discriminator2", reuse=True)

        self.g_loss1 = -tf.reduce_mean(D_fake_logits1)
        self.g_loss2 = -tf.reduce_mean(D_fake_logits2)

        self.d_loss1 = -tf.reduce_mean(D_real_logits1)+tf.reduce_mean(D_fake_logits1)
        self.d_loss2 = -tf.reduce_mean(D_real_logits2)+tf.reduce_mean(D_fake_logits2)

        epsilon = tf.random_uniform([], 0.0, 1.0)
        x_hat1 = epsilon * self.inputs + (1 - epsilon) * G1
        d_hat1 = Discriminator_Celeba(x_hat1, "discriminator1", reuse=True)
        scale = 10.0
        ddx1 = tf.gradients(d_hat1, x_hat1)[0]
        ddx1 = tf.sqrt(tf.reduce_sum(tf.square(ddx1), axis=1))
        ddx1 = tf.reduce_mean(tf.square(ddx1 - 1.0) * scale)
        self.d_loss1 = self.d_loss1 + ddx1

        epsilon = tf.random_uniform([], 0.0, 1.0)
        x_hat2 = epsilon * self.inputs + (1 - epsilon) * G2
        d_hat2 = Discriminator_Celeba(x_hat2, "discriminator2", reuse=True)
        scale = 10.0
        ddx2 = tf.gradients(d_hat2, x_hat2)[0]
        ddx2 = tf.sqrt(tf.reduce_sum(tf.square(ddx2), axis=1))
        ddx2 = tf.reduce_mean(tf.square(ddx2 - 1.0) * scale)
        self.d_loss2 = self.d_loss2 + ddx2

        self.GAN_gen1 = G1
        self.GAN_gen2 = G2

        self.isPhase = 0

        #VAE loss start

        z_mean1, z_log_sigma_sq1 = Encoder_Celeba2(self.gan_inputs, "encoder", batch_size=64, reuse=False)
        continous_variables1 = z_mean1 + z_log_sigma_sq1 * tf.random_normal(tf.shape(z_mean1), 0, 1, dtype=tf.float32)

        domain_logit, domain_class = CodeImage_classifier(continous_variables1, "encoder_domain")
        log_y = tf.log(tf.nn.softmax(domain_logit) + 1e-10)
        discrete_real = my_gumbel_softmax_sample(log_y, np.arange(self.len_discrete_code))

        self.domain_classloss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=domain_logit, labels=self.gan_domain))

        y_labels = tf.argmax(self.gan_domain, 1)
        y_labels = tf.cast(y_labels, dtype=tf.float32)
        y_labels = tf.reshape(y_labels, (-1, 1))

        code = tf.concat((continous_variables1, discrete_real), axis=1)
        VAE1 = Generator_Celeba("VAE_Generator", code, reuse=False)

        KL_divergence1 = 0.5 * tf.reduce_sum(
            tf.square(z_mean1 - y_labels) + tf.square(z_log_sigma_sq1) - tf.log(1e-8 + tf.square(z_log_sigma_sq1)) - 1,
            1)
        KL_divergence1 = tf.reduce_mean(KL_divergence1)

        reconstruction_loss1 = tf.reduce_mean(tf.reduce_sum(tf.square(VAE1 - self.gan_inputs), [1, 2, 3]))

        vaeloss1 = reconstruction_loss1 + KL_divergence1
        self.vaeLoss = vaeloss1

        T_vars = tf.trainable_variables()
        discriminator_vars1 = [var for var in T_vars if var.name.startswith('discriminator1')]
        discriminator_vars2 = [var for var in T_vars if var.name.startswith('discriminator2')]

        GAN_generator_vars1 = [var for var in T_vars if var.name.startswith('GAN_generator1')]
        GAN_generator_vars2 = [var for var in T_vars if var.name.startswith('GAN_generator2')]
        VAE_encoder_vars = [var for var in T_vars if var.name.startswith('encoder')]
        VAE_generator_vars = [var for var in T_vars if var.name.startswith('VAE_Generator')]
        VAE_encoder_domain_vars = [var for var in T_vars if var.name.startswith('encoder_domain')]

        vae_vars = VAE_encoder_vars + VAE_generator_vars+VAE_encoder_domain_vars
        # optimizers
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.d_optim1 = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1) \
                .minimize(self.d_loss1, var_list=discriminator_vars1)
            self.g_optim1 = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1) \
                .minimize(self.g_loss1, var_list=GAN_generator_vars1)
            self.d_optim2 = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1) \
                .minimize(self.d_loss2, var_list=discriminator_vars2)
            self.g_optim2 = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1) \
                .minimize(self.g_loss2, var_list=GAN_generator_vars2)
            self.vae_optim = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1) \
                .minimize(self.vaeLoss, var_list=vae_vars)
            self.domain_optim = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1) \
                .minimize(self.domain_classloss, var_list=VAE_encoder_domain_vars)

        self.fakeImage = G1
        b1 = 0

    def predict(self):
        # define the classifier
        label_logits = Image_classifier(self.inputs, "label_classifier", reuse=True)
        label_softmax = tf.nn.softmax(label_logits)
        predictions = tf.argmax(label_softmax, 1, name="predictions")
        return predictions

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

    def test(self):
        config = tf.ConfigProto(allow_soft_placement=True)
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
            self.saver = tf.train.Saver()

            self.sess = sess
            sess.run(tf.global_variables_initializer())
            self.saver.restore(sess, 'models/TwoTeacherStudent_Unsupervised_Celeba')

            # load Human face
            batch_size = 64
            img_path = glob('C:/CommonData/img_celeba2/*.jpg')  # 获取新文件夹下所有图片
            data_files = img_path
            data_files = sorted(data_files)
            data_files = np.array(data_files)  # for tl.iterate.minibatches
            n_examples = 202599
            total_batch = int(n_examples / self.batch_size)

            myIndex = 5
            batch_files = data_files[myIndex * self.batch_size:
                                     (myIndex + 1) * self.batch_size]

            myIndex = 6
            batch_files2 = data_files[myIndex * self.batch_size:
                                      (myIndex + 1) * self.batch_size]

            batch = [get_image(
                sample_file,
                input_height=128,
                input_width=128,
                resize_height=64,
                resize_width=64,
                crop=True)
                for sample_file in batch_files]

            batch_images = np.array(batch).astype(np.float32)
            x_fixed1 = batch_images

            batch = [get_image(
                sample_file,
                input_height=128,
                input_width=128,
                resize_height=64,
                resize_width=64,
                crop=True)
                for sample_file in batch_files2]

            batch_images = np.array(batch).astype(np.float32)
            x_fixed1_1 = batch_images

            # load 3D chairs
            img_path = glob('C:/CommonData/CACD2000/CACD2000/*.jpg')  # 获取新文件夹下所有图片
            data_files = img_path
            data_files = sorted(data_files)
            data_files = np.array(data_files)  # for tl.iterate.minibatches
            n_examples = np.shape(data_files)[0]
            total_batch = int(n_examples / self.batch_size)

            batch_files = data_files[0:
                                     self.batch_size]

            # load dataset
            count1 = 0
            image_size = 64

            total_batch = int(n_examples / batch_size)
            total_batch = int(total_batch)
            batch_files = data_files[0:
                                     batch_size]


            # load dataset
            count1 = 0
            image_size = 64

            total_batch = int(n_examples / batch_size)
            total_batch = int(total_batch)
            myIndex = 10
            batch_files = data_files[myIndex * self.batch_size:
                                     (myIndex + 1) * batch_size]
            myIndex = 11
            batch_files2 = data_files[myIndex * self.batch_size:
                                      (myIndex + 1) * batch_size]

            batch = [get_image(
                batch_file,
                input_height=250,
                input_width=250,
                resize_height=64,
                resize_width=64,
                crop=False) \
                for batch_file in batch_files]
            batch2 = batch = [get_image(
                batch_file,
                input_height=250,
                input_width=250,
                resize_height=64,
                resize_width=64,
                crop=False) \
                for batch_file in batch_files2]

            batch_images = np.array(batch).astype(np.float32)
            x_fixed2 = batch_images

            batch_images2 = np.array(batch2).astype(np.float32)
            x_fixed2_2 = batch_images2

            myIndex = 2

            testX = np.concatenate((x_fixed1, x_fixed2), axis=0)
            testY = np.zeros((self.batch_size * 2, 4))
            testY[0:self.batch_size, 0] = 1
            testY[self.batch_size:self.batch_size * 2, 0] = 1

            index = [i for i in range(np.shape(testX)[0])]
            random.shuffle(index)
            testX = testX[index]
            testY = testY[index]

            batch_z = np.random.uniform(-1, 1, [self.batch_size, self.z_dim]).astype(np.float32)

            g1,g2,g3,g4 = sess.run([self.GAN_gen1,self.GAN_gen2,self.GAN_gen3,self.GAN_gen4],feed_dict={self.z:batch_z})
            ims("results/" + "GAN" + str(0) + ".jpg", merge2(g1[:64], [8, 8]))
            ims("results/" + "GAN" + str(1) + ".jpg", merge2(g2[:64], [8, 8]))
            ims("results/" + "GAN" + str(2) + ".jpg", merge2(g3[:64], [8, 8]))
            ims("results/" + "GAN" + str(3) + ".jpg", merge2(g4[:64], [8, 8]))
            h = 0

    def Generate_GAN_Samples(self, n_samples, classN):
        myArr = []
        for tt in range(classN):
            y1 = np.zeros((self.batch_size, 4))
            y1[:, 0] = 1
            num1 = int(n_samples / self.batch_size)
            for i in range(num1):
                batch_z = np.random.uniform(-1, 1, [self.batch_size, self.z_dim]).astype(np.float32)
                batch_labels = np.random.multinomial(1,
                                                     10 * [
                                                         float(1.0 / 10)],
                                                     size=[self.batch_size])
                g_outputs = self.sess.run(
                    self.GAN_output,
                    feed_dict={self.z: batch_z, self.y: y1, self.labels: batch_labels})
                for t1 in range(self.batch_size):
                    myArr.append(g_outputs[t1])

        myArr = np.array(myArr)
        return myArr

    def Domain_Predict(self):
        z_mean1, z_log_sigma_sq1 = Encoder_Celeba2(self.gan_inputs, "encoder", batch_size=64, reuse=True)
        continous_variables1 = z_mean1 + z_log_sigma_sq1 * tf.random_normal(tf.shape(z_mean1), 0, 1, dtype=tf.float32)

        domain_logit, domain_class = CodeImage_classifier(continous_variables1, "encoder_domain",reuse=True)
        label_softmax = tf.nn.softmax(domain_logit)
        predictions = tf.argmax(label_softmax, 1)
        return predictions

    def Predict_DomainLabels(self,testX,myPrediction):
        totalN = np.shape(testX)[0]
        myN = int(totalN / self.batch_size)
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

    def train(self):

        config = tf.ConfigProto(allow_soft_placement=True)
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
        config.gpu_options.allow_growth = True

        img_path = glob('../img_celeba2/*.jpg')  # 获取新文件夹下所有图片
        #img_path = glob('C:/CommonData/img_celeba2/*.jpg')  # 获取新文件夹下所有图片
        data_files = img_path
        data_files = sorted(data_files)
        data_files = np.array(data_files)  # for tl.iterate.minibatches
        celebaFiles = data_files

        myPrediction = self.Domain_Predict()

        x_fixed = [get_image(
            sample_file,
            input_height=128,
            input_width=128,
            resize_height=64,
            resize_width=64,
            crop=True)
            for sample_file in celebaFiles[0:self.batch_size]]
        x_fixed = np.array(x_fixed)

        # load 3D chairs
        img_path = glob('../CACD2000/CACD2000/*.jpg')  # 获取新文件夹下所有图片
        data_files = img_path
        data_files = sorted(data_files)
        data_files = np.array(data_files)  # for tl.iterate.minibatches
        cacdFiles = data_files

        file_dir = "../rendered_chairs/"
        files = file_name2(file_dir)
        data_files = files
        data_files = sorted(data_files)
        data_files = np.array(data_files)  # for tl.iterate.minibatches
        chairFiles = data_files

        files = file_name(1)
        data_files = files
        data_files = sorted(data_files)
        data_files = np.array(data_files)  # for tl.iterate.minibatches
        zimuFiles = data_files

        taskCount = 4

        isFirstStage = False
        with tf.Session(config=config) as sess:
            self.sess = sess
            sess.run(tf.global_variables_initializer())
            self.saver = tf.train.Saver()

            # saver to save model
            self.saver = tf.train.Saver()
            batch_size = self.batch_size

            for taskIndex in range(taskCount):

                if taskIndex == 0:
                    x_train = celebaFiles
                    x_fixed = x_train[0:batch_size]
                elif taskIndex == 1:
                    x_train = cacdFiles
                    x_fixed = x_train[0:batch_size]
                elif taskIndex == 2:
                    x_train = chairFiles
                    x_fixed = x_train[0:batch_size]
                elif taskIndex == 3:
                    x_train = zimuFiles
                    x_fixed = x_train[0:batch_size]

                n_samples = np.shape(np.array(x_train))[0]
                total_batch = int(n_samples / batch_size)

                start_epoch = 0

                tt = 2
                if taskIndex == 0:
                    tt=1

                # loop for epoch
                start_time = time.time()
                for epoch in range(start_epoch, self.epoch):
                    count = 0
                    # Random shuffling
                    index = [i for i in range(n_samples)]
                    random.shuffle(index)
                    x_train[index] = x_train[index]

                    counter = 0
                    # get batch data
                    for i in range(total_batch*tt):

                        batchFiles = x_train[i * batch_size:i * batch_size + batch_size]

                        if taskIndex == 0:
                            batch = [get_image(
                                sample_file,
                                input_height=128,
                                input_width=128,
                                resize_height=64,
                                resize_width=64,
                                crop=True)
                                for sample_file in batchFiles]
                        elif taskIndex == 1:
                            batch = [get_image(
                                sample_file,
                                input_height=250,
                                input_width=250,
                                resize_height=64,
                                resize_width=64,
                                crop=True)
                                for sample_file in batchFiles]
                        elif taskIndex == 2:
                            image_size = 64
                            batch = [get_image2(batch_file, 300, is_crop=True, resize_w=image_size, is_grayscale=0) \
                                     for batch_file in batchFiles]
                        elif taskIndex == 3:
                            batch = [get_image(batch_file, 105, 105,
                                               resize_height=64, resize_width=64,
                                               crop=False, grayscale=False) \
                                     for batch_file in batchFiles]
                            batch = np.array(batch)
                            batch = np.reshape(batch, (64, 64, 64, 1))
                            batch = np.concatenate((batch, batch, batch), axis=-1)

                        # Compute the offset of the current minibatch in the data.
                        batch_xs_target = batch
                        x_fixed = batch
                        batch_xs_input = batch
                        batch_images = batch
                        batch_images = np.array(batch_images)

                        if taskIndex == 0:
                            # update GAN
                            batch_z = np.random.uniform(-1, 1, [self.batch_size, self.z_dim]).astype(np.float32)

                            # update D network
                            _, d_loss,_ = self.sess.run([self.d_optim1,self.d_loss1,self.d_optim2],
                                                      feed_dict={self.inputs: batch_images,
                                                                 self.z: batch_z})

                            # update G and Q network
                            _, g_loss,_ = self.sess.run(
                                [self.g_optim1, self.g_loss1,self.g_optim2],
                                feed_dict={self.inputs: batch_images, self.z: batch_z})

                            newY = np.zeros((self.batch_size,4))
                            newY[:,0] = 1
                            _, vaeLoss, _ = self.sess.run(
                                [self.vae_optim, self.vaeLoss, self.domain_optim],
                                feed_dict={self.inputs: batch_images, self.z: batch_z,
                                           self.gan_inputs: batch_images,self.gan_domain:newY })

                        elif taskIndex %2 == 0:
                            batch_z = np.random.uniform(-1, 1, [self.batch_size, self.z_dim]).astype(np.float32)
                            generatedImages = self.sess.run(
                                self.GAN_gen1,
                                feed_dict={self.inputs: x_fixed, self.z: batch_z,
                                            self.gan_inputs: x_fixed})

                            generatedY = self.Predict_DomainLabels(generatedImages,myPrediction)

                            newY = np.zeros((32,4))
                            newY[:,taskIndex] = 1
                            newX = np.concatenate((batch_images,generatedImages[0:32]),axis=0)
                            newY = np.concatenate((newY,generatedY[0:32]),axis=0)

                            # update D network
                            _, d_loss = self.sess.run([self.d_optim2, self.d_loss2],
                                                         feed_dict={self.inputs: newX,
                                                                    self.z: batch_z})

                            # update G and Q network
                            _, g_loss = self.sess.run(
                                [self.g_optim2, self.g_loss2],
                                feed_dict={self.inputs: newX, self.z: batch_z})

                            _, vaeLoss, _ = self.sess.run(
                                [self.vae_optim, self.vaeLoss, self.domain_optim],
                                feed_dict={self.inputs: newX, self.z: batch_z,
                                           self.gan_inputs: newX, self.gan_domain: newY})
                        else:
                            batch_z = np.random.uniform(-1, 1, [self.batch_size, self.z_dim]).astype(np.float32)
                            generatedImages = self.sess.run(
                                self.GAN_gen2,
                                feed_dict={self.inputs: x_fixed, self.z: batch_z,
                                           self.gan_inputs: x_fixed})

                            generatedY = self.Predict_DomainLabels(generatedImages,myPrediction)

                            newY = np.zeros((32, 4))
                            newY[:, taskIndex] = 1
                            newX = np.concatenate((batch_images, generatedImages[0:32]), axis=0)
                            newY = np.concatenate((newY, generatedY[0:32]), axis=0)

                            # update D network
                            _, d_loss = self.sess.run([self.d_optim1, self.d_loss1],
                                                      feed_dict={self.inputs: newX,
                                                                 self.z: batch_z})

                            # update G and Q network
                            _, g_loss = self.sess.run(
                                [self.g_optim1, self.g_loss1],
                                feed_dict={self.inputs: newX, self.z: batch_z})

                            _, vaeLoss, _ = self.sess.run(
                                [self.vae_optim, self.vaeLoss, self.domain_optim],
                                feed_dict={self.inputs: newX, self.z: batch_z,
                                           self.gan_inputs: newX, self.gan_domain: newY})

                        counter += 1
                        print(
                            "Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f, vae_loss:%.8f. c_loss:%.8f" \
                            % (epoch, i, 0, time.time() - start_time, d_loss, g_loss, vaeLoss, 0))

                    outputs1 = self.sess.run(
                        self.GAN_gen1,
                        feed_dict={self.inputs: x_fixed, self.z: batch_z})

                    ims("results/" + "Celeba" + str(epoch) + ".jpg", merge2(outputs1[:64], [8, 8]))

            self.saver.save(self.sess, "models/TwoTeacherStudent_Unsupervised_Celeba")

infoMultiGAN = LifeLone_MNIST()
infoMultiGAN.build_model()
infoMultiGAN.train()
# infoMultiGAN.train_classifier()
#infoMultiGAN.test()
