from keras.datasets import mnist
import time
from utils import *
from scipy.misc import imsave as ims
from ops import *
from utils import *
import random as random
from glob import glob
import os,gzip
from Basic_structure import *

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

class LifeLone_MNIST(object):
    def __init__(self):
        self.batch_size = 64
        self.input_height = 28
        self.input_width = 28
        self.c_dim = 1
        self.z_dim = 100
        self.len_discrete_code = 10
        self.epoch = 10

        self.learning_rate = 0.0002
        self.beta1 = 0.5

        # MNIST dataset
        mnistName = "mnist"
        fashionMnistName = "Fashion"

        data_X, data_y = load_mnist(mnistName)
        x_train = data_X[0:60000]
        x_test = data_X[60000:70000]
        y_train = data_y[0:60000]
        y_test = data_y[60000:70000]

        self.mnist_train_x = x_train
        self.mnist_train_y = y_train
        self.mnist_test_x = x_test
        self.mnist_test_y = y_test

        data_X, data_y = load_mnist(fashionMnistName)

        x_train1 = data_X[0:60000]
        x_test1 = data_X[60000:70000]
        y_train1 = data_y[0:60000]
        y_test1 = data_y[60000:70000]

        self.mnistFashion_train_x = x_train1
        self.mnistFashion_train_y = y_train1
        self.mnistFashion_test_x = x_test1
        self.mnistFashion_test_y = y_test1

    def build_model(self):
        min_value = 1e-10
        # some parameters
        image_dims = [self.input_height, self.input_width, self.c_dim]
        bs = self.batch_size

        """ Graph Input """
        # images
        self.inputs = tf.placeholder(tf.float32, [bs] + image_dims, name='real_images')

        # noises
        self.z = tf.placeholder(tf.float32, [bs, self.z_dim], name='z')

        self.y = tf.placeholder(tf.float32, [bs, self.len_discrete_code])

        self.isPhase = 0

        z_mean, z_log_sigma_sq, out_logit, softmaxValue = Encoder_mnist(self.inputs,"encoder1", batch_size=64, reuse=False)
        discrete = out_logit
        discrete_softmax = tf.nn.softmax(discrete) + 1e-10
        log_y = tf.log(discrete_softmax)
        discrete_real = my_gumbel_softmax_sample(log_y, np.arange(self.len_discrete_code))
        continous_variables = z_mean + z_log_sigma_sq * tf.random_normal(tf.shape(z_mean), 0, 1, dtype=tf.float32)
        z1 = tf.concat((continous_variables,discrete_real),axis=1)

        z_mean2, z_log_sigma_sq2, out_logit2, softmaxValue2 = Encoder_mnist(self.inputs, "encoder2", batch_size=64,
                                                                         reuse=False)
        discrete2 = out_logit2
        discrete_softmax2 = tf.nn.softmax(discrete2) + 1e-10
        log_y2 = tf.log(discrete_softmax2)
        discrete_real2 = my_gumbel_softmax_sample(log_y2, np.arange(self.len_discrete_code))
        continous_variables2 = z_mean2 + z_log_sigma_sq2 * tf.random_normal(tf.shape(z_mean2), 0, 1, dtype=tf.float32)
        z2 = tf.concat((continous_variables2,discrete_real2),axis=1)

        reco1 = Generator_mnist( "generator1",z1, reuse=False)
        reco2 = Generator_mnist( "generator2",z2, reuse=False)

        #VAE loss
        reconstruction_loss1 = tf.reduce_mean(tf.reduce_sum(tf.square(reco1 - self.inputs), [1, 2, 3]))
        reconstruction_loss2 = tf.reduce_mean(tf.reduce_sum(tf.square(reco2 - self.inputs), [1, 2, 3]))

        KL_divergence1 = 0.5 * tf.reduce_sum(
            tf.square(z_mean) + tf.square(z_log_sigma_sq) - tf.log(1e-8 + tf.square(z_log_sigma_sq)) - 1, 1)
        KL_divergence1 = tf.reduce_mean(KL_divergence1)

        # KL divergence on gumble softmax
        KL_y1 = tf.reduce_sum(softmaxValue * (tf.log(softmaxValue + 1e-10) - tf.log(1.0 / 10.0)), 1)
        KL_y1 = tf.reduce_mean(KL_y1)

        KL_divergence2 = 0.5 * tf.reduce_sum(
            tf.square(z_mean2) + tf.square(z_log_sigma_sq2) - tf.log(1e-8 + tf.square(z_log_sigma_sq2)) - 1, 1)
        KL_divergence2 = tf.reduce_mean(KL_divergence2)

        # KL divergence on gumble softmax
        KL_y2 = tf.reduce_sum(softmaxValue2 * (tf.log(softmaxValue2 + 1e-10) - tf.log(1.0 / 10.0)), 1)
        KL_y2 = tf.reduce_mean(KL_y2)

        self.vae_loss1 = reconstruction_loss1 + KL_divergence1 + KL_y1*0.001
        self.vae_loss2 = reconstruction_loss2 + KL_divergence2 + KL_y2*0.001

        # get loss for generator
        self.classifier_loss1 = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=out_logit, labels=self.y))

        self.classifier_loss2 = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=out_logit2, labels=self.y))

        """ Training """
        # divide trainable variables into a group for D and a group for G
        T_vars = tf.trainable_variables()
        encoder_vars1 = [var for var in T_vars if var.name.startswith('encoder1')]
        encoder_vars2 = [var for var in T_vars if var.name.startswith('encoder2')]
        generator1_vars = [var for var in T_vars if var.name.startswith('generator1')]
        generator2_vars = [var for var in T_vars if var.name.startswith('generator2')]
        classifier1_vars = encoder_vars1 + generator1_vars
        classifier2_vars = encoder_vars2 + generator2_vars

        vae1_vars = encoder_vars1 + generator1_vars
        vae2_vars = encoder_vars2 + generator2_vars

        self.output1 = reco1
        self.output2 = reco2

        # optimizers
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.vae1_optim = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1) \
                .minimize(self.vae_loss1, var_list=vae1_vars)
            self.vae2_optim = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1) \
                .minimize(self.vae_loss2, var_list=vae2_vars)
            self.classifier1_optim = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1) \
                .minimize(self.classifier_loss1, var_list=classifier1_vars)
            self.classifier2_optim = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1) \
                .minimize(self.classifier_loss2, var_list=classifier2_vars)


        b1 = 0

    def predict(self):
        z_mean1, z_log_sigma_sq1, out_logit1, softmaxValue1 = Encoder_mnist(self.inputs, "encoder1", batch_size=64,
                                                                         reuse=True)

        z_mean2, z_log_sigma_sq2, out_logit2, softmaxValue2 = Encoder_mnist(self.inputs, "encoder2", batch_size=64,
                                                                            reuse=True)

        diff1 = tf.reduce_mean(tf.reduce_sum(tf.square(self.output1[0] - self.inputs[0])))
        diff2 = tf.reduce_mean(tf.reduce_sum(tf.square(self.output2[0] - self.inputs[0])))

        return softmaxValue1[0],softmaxValue2[0],diff1,diff2

    def test(self):
        with tf.Session() as sess:
            self.saver = tf.train.Saver()

            self.sess = sess
            sess.run(tf.global_variables_initializer())
            self.saver.restore(sess, 'models/LifeLone_MNIST')

            mnist_x_test = self.mnist_test_x
            mnist_y_test = self.mnist_test_y

            mnistFashion_x_test = self.mnistFashion_test_x
            mnistFashion_y_test = self.mnistFashion_test_y

            x_test = tf.concat((mnist_x_test,mnistFashion_x_test),axis=0)
            y_test = tf.concat((mnist_y_test,mnistFashion_y_test),axis=0)

            out1,out2,diff1,diff2 = self.predict()
            n_samples = np.shape(mnist_x_test)[0]
            k = n_samples

            totalLabels = []
            x_fixed = mnist_x_test[0:self.batch_size]

            for i in range(k):
                x1 = x_test[i*self.batch_size:(i+1)*self.batch_size]
                y1 = y_test[i*self.batch_size:(i+1)*self.batch_size]

                x_fixed[0] = mnist_x_test[0]

                label1,label2,d1,d2 = sess.run([out1,out2,diff1,diff2],feed_dict={self.inputs: x_fixed})

                if d1 < d2:
                    totalLabels.append(label1)
                else:
                    totalLabels.append(label2)

            x_fixed = mnist_x_test[0:64]
            x_fixed = np.reshape(x_fixed,(-1,28,28))
            ims("results/" + "Real" + str(1) + ".jpg", merge(x_fixed[:64], [8, 8]))

            y_prediction_labels = [np.argmax(one_hot)for one_hot in totalLabels]
            mnist_y_test_1 = [np.argmax(one_hot)for one_hot in mnist_y_test]
            mnist_y_test_1 = mnist_y_test_1[0:np.shape(y_prediction_labels)[0]]

            predictCount = 0
            for i in range(np.shape(y_prediction_labels)[0]):
                if mnist_y_test_1[i] == y_prediction_labels[i]:
                    predictCount = predictCount + 1

            accuracy = predictCount / np.shape(y_prediction_labels)[0]
            print(accuracy)
            bc = 0

    def train(self):

        config = tf.ConfigProto(allow_soft_placement=True)
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
        config.gpu_options.allow_growth = True

        isMnist = False
        with tf.Session(config=config) as sess:
            self.sess = sess
            sess.run(tf.global_variables_initializer())
            self.saver = tf.train.Saver()

            #self.saver.restore(sess, 'models/LifeLone_MNIST')

            # saver to save model
            self.saver = tf.train.Saver()

            counter = 0

            n_examples = np.shape(self.mnist_train_x)[0]

            start_epoch = 0
            start_batch_id = 0
            self.num_batches = int(n_examples / self.batch_size)

            # loop for epoch
            start_time = time.time()
            for epoch in range(start_epoch, self.epoch):
                count = 0
                # Random shuffling
                index = [i for i in range(n_examples)]
                random.shuffle(index)
                self.mnist_train_x = self.mnist_train_x[index]
                self.mnist_train_y = self.mnist_train_y[index]

                self.mnistFashion_train_x = self.mnistFashion_train_x[index]
                self.mnistFashion_train_y = self.mnistFashion_train_y[index]

                # get batch data
                for idx in range(start_batch_id, self.num_batches):
                    if isMnist:
                        batch_images = self.mnist_train_x[idx*self.batch_size:(idx+1)*self.batch_size]
                        batch_labels = self.mnist_train_y[idx*self.batch_size:(idx+1)*self.batch_size]
                    else:
                        batch_images = self.mnistFashion_train_x[idx * self.batch_size:(idx + 1) * self.batch_size]
                        batch_labels = self.mnistFashion_train_y[idx * self.batch_size:(idx + 1) * self.batch_size]

                    #if idx % 2 == 0:
                    # update VAE
                    if isMnist:
                        _,loss1 = self.sess.run([self.vae1_optim,self.vae_loss1],feed_dict={self.inputs: batch_images, self.y: batch_labels})
                        _,class_loss = self.sess.run([self.classifier1_optim,self.classifier_loss1],feed_dict={self.inputs: batch_images, self.y: batch_labels})

                        outputs = self.sess.run(
                            self.output1,
                            feed_dict={self.inputs: batch_images, self.y: batch_labels})

                    else:
                        _,loss1 = self.sess.run([self.vae2_optim,self.vae_loss2],feed_dict={self.inputs: batch_images, self.y: batch_labels})
                        _,class_loss = self.sess.run([self.classifier2_optim,self.classifier_loss2],feed_dict={self.inputs: batch_images, self.y: batch_labels})

                        outputs = self.sess.run(
                            self.output2,
                            feed_dict={self.inputs: batch_images, self.y: batch_labels})

                    # display training status
                    counter += 1
                    print("Epoch: [%2d] [%4d/%4d] time: %4.4f, vae_loss: %.8f, class_loss:%.8f" \
                          % (epoch, idx, self.num_batches, time.time() - start_time, loss1,class_loss))

                y_RPR = np.reshape(outputs, (-1, 28, 28))
                ims("results/" + "VAE" + str(epoch) + ".jpg", merge(y_RPR[:64], [8, 8]))

                self.saver.save(self.sess, "models/LifeLone_MNIST")


infoMultiGAN = LifeLone_MNIST()
infoMultiGAN.build_model()
infoMultiGAN.train()
infoMultiGAN.test()




