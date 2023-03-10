# Lifelong Dual Generative Adversarial Nets Learning in Tandem

>📋 This is the implementation of Lifelong Dual Generative Adversarial Nets Learning in Tandem

# Title : Lifelong Dual Generative Adversarial Nets Learning in Tandem

# Abstract

Continually capturing novel concepts without forgetting is one of the most critical functions sought for in artificial intelligence systems. However, modern deep learning models are far from such capabilities. Even the most advanced deep learning networks are prone to quickly forgetting previously learnt knowledge after training with new data. The proposed Lifelong Dual Generative Adversarial Networks (LD-GANs) consists of two GAN networks, namely a Teacher and an Assistant teaching each other in tandem while successively learning a series of tasks. A single Discriminator is used to decide the realism of generated images by the dual GANs. A new training algorithm, called the Lifelong Self Knowledge Distillation (LSKD) is proposed for training LD-GANs when learning each new task during the lifelong learning. LSKD enables the transfer of knowledge from a more knowledgeable player for training another generator jointly with learning the information from a newly given data set within an adversarial playing game setting. In contrast to other lifelong learning models, LD-GANs is memory efficient and does not require freezing model's parameters after learning each given task. Furthermore, we extend the LD-GANs to a Teacher-Student network for assimilating data representations across several domains acquired during the lifelong learning. Experimental results indicate a better performance for the proposed framework in unsupervised lifelong representation learning when compared to other methods.


# Environment

1. Tensorflow 2.1
2. Python 3.6

# Training and evaluation

>📋 Python xxx.py, the model will be automatically trained and then report the results after the training.

# BibTex
>📋 If you use our code, please cite our paper as:


