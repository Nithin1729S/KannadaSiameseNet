One Shot Learning with Siamese Networks to recognize Kannada Alphabets
===================================================


1\. Introduction
================

Deep Convolutional Neural Networks have become the state of the art methods for image classification tasks. However, one of the biggest limitations is they require a lots of labelled data. In many applications, collecting this much data is sometimes not feasible. One Shot Learning aims to solve this problem.

2\. Classification vs One Shot Learning
=======================================

In case of **standard classification**, the input image is fed into a series of layers, and finally at the output we generate a probability distribution over all the classes (typically using a Softmax). For example, if we are trying to classify an image as cat or dog or horse or elephant, then for every input image, we generate 4 probabilities, indicating the probability of the image belonging to each of the 4 classes. Two important points must be noticed here. **First**, during the training process, we require a **large** number of images for each of the class (cats, dogs, horses and elephants). **Second**, if the network is trained only on the above 4 classes of images, then we cannot expect to test it on any other class, example “zebra”. If we want our model to classify the images of zebra as well, then we need to first get a lot of zebra images and then we must **re-train** the model again. There are applications wherein we neither have enough data for each class and the total number classes is huge as well as dynamically changing. Thus, the cost of data collection and periodical re-training is too high.

On the other hand, in a **one shot classification**, we require only one training example for each class. Yes you got that right, just one. Hence the name **One Shot**. Let’s try to understand with a real world practical example.

Assume that we want to build face recognition system for a small organization with only 10 employees (small numbers keep things simple). Using a traditional classification approach, we might come up with a system that looks as below:

![](https://miro.medium.com/v2/resize:fit:875/1*A49puFRGzvHjRJJBHTxryg.jpeg)

Standard classification using CNN

**Problems:**

a) To train such a system, we first require a lot of **different** images of each of the 10 persons in the organization which might not be feasible. (Imagine if you are doing this for an organization with thousands of employees).

b) What if a new person joins or leaves the organization? You need to take the pain of collecting data again and re-train the entire model again. This is practically not possible specially for large organizations where recruitment and attrition is happening almost every week.

Now let’s understand how do we approach this problem using one shot classification which helps to solve both of the above issues:

![](https://miro.medium.com/v2/resize:fit:875/1*g-561DsAfbU6gcVEk9AC4g.jpeg)

## One Shot Classification

Instead of directly classifying an input(test) image to one of the 10 people in the organization, this network instead takes an extra reference image of the person as input and will produce a similarity score denoting the chances that the two input images belong to the same person. Typically the similarity score is squished between 0 and 1 using a sigmoid function; wherein 0 denotes no similarity and 1 denotes full similarity. Any number between 0 and 1 is interpreted accordingly.

Notice that this network is not learning to classify an image directly to any of the output classes. Rather, it is learning a **similarity function**, which takes two images as input and expresses how similar they are.

How does this solve the two problems we discussed above?

a) In a short while we will see that to train this network, you do not require too many instances of a class and only few are enough to build a good model.

b) But the biggest advantage is that , let’s say in case of face recognition, we have a new employee who has joined the organization. Now in order for the network to detect his face, we only require a **single** image of his face which will be stored in the database. Using this as the reference image, the network will calculate the similarity for any new instance presented to it. Thus we say that network predicts the score in **one shot**.


4\. Omniglot Dataset
====================

For this project, we will use the Omniglot dataset which is a collection of 1623 hand drawn characters from 50 different alphabets. For every character there are just 20 examples, each drawn by a different person. Each image is a gray scale image of resolution 105x105.

Before I continue, I would like to clarify the difference between a character and an alphabet. In case of English the set A to Z is called as the alphabet while each of the letter A, B, etc. is called a character. Thus we say that the English alphabet contains 26 characters (or letters).

So I hope this clarifies the point when I say 1623 characters spanning over 50 different alphabets.

Let’s look at some images of characters from different alphabets to get a better feel of the dataset.

![](https://miro.medium.com/v2/resize:fit:875/1*GoAVSgNTIXeVbM4nA916HQ.jpeg)

Thus we have 1623 different classes(each character can be treated as a separate class) and for each class we have only 20 images. Clearly, if we try to solve this problem using the traditional image classification method then definitely we won’t be able to build a good generalized model. And with such less number of images available for each class, the model will easily overfit.

You can use the dataset from here [Kaggle](https://www.kaggle.com/datasets/watesoyan/omniglot) \
images\_background folder contains characters from 30 alphabets and will be used to train the model, while images\_evaluation folder contains characters from the other 20 alphabets which we will use to test our system.

Once you unzip the files, you will see below folders (alphabets) in the images\_background folder(used for training purpose):

![](https://miro.medium.com/v2/resize:fit:348/1*RlaIu4FZ6uczRGFhEx50_A.jpeg)

Contents of images\_background directory

And you will see below folders (alphabets) in the images\_evaluation folder (used for testing purpose):

![](https://miro.medium.com/v2/resize:fit:261/1*cASkYmQo2id1MLx6V4pECg.jpeg)

Contents of images\_evaluation directory

Notice that we will train the system on one set of characters and then test it on a completely different set of characters which were never used during the training. This is not possible in a traditional classification cycle.

5\. Model Architecture and Training
===================================

This code is an implementation of the methodology described in this [_research paper_](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf) by Gregory Koch et al_._ Model architecture and hyper-parameters that I have used are all as described in the paper.

Let’s first understand the architecture on a high level before diving into the details. Below I present an intuition of the architecture.

![](https://miro.medium.com/v2/resize:fit:1250/1*dFY5gx-Vze3micJ0AMVp0A.jpeg)

A high level architecture

**Intuition**: The term Siamese means twins. The two Convolutional Neural Networks shown above are not different networks but are two copies of the same network, hence the name Siamese Networks. Basically they share the same parameters. The two input images (x1 and x2) are passed through the ConvNet to generate a fixed length feature vector for each (h(x1) and h(x2)). Assuming the neural network model is trained properly, we can make the following hypothesis: If the two input images belong to the same character, then their feature vectors must also be similar, while if the two input images belong to the different characters, then their feature vectors will also be different. Thus the element-wise absolute difference between the two feature vectors must be very different in both the above cases. And hence the similarity score generated by the output sigmoid layer must also be different in these two cases. This is the central idea behind the Siamese Networks.

Given the above intuition let’s look at the picture of the architecture with more finer details taken from the research paper itself:

![](https://miro.medium.com/v2/resize:fit:3201/1*v40QXakPBOmiq4lCKbPu8w.png)

The below function is used to create the model architecture:

Siamese Model Architecture

Notice that there is no predefined layer in Keras to compute the absolute difference between two tensors. We do this using the Lambda layer in Keras which is used to add customized layers in Keras.

To understand the shape of the tensors passed at different layers, refer the below image generated using the plot\_model utility of Keras.

![](https://miro.medium.com/v2/resize:fit:875/1*RvqlZBlfOT9TcnEYhe_IQw.png)

Tensor shapes at every level

The model was compiled using the adam optimizer and binary cross entropy loss function as shown below. Learning rate was kept low as it was found that with high learning rate, the model took a lot of time to converge. However these parameters can well be tuned further to improve the present settings.

optimizer = Adam(lr = 0.00006)  
model.compile(loss="binary\_crossentropy",optimizer=optimizer)

The model was trained for 20000 iterations with batch size of 32.

After every 200 iterations, model validation was done using 20-way one shot learning and the accuracy was calculated over 250 trials. This concept is explained in the next section.

9\. Validating the Model
========================
