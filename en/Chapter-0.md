---
layout: default
title: Chapter 0
---

# Chapter 0: The problem
---

## *Vanishing/Exploding Gradients*: The Enemies of the multilayer perceptron

As I mentioned to you in the introduction, surely you have heard a lot about the problems of *Exploding Gradients* and *Vanishing Gradients*, which cause a reduction in the learning of the neural network and in the most serious cases, its death.

But, **what are they really and what causes them?**

Well, to answer this question we have to analyze the flow of data through the network, and understand the transformations they suffer, and how this affects learning, which will be our focus of analysis in the following chapters.

---

### Remembering the MLP architecture (Forward Pass)

#### Pre-activations

Good, we know that the fundamental operation executed in a perceptron is the following:

$$ y = w_1x_1 + w_2x_2 + w_3x_3 + \cdots + w_nx_n $$

> * For simplicity effects we will ignore the bias $b$ (*bias*). Later we will address a detailed explanation of why.
> * $y$ is also known as the pre-activation $z$.

Where $w_i$ is weight $i$ of the layer and $x_i$ input $i$ to the layer, which comes from the previous layer. Since each output of the previous layer is connected with each perceptron of the next layer, and each connection assumes a weight, if we have $n$ perceptrons in the previous layer, we will have $n$ weights for the next layer.

And the pre-activation $y$ is defined as the weighted sum of the weights $w_i$ of the layer and the inputs $x_i$ to the layer.

> Linear Algebra also gives us two other fantastic definitions:
> 1. "$y$ is equal to the **linear combination** of the inputs $x_1, x_2, ..., x_n$, where the weights $w_1, w_2, ..., w_n$ are the scalars". If you check the definition of linear combination, you will realize that it makes a lot of sense, and in fact, this definition gives us the answer to why we need activation functions in perceptrons, because without them, no matter how many layers we have, we would only be executing a linear operation through the whole network. If the phenomenon we try to predict in reality **is not linear**, our model will never be able to approximate it.
> 2. "$y$ is the **dot product** between the weight vector $W$ and the input vector $X$". If you group the weights $w_1, w_2, ..., w_n$ and the inputs $x_1, x_2, ..., x_n$ into vectors, when doing their dot product, you obtain exactly $y$. Since it is the most efficient, this is the way computers calculate it. And it is also the logic and notation used in real Deep Learning code.

Now then, from here on, we can also use the simplified summation notation, when convenient:

$$ y = \sum_{i = 1}^{n} w_ix_i $$

----
#### Activations

The next step in a perceptron after calculating the pre-activation, is to apply a non-linear function to the data, so that we break with the natural linearity of weighted sums, and we can give the model the freedom to transform the values of the data as it likes.

If $$f$$ is a non-linear function, the activation in a perceptron is defined as:

$$ a = f(w_1x_1 + w_2x_2 + w_3x_3 + \cdots + w_nx_n) $$

> In practice, we cannot use just any non-linear function. Imagine that we use the function $$x^2$$ as non-linearity for our network. But the phenomenon we want to model applies the transformation $$ x^3$$ to the data. Our model will never be able to approximate $$x^3$$ by means of squares of linear combinations since the final activations will be polynomials with terms raised to even powers $2, 4, 6, 8,...,2n $, and to obtain a value of the form $x^3$ you need at least one term of the form $x^3$ in your final activation.

Some of the most famous non-linear functions (and most used) to this day, because of their ability to break linearity and approximate other functions of any kind are:

* Sigmoid Function: $$ \sigma(x) = \frac{1}{1+e^{-x}} $$
* Hyperbolic Tangent: $$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}} $$
* ReLU Function (Rectified Linear Unit): $$ ReLU(x) = max(0, x)  $$

For this analysis, we will take as non-linearity the **Hyperbolic Tangent function $$tanh$$**.

Now. The activation step has a huge impact (almost decisive) on the learning process of the network, carried out by the back-propagation algorithm (*backpropagation*).

But before being able to talk about backward propagation, it is necessary to remember and introduce the notation that we will be using in the rest of the derivation.

### Some notation...

We have addressed how perceptrons perform the transformation of the data.
But to talk about a neural network, or more precisely, of a multilayer perceptron (*Multi-Layer Perceptron*) we must understand that this process is repeated through all the layers of the network, until reaching the prediction.

* We call the first layer of the network the **input layer** (*Input Layer*), which does not really contain perceptrons, but only the raw input data. One value for each "neuron" of the layer. In a lot of real code, you will find the input values grouped in an input vector $$X$$.

>In the first architectures of language models, if a word was encoded, for example, with a vector of 30 dimensions (that is, one value for each component), and if the sentence was, for example, 3 words, one had to use an input layer with 90 values.

* All the layers of the network that perform activation operations based on the outputs of the previous layer are called **hidden layers** or intermediate layers (*Hidden Layers*).

> An intuitive reason why the term "hidden" was added comes from the so-called **Interpretability Problem**. By getting rid of determinism and giving so much flexibility to the network to learn to predict all kinds of phenomena, we lose the ability to interpret the intermediate patterns or the real meanings of the values that neurons acquire in the intermediate layers. Nowadays, there is a branch of Machine Learning dedicated to this, called Explainable AI (XAI).

* The last intermediate layer is called the output layer (*Output Layer*), this layer is special because it is the final prediction of the whole network, depending on what training data we train the network with (and unlike hidden layers), the values of this layer will acquire one meaning or another.

> There is a very big debate about whether the last layer should be considered or not as the last intermediate layer, since, although it performs an activation operation, its values do have a meaning, unlike hidden layers. This depends on the context and many times does not affect the analysis.

The following example diagram lets us visualize the basic structure of a multilayer perceptron with an input layer of 6 values, two intermediate layers with 4 and 3 perceptrons respectively, and an output layer with a single output value:

<br>
<p align="center">
  <img src="https://raw.githubusercontent.com/ledell/sldm4-h2o/master/mlp_network.png" width="480" height="200" alt="Perceptron-Multi-layer">
</p>

<br>
You can notice two very important relations between layers:
1. **The number of outputs (activations) of a layer determines the number of inputs for each neuron of the next layer, or what is the same, the number of inputs to the next layer**.

> In the example diagram, the first layer has 6 values, which we can consider indistinctly inputs or outputs since no operation is applied, then, the first hidden layer receives 6 inputs in each neuron.

And consequently:

2. **The number of outputs of a layer determines the number of weights for each neuron of the next layer**

> Each neuron of the second layer receives 6 inputs, and since each connection from the first layer to the second implies a weight, each neuron in the second layer has 6 weights. In total, the second layer has $$4
 \text{ neurons}\cdot 6 \text{ weights } = 24 \text{ weights} $$.

Now. We will land all the concepts with concrete notation, to refer to them in a faster way:

* We will refer to a **pre-activation** in layer $L$ as: $z^{(L)}$
* We will refer to an **activation** in layer $L$ as: $a^{(L)}$

> $L$ is some layer of the network except the first one.

