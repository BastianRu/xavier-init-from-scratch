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

--
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
