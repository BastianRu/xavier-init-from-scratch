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

#### What about the weights?

Well, because in the connection between neurons of consecutive layers there is an associated weight, it becomes necessary to introduce two more indices.

> I must warn you that from here on the notation becomes more robust, so it would not be strange at all if you need several re-readings to assimilate it.

* We denote a specific weight between some neuron of layer $L$ and some other neuron of the next layer $L+1$ as: $ w_{ij}  $

Where index $i$ refers to the i-th neuron of layer $L+1$ and $j$ the j-th neuron of layer $L$.

> Notice how this notation seems to be "backwards", because $i$ denotes neurons of the layer to which the propagation data **arrive**, while $j$ denotes neurons of the layer from which the propagation data **leave**. This is because most of the problems we face in neural networks reside in backpropagation, where the information flows backward, so when analyzing it, it is easier to invert the indices.

#### $L$?, $L+1$?, $L-1$?

I consider it necessary to stop briefly to reflect on the indices of the layers. Usually, we only analyze the flow of information between two adjacent layers, since this reasoning is recursive and we can apply it to as many pairs of consecutive layers as we want.

If $L$ is the layer we are reasoning about:

* $L+1$ is the layer immediately after $L$.
* $L-1$ is the layer immediately before $L$.

Since our analysis is limited to adjacent layers, we can choose to study the pair $(L, L+1)$ or $(L-1, L)$. In this article, we will use one and only one $(L, L+1)$, with $L$ being the layer before $L+1$.

Then, with all this in mind, we can re-write our definition of activation for a neuron $i$ of layer $L+1$ as:

$$ a_i^{(L+1)} = f(w_{i1}^{(L+1)}a_1^{(L)} + w_{i2}^{(L+1)}a_2^{(L)} + \cdots + w_{ij}^{(L+1)} a_{j}^{(L)})  $$

Where $j = 1, 2, 3, ..., n_{in}$ ($n_{in} = \text{number of inputs to each neuron of layer } L+1 $)

> Remember the relations between layers that I mentioned to you!.

It is evident how the notation becomes more and more cumbersome to read, that is why it is usually convenient to use summations:

$$ a_i^{(L+1)} = f( \sum_{j = 1}^{n_{in}} w_{ij}^{(L+1)} a_j^{(L)} ) $$

> Do not forget that the pre-activation still exists: $ z_i^{(L+1)} = \sum_{j = 1}^{n_{in}} w_{ij}^{(L+1)} a_j^{(L)} $, therefore, $ a_i^{(L+1)} = f( z_i^{(L+1)})$.

Good, with all this we are ready to go into backward propagation, which is where the explanation to our problems truly is.

---
### Remembering the MLP architecture (Backward Pass)

#### The loss function (*Loss Function*)

The literature proposes the need for a measure of how wrong the data predicted by our model are in relation to the real or expected data.
And this is where cost or loss functions come in. There are several types but basically all of them give a result of interest, a value that we need to minimize as much as possible, since if this value is close to 0, we will find a model that approximates the input data very well to the data that we consider as "real" or "true". This process is called optimization of the loss function.

> No matter how effective our training algorithm is, in practice, we will not always be able to make the loss function be 0. For example, in modern language models, which, based on a sequence of words predict the next word of the sentence, the training data are sequences of words together with the **possible** next word. But in language, a sequence of words can have many possible next words, which results in the training data having input sequences, which have as possible output data many different words. <br><br>
We would be communicating to the model that there are several valid "answers" to a single question. This is not always bad, when there are several possible options, **creativity** is born, although this term is more of generative models, I would prefer to say that our model is able to **generalize** over the data. <br><br>
> Ironically..., the impossibility of the error being 0, is what makes generative AI useful in the real world.

For this derivation, it will not be necessary to define a concrete loss function, since loss functions share enough characteristics in common that we can extract for the analysis:

* A loss function is a function of the final activation $a^{(L+1)}$ (network *output*) and of the expected value $y$ (Do not confuse it with the $y$ from forward!).
> I must clarify that there can exist MLPs in which the last layer has more than one neuron, that is, more than one final activation, for those cases the loss function is a function of the **mean** of the losses of each neuron with respect to the expected values $y_i$ separately, that is:
$$ \mathcal{L} = \frac{1}{m} \sum_{i=1}^{m} l(a_i^{(L+1)}, y_i)$$. To simplify notation in this article, we will assume $m=1$.
* A loss function is continuous and differentiable on the relevant domain.

Since the activations of a layer are functions of the weights of the layer and of the activations of the previous layer, and in turn, those activations are functions of the weights of the previous layer, and so on, the cost function ends up being a function of all the weights of the network, of the inputs of the network, and of the expected values.

Let us remember that **all** the weights through the network are trainable parameters, which means that their value is alterable, unlike the input values and the expected values, which are not. Therefore, we treat the cost function as a function whose only independent variables are the weights, since they are the only values that we have the power to alter to minimize the error.

If we group all the weights of a layer into a matrix that we will call $W^{(L)}$, where each row $i$ contains the weights of a neuron $i$.
Then we can group all the weights of the neural network in the following way:

$$ \theta = \mathrm{\{W^{(l)}}\}_{l=1}^{k}  $$

>Surely you did not know this notation, because it was created by the Deep Learning community to be practical.

Where $\theta$ is known as the set of parameters and groups all the weight matrices $W^{(L)}$ from layer $1, 2, 3, \cdots, k$. And $k$ represents the output layer of the network.

Then the loss $\mathcal{L}$ is a function of $(\theta)$: $\mathcal{L}(\theta)$.

#### The gradient vector

Good, it is clear that the cost function measures how "badly" our network predicts. But, how do we really minimize the loss?.

Vector calculus gives us the answer. If we have a function $f$ of several variables $x_1, x_2, x_3, \cdots, x_n$, the **gradient vector** of that function indicates the direction of maximum growth of the function. That is, what shape the changes in $x_1, x_2, x_3, \cdots, x_i$ have that produce the maximum rate of increase in the value of $f$.

The gradient vector of $f$ is defined as:

$$ \nabla f = \left[ \frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2}, \frac{\partial f}{\partial x_3}, \cdots, \frac{\partial f}{\partial x_n}  \right] $$

Where $\frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2}, \frac{\partial f}{\partial x_3}, \cdots, \frac{\partial f}{\partial x_n}$ are the **partial derivatives** of $f$ with respect to $x_1, x_2, x_3, \cdots, x_i$.

> Remember that, for the gradient vector, just like in normal derivatives, we only obtain a numerical value when we evaluate the function at a concrete point, that is, we assign values to $x_1, x_2, x_3, \cdots, x_n$. This is important because the direction of maximum growth of the function depends on the points where it is evaluated. Intuitively, imagine that you are climbing a mountain from the north, and a friend of yours is also climbing it but from the south, if you call your friend and ask in what direction the summit is, he will answer north, because from his perspective it is correct, however for you, the summit is toward the south. For the gradient vector exactly the same thing happens.

Now. Since $\mathcal{L}$ is a function of $\theta$, which is the set of all the weight matrices of the network. Therefore indeed, we can find the gradient vector for $\mathcal{L}$:

$$ \nabla \mathcal{L} = \left[  \frac{\partial \mathcal{L} }{\partial w_1}, \frac{\partial \mathcal{L}}{\partial w_2}, \frac{\partial \mathcal{L}}{\partial w_3}, \cdots, \frac{\partial \mathcal{L}}{\partial w_N}  \right]  $$

With $N$ as the **total** number of weights in the network.

>Each partial derivative of this vector tells us how the loss function $\mathcal{L}$ changes with respect to each specific weight. For example $\frac{\partial \mathcal{L}}{\partial w_1}$ tells us how "sensitive" $\mathcal{L}$ is with respect to $w_1$, it answers the question: If I modify the value of $w_1$ a little, how much does the loss change?.

Where $N$ is the total number of parameters in the network.

But, notice how there is something strange here, the gradient vector will indicate the direction (that is how to modify the value of the weights) to obtain the maximum growth of the function, that is to increase the loss function, but this is precisely the opposite of what we want!. What we seek is to optimize the loss function, that is to minimize it, not to increase it. Well, because of that vector calculus comes once again to the rescue and gives us the solution:

**If $\nabla f$ indicates the direction of maximum growth of $f$, $-\nabla f$ gives us the direction of maximum descent of $f.$**

Interpolating, $-\nabla \mathcal{L}$ tells us how we must modify the values of the weights to reduce our loss function.

> * Seeing it algebraically, if we could write the loss function as a giant equation where the constants are the input values to the network plus the expected output values, and the unknowns were the weights, we would be trying to find all the values of those weights such that by replacing them in the equation, it becomes 0. And I say trying, because you already know that in practice, we try to minimize that number as much as possible.
> * A while ago I mentioned to you one of the intuitive reasons why in some models it is possible that an absolute value of 0 is never reached for the loss function. Well, with the gradient as a tool, there is another justification for it. If you remember differential calculus, you will know that a function can have several **local minimum values**, which are points where the function seems to decrease until a certain point, and from there, it starts to grow again, giving the false impression that this is the absolute minimum value that the function can take in its whole domain. The consequence of this in our system is that our gradient vector will guide us to the nearest local minimum that exists, but it is not a guarantee that this is the **global** or absolute minimum of the function, and once we arrive at that point (or very close to it), since the negative gradient only points to the direction of maximum descent, and we can no longer descend more, the training will die at that point. In high dimensions this problem is even more active.

#### The chain rule

Good, to obtain our gradient vector, we must calculate the **partial derivative** of the loss function with respect to **every** weight in $\theta$.

For a partial derivative of a function of, for example two variables, $f(x, y)$, if $x$ and $y$ are directly the independent variables of $f$, that is that there are no other functions on which $x$ and $y$ depend, we can calculate their partial derivatives $\frac{\partial f}{\partial x}$ and $\frac{\partial f}{\partial y}$ directly.

However, we must remember that in this architecture, the weights (above all those of the first layers) do not influence $\mathcal{L}$ directly but do so through several functions which they do affect directly (the pre-activation and the activation).

So, analogically and taking again our example $f(x, y)$. If $x$ is not an independent but a function of another variable, for example, $x = g(a)$, and the same for $y$, and $y = h(b)$. Then now, if $a$ and $b$ are arbitrary and we are interested in knowing $\frac{\partial f}{\partial a}$ and $\frac{\partial f}{\partial b}$, the thing changes.

This is where the chain rule (*Chain Rule*) comes in, which indicates the way to obtain those derivatives:

Let $f(x, y)$ where $x = g(a)$ and $y = h(b)$, then:

$$ \frac{\partial f}{\partial a} = \frac{\partial f}{\partial x} \cdot \frac{\partial x}{\partial a} $$

$$ \frac{\partial f}{\partial b} = \frac{\partial f}{\partial y} \cdot \frac{\partial y}{\partial b} $$

> * I love this rule because it is very intuitive!. Notice how, if we want to find the derivative of $f$ with respect to $a$, we go from the biggest to the smallest, first we find $f$ with respect to $x$ and then $x$ with respect to $a$. Simple!.
> * In literature, documentation, code, and so on, you may find the terms of the derivatives in another order, because multiplication is commutative.

This rule applies for as many relations between functions and variables as we have.

Good, with this tool. We can go into the calculation of the derivatives of the gradient vector.

With our activation defined as:

$$ a_i^{(L+1)} = f(z_i^{(L+1)}) $$

$$ z_i^{(L+1)} = w_{i1}^{(L+1)}a_1^{(L)} + w_{i2}^{(L+1)}a_2^{(L)} + \cdots + w_{ij}^{(L+1)} a_{j}^{(L)} $$

> Before continuing reading, observe and analyze the dependency: $\mathcal{L}$ depends on $a_i^{(L+1)}$, $a_i^{(L+1)}$ depends on $z_i^{(L+1)}$, $z_i^{(L+1)}$ depends on $w_{i1}^{(L+1)}a_1^{(L)} + w_{i2}^{(L+1)}a_2^{(L)} + \cdots + w_{ij}^{(L+1)} a_{j}^{(L)}$. In your mind, visualize each product $w_{ij}^{(L+1)}a_j^{(L)}$ as a single variable. What we are going to find is the derivative of $\mathcal{L}$ with respect to one of those variables.

The partial derivative of the loss function $\mathcal{L}$ with respect to a certain weight $w_{ij}$ of layer $L$ is:

$$ \frac{\partial \mathcal{L} }{\partial w_{ij}^{L+1}} = \frac{\partial \mathcal{L} }{\partial a_i^{L+1}} \cdot \frac{\partial a_i^{L+1} }{\partial z_i^{L+1} } \cdot \frac{\partial z_i^{L+1} }{\partial w_{ij}^{L+1} }  $$

Now, there is something that I omitted when talking to you about the loss function. In reality, models are trained based on a training dataset. If we want our model to be able to predict correctly regardless of the diversity of the input data, we need to train it by also showing it diverse examples, as many as necessary to cover the great majority of possible cases of the phenomenon. Mathematically, this is achieved by optimizing over the average of all the losses of each one of the training examples.

So if we want to obtain the "true" gradient vector of our model, for each partial derivative with respect to a weight, we have to obtain the average over all the partial derivatives of each one of the examples separately. That is:

$$ \frac{\partial \mathcal{L}_T }{\partial w_{ij}^{L+1}} = \frac{1}{N} \sum_{k=1}^{N} \frac{\partial \mathcal{L_k} }{\partial w_{ij}^{L+1}}  $$

Where $N$ is the total number of examples in our training dataset.

Our total gradient vector would look like this:

$$ \nabla \mathcal{L} = \begin{bmatrix} \frac{\partial \mathcal{L}_T }{\partial w_{1}} = \frac{1}{N} \sum_{k=1}^{N} \frac{\partial \mathcal{L_k} }{\partial w_{1}} \\ \frac{\partial \mathcal{L}_T }{\partial w_{2}} = \frac{1}{N} \sum_{k=1}^{N} \frac{\partial \mathcal{L_k} }{\partial w_{2}}\\ \vdots \\ \frac{\partial \mathcal{L}_T }{\partial w_{n}} = \frac{1}{N} \sum_{k=1}^{N} \frac{\partial \mathcal{L_k} }{\partial w_{n}} \end{bmatrix} $$

$n$ is the total number of weights in the network.

However, in the *backprop* process of all training examples, the gradients are calculated exactly the same, that is why for this analysis we will take $N = 1$.

> * In reality, this is not the whole backpropagation algorithm. Normally, after calculating the gradient of the network, we would perform a **gradient descent** (Gradient Descent), by means of which we modify sequentially the value of the weights based on the information of the gradient vector. Usually we do not use the whole magnitude of the gradient but a part of it, this factor is known as the learning rate $ \alpha $, (**in general** $ 0 \lt \alpha \leqslant 1$), this is because the complete gradient is too "heavy", especially at the beginning of training. The fact that it points to the direction of maximum descent does not mean that its magnitude is the correct one to arrive directly at the minimum point, sometimes it can be much larger than required, causing $\mathcal{L}$ to oscillate around the minimum, or very small, causing it to advance very slowly. $\alpha$ is a **hyperparameter**, which means that it is chosen depending on the context, it is not exactly arbitrary, only that it is adjusted depending on the type of problem we are dealing with.
> * Have you asked yourself why it is said that a neural network is a **NON-deterministic** system?. It is strange, if you think well, if all the training examples are known, in theory, we could calculate each training step, from start to finish. This contradicts the assumption. For non-determinism to exist, at some point in the system we would have to be unable to know what its next state is, and then? <br><br>
> I told you that the "true" gradient is calculated over the average of all the losses of each training example individually, assuming that we use **all** the data of the training dataset. In practice, we almost never use the "real" gradient, basically, because it has more disadvantages than advantages. First, computationally it is unthinkable, the hardware and time required for this calculation are monstrous, they grow significantly with the size of the network and the size of the training dataset. We could take weeks for a single training iteration. And second but not less important, sometimes using the "absolute" direction toward the minimum is not the most viable. <br><br>
> And this is where **stochastic gradient descent** (SGD) comes in. SGD proposes: Why instead of using all the training examples at the same time to calculate an "absolute" gradient, do we not use only a small **random** group of them for each iteration?. This group is known as a **mini-batch**, and of course, you could think that by calculating the gradient on a subset of the complete batch, we would not get close to the minimum, or that we would get close very slowly. But, it turns out that it is not like that!. Experimentally it has been proven that this works just as well and even better than the full gradient. By using an "approximate" gradient, noise is introduced into the process that on occasions helps the gradient reach better minima, or avoid getting stuck in bad minima. Not to mention that computation for SGD is much friendlier and possible. Then, we sacrifice determinism and choose to have a less exact gradient, in exchange for greater speed, and some beneficial noise. Since the training examples for each iteration are chosen randomly, the process becomes stochastic.

If you have reached here and you have all the concepts up to date. We are now ready to go into the problem without worrying about notation discordances. Let's do it!

---
### The terms of the gradient of a weight

Good, we already know how the data flow through the network, forward and backward. In the jargon of the field these flows are better known as **inference** (forward) and **training** (backward), which are intimately related because one depends on the other in a sequential relation. The inference of one step affects the training of the next, and the training of the next affects the inference of the next one, and so on.

But it is worth focusing particularly on the training algorithm, back-propagation, which makes possible the "learning of the network", as I mentioned to you.

But to be able to understand its behavior, we must know how the terms of the gradient relate to each other. And it is that, they have some problematic things that we must consider

> You could think that these problems are derived from entropy. But as we will see, it is not so, they do not occur by chance, but by the very nature of the functions we choose. As the Merovingian from the Matrix saga says: **"Where some see coincidence, I see consequence"**.

With our derivative of the loss with respect to a specific weight:

$$ \frac{\partial \mathcal{L} }{\partial w_{ij}^{L+1}} = \frac{\partial \mathcal{L} }{\partial a_i^{L+1}} \cdot \frac{\partial a_i^{L+1} }{\partial z_i^{L+1} } \cdot \frac{\partial z_i^{L+1} }{\partial w_{ij}^{L+1} }  $$

The derivative $\frac{\partial \mathcal{L} }{\partial a_i^{L+1}}$ is calculated depending on two things, first, on whether $L+1$ is or is not the last layer, and second, on the specific loss function, even so, because all loss functions are functions of the final activations, we can analyze their behavior, but we will leave it for later sections, for now we will focus on the last two terms of this expression. In $\frac{\partial z_j^{L+1} }{\partial w_{ij}^{L+1}}$ as the weight is the independent variable, and the previous activation a constant, then:

$$ \frac{\partial z_i^{L+1} }{\partial w_{ij}^{L+1}} = a_j^{L} $$

And in $ \frac{\partial a_i^{L+1} }{\partial z_i^{L+1} } $, if $f = \tanh$, the derivative of the activation with respect to the pre-activation is (by chain rule) simply the derivative of the tangent evaluated at that same pre-activation. That is:

$$ \frac{\partial a_i^{L+1} }{\partial z_i^{L+1} } = f'(z_i^{L+1}) = \tanh'(z_i^{L+1}) $$

Notice how for both cases, $\frac{\partial z_i^{L+1} }{\partial w_{ij}^{L+1}}$ and $ \frac{\partial a_i^{L+1} }{\partial z_i^{L+1} }$ end up depending directly on the previous activations and the pre-activation of the current layer. Also, observe how $a_j^{L}$ is actually $\tanh(z_j^{L})$ (a pre-activation in L), which means that in the end, the activation function determines absolutely all the behavior of the flow of information through the network, both in forward propagation and in backward propagation.

> For this reason it is very important to choose the activation function properly for each model. And as we will see later, there exist proposed solutions that solve the problem only for some activations, not for all.

As we have shown, all the attention is centered on the activation function, for our case, the hyperbolic tangent, and its derivative. So let us investigate its curious behavior.

> You can try to apply the same analysis to the other activation functions, you will realize how the great majority suffer from problems similar to the ones we will show.

The graph of $\tanh$ looks something like this:

<br>
<p align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/8/87/Hyperbolic_Tangent.svg" width="480" height="200" alt="Perceptron-Multi-layer">
</p>

This function "compresses" the input values to values between -1 and 1. Theoretically 1 and -1 are only reached when $x \to \pm\infty$, but computationally, it is enough that the inputs move a little away from the origin for rounding to bring the values close to 1 and -1.

For example, theoretically:

$$ \tanh(2) \approx 0.96402758007 $$

**PyTorch**, the library par excellence for *Machine Learning* development, has the function implemented, ready to use:

```python
torch.tensor(2).tanh()
# Output: tensor(0.9640)
```

> in PyTorch, **tensors** are the fundamental structure, all values are represented through tensors. A tensor can represent scalars, vectors and matrices of any dimension.

Notice how only **4 significant digits** are used. This has a lot of impact. Let us also see other examples:

* $$ \tanh(5) \approx 0.99990920426 $$

```python
torch.tensor(5).tanh()
# Output: tensor(0.9999)
```
> Since the fifth significant digit is 0, the rounding leaves the number at 0.9999.

* $$\tanh(6) \approx 0.99998771165 $$

```python
torch.tensor(6).tanh()
# Output: tensor(1.0000)
```

> However here, the fifth digit is 8, so the rounding approximates the number.

In reality, these examples are only conceptual, so that you understand that there exists a limit in the capacity of machines to represent irrational numbers (like those produced by $\tanh$). The default **display** precision of PyTorch is 4 decimals, but it can be adjusted at discretion. The real limit is in the data type handled by tensors, which is `float32` (32 bits), which means that we would only have approximately 7 digits of precision for the activation values:

* ```python
  torch.set_printoptions(precision=10)
  torch.tensor(9).tanh()
  # Output: tensor(0.9999999404)
  ```
And

* ```python
  torch.set_printoptions(precision=10)
  torch.tensor(10).tanh()
  # Output: tensor(1.)
  ```

As shown, adjusting visual precision lets us see the digits that PyTorch normally hides, helping us differentiate values from absolute 1 or -1. However, this **is purely visual**: for an input of 10, the difference with 1 is so small that float32 runs out of capacity to represent it.

Therefore, pre-activations with values of approximately 10 or more (or -10), translate into activations purely of 1 (or -1). In practice, this occurs often depending on the initialization of the weights. If these initial values are not controlled, the pre-activations tend to reach these extremes easily and even greater values in networks with many layers, we will see it in detail later, but for now, we will call "large" pre-activations values of approximately 10 or more, or approximately -10 or less.

Let us remember the terms:

* $$ \frac{\partial z_i^{L+1} }{\partial w_{ij}^{L+1}} = \tanh(z_j^{L}) $$

>Remember that: $ a_j^L = \tanh(z_j^L)$

* $$ \frac{\partial a_i^{L+1} }{\partial z_i^{L+1} }  = \tanh'(z_i^{L+1})  $$

From the analyzed we can draw two conclusions:

1. If the pre-activations $ z_j^{L} $ acquire large values, the terms $ \frac{\partial z_i^{L+1} }{\partial w_{ij}^{L+1}}$ will be approximately 1 or -1. This means that if the pre-activations are very large, in the backward propagation process, at most, the only thing they will contribute will be a sign change to the gradient of the weights!. And the learning that a binary change contributes is quite precarious in many cases.

> Careful, it does not always contribute only a sign change, this is only if the pre-activations are large enough, if they are not. It contributes a factor between 1 and -1 that multiplies the rest of the gradient.

2. **The important one**: $$\tanh'(x) = 1 - \tanh^2(x)$$
. If $ x \to \pm\infty $, then $\tanh'(x) \to 0$. But as we already saw, computationally, if the pre-activations $z_i^{L+1}$ tend to be large, the terms $ \frac{\partial a_i^{L+1} }{\partial z_i^{L+1} }$ will tend to 0!.

Let us formalize these two rules for the extreme cases:

   * If $ z_j^{L} \to \pm\infty$, then, $ \frac{\partial z_i^{L+1} }{\partial w_{ij}^{L+1}} \to \pm1 $.

   * If $z_i^{L+1} \to \pm\infty$, then $\frac{\partial a_i^{L+1} }     {\partial z_i^{L+1} } \to 0$.

Excellent, we have analyzed the behavior of two of the three terms of the gradient for a weight of the network. But we still lack the most important one.

#### Gradient propagation

As we saw in previous sections, the gradient of a weight also depends on a third term. The derivative of the loss with respect to the activation of the present layer:

$$ \frac{\partial \mathcal{L} }{\partial a_i^{L+1}} $$

As I mentioned to you previously, this term is calculated based on the chosen cost function. Since in this analysis we have represented $\mathcal{L}$ in a general way, it will not be necessary to know the derivative. However, there is an important fact to know about this term:

$$ \frac{\partial \mathcal{L} }{\partial a_i^{L+1}} \not\equiv  \frac{\partial \mathcal{L} }{\partial a_j^{L}} $$

If $L+1$ is the last layer of the network, the derivative of the loss with respect to an activation in the same layer, is **structurally different** from the derivative of the loss with respect to an activation in the previous layer.

> The symbol $\not \equiv$ means Non-Identity. If we used $\not=$ we would be saying that the values of both derivatives are not equal, which well you might think is true, because they are calculated differently, but it may happen that they end up coinciding in values, then the statement is contradicted. With $\not\equiv$ we indicate that the **nature of both expressions is different**, even if they come to acquire similar values.

Since in the last layer of the network the loss function is a function of the final activations of the network, the derivative only depends on those last activations and therefore is obtained directly by applying differentiation rules. Nevertheless, if it is required to know the same derivative for one of the activations in the intermediate layers, as concerns us in this article, the calculation changes.

In other words, the derivative $\frac{\partial \mathcal{L} }{\partial a_i^{L+1}}$ is calculated differently depending on whether $L+1$ is the last layer of the network, or not.

As we did during the *Forward Pass*, we will continue reasoning about the pair of layers $(L,L+1)$. To maintain this convention, we will analyze the derivative of the loss with respect to the activations of layer $L$:

$$ \frac{\partial \mathcal{L} }{\partial a_j^{L}} $$

Here we will assume that $L$ represents an intermediate layer of the network. Consequently, the layer $L+1$ can be another intermediate layer or even the output layer.

> We could have decided to study the derivative $\frac{\partial \mathcal{L} }{\partial a_i^{L+1}}$. However, this would force us to impose that $L+1$ is not the output layer, since in that case the expression would be immediate. <br><br>
Instead by working with $\frac{\partial \mathcal{L} }{\partial a_j^{L}} $ we can keep our convention intact. Besides, this reflects better the behavior of backward propagation, where the flow is from back to front:
> > Forward Pass:         $L  \to  L+1$ <br>
> > Backward Pass:       $L  \gets  L+1$

Good, with this assumption in mind, $\frac{\partial \mathcal{L} }{\partial a_j^{L}}$ is special because here resides the so-called concept of gradient propagation (*gradient propagation*). And it is that, the nature of the chain rule makes the calculation of gradients recursive. This property allows reusing intermediate calculations when propagating the gradient backward in the network, which enormously reduces the computational cost of training.

---
**The term** $\delta$

Before moving on to the calculation of that expression, I consider it prudent to introduce another notation with which we are going to work and that you will very probably find in related material.

As we already saw $ \frac{\partial \mathcal{L} }{\partial a_i^{L+1}} $ tells us that the loss is a function of the activation $a_i^{L+1}$. And $a_i^{L+1}$ is a function of $z_i^{L+1}$. This means that we can find the derivative of the loss with respect to $z_i^{L+1}$, so let us apply the chain rule again:

$$ \frac{\partial \mathcal{L}}{\partial z_i^{L+1} } = \frac{\partial \mathcal{L} }{\partial a_i^{L+1}} \cdot  \frac{\partial a_i^{L+1}}{\partial z_i^{L+1} } $$

This term is particular because it will appear recursively (as we will see shortly). Therefore, in the literature, the following special notation was given to it:

$$\delta_i^{L+1} \equiv \frac{\partial \mathcal{L}}{\partial z_i^{L+1} }  $$

Now, notice how $\frac{\partial \mathcal{L} }{\partial a_i^{L+1}}$ and $\frac{\partial a_i^{L+1}}{\partial z_i^{L+1} }$ appear in $\frac{\partial \mathcal{L} }{\partial w_{ij}^{L+1}}$. So we can substitute expressions:

$$ \frac{\partial \mathcal{L} }{\partial w_{ij}^{L+1}} = \delta_i^{L+1} \cdot \frac{\partial z_i^{L+1} }{\partial w_{ij}^{L+1} } $$

We already saw that $\frac{\partial z_i^{L+1} }{\partial w_{ij}^{L+1} } = a_j^L $, then:

$$ \frac{\partial \mathcal{L} }{\partial w_{ij}^{L+1}} = \delta_i^{L+1} \cdot a_j^L $$

> Here is another definition of the gradient of a weight with which you could run into in your learning.

---

Let us highlight an important fact inside forward propagation:

Because of how pre-activation is defined, a single activation $a_j$ in layer $L$ **will influence all** the activations $a_i$ that exist in layer $L+1$. Which means that any change in one $a_j^L$ will affect each and every one of the activations $a_i^{L+1}$

Since $a_i^{L+1}$ = $f(z_i^{L+1})$ [$f=\tanh]$, then a single $a_j^L$ influences all the pre-activations of the next layer:

$$ z_1^{L+1}, z_2^{L+1}, z_3^{L+1}, \cdots, z_n^{L+1} $$

Said in another way, all the pre-activations $z_i$ of layer $L+1$ are functions (among other variables) of the activation $a_j$ of layer $L$, formally written:

$$ z_1^{L+1}(a_1^L, \cdots, a_j^L , \cdots, a_m^L) $$ <br>
$$ z_2^{L+1}(a_1^L, \cdots, a_j^L , \cdots, a_m^L)$$  <br>
$$ z_3^{L+1}(a_1^L, \cdots, a_j^L , \cdots, a_m^L)$$ <br>
$$ \vdots $$  <br>
$$ z_n^{L+1}(a_1^L, \cdots, a_j^L , \cdots, a_m^L) $$

Where $m$ is the number of neurons in layer $L$.

> If you are already fluent with the Forward Pass, maybe the previous expressions seem obvious to you, but to be able to understand the next step, it is necessary to specifically highlight that $a_j^L$ is present in all $z_i^{L+1}$.

Finally, the loss function is a function (**indirectly**) of all the pre-activations $ z_1^{L+1}, z_2^{L+1}, z_3^{L+1}, \cdots, z_n^{L+1} $:

$$ \mathcal{L}(z_1^{L+1}, z_2^{L+1}, z_3^{L+1}, \cdots, z_n^{L+1}) $$

> The loss function is a function of the final activations but these also depend on previous activations and pre-activations, regardless of whether $L+1$ is the last layer or not.

**The total chain rule**

Previously we had already seen how the chain rule allowed us to calculate derivatives of functions that had intermediate variables that depended in turn on other variables. Usually it is only called chain rule, plainly, but its full name is **multivariable chain rule**, and it is like that because it is used only in multivariable functions.

However, in this case, things are somewhat different. We are supposed to be trying to find the term $\frac{\partial \mathcal{L} }{\partial a_j^{L}}$, that is how much $\mathcal{L}$ changes when $a_j^L$ changes. But, how do we calculate it, if a single change in $a_j^{L}$ causes a change in all the $z_i^{L+1}$ which in turn **all** cause a change in $\mathcal{L}$?

As always, calculus gets us out of trouble. Unlike the multivariable chain rule, the **total chain rule** helps us differentiate cases like this, where a function of interest, in this case $\mathcal{L}$ is affected by a variable (for the case $a_j^{L}$) through several paths, or several intermediate functions.

Let $g=f(a, b)$, $a=f_1(x)$ and $b=f_2(x)$, the derivative of $g$ with respect to $x$ is obtained in the following way:

$$ \frac{ dg }{ dx } = \frac{ \partial g }{ \partial a } \cdot \frac{ \partial a }{ \partial x } + \frac{ \partial g }{ \partial b } \cdot \frac{ \partial b }{ \partial x }   $$

> * This chain rule is also very intuitive!. We go from the biggest to the smallest.

Each addend in the expression is known as a **contribution** (similar to the contributions in the integral), a name that makes a lot of sense, since each one contributes its own information of how $x$ produces changes in $g$.

Translating this logic to where we are interested, taking into account the dependencies mentioned before:

$$ \frac{ \partial \mathcal{L} }{\partial a_j^L} = \frac{ \partial \mathcal{L} }{\partial z_1^{L+1}} \cdot \frac{ \partial z_1^{L+1} }{\partial a_j^L} + \frac{ \partial \mathcal{L} }{\partial z_2^{L+1}} \cdot \frac{ \partial z_2^{L+1} }{\partial a_j^L} + \cdots + \frac{ \partial \mathcal{L} }{\partial z_n^{L+1}} \cdot \frac{ \partial z_n^{L+1} }{\partial a_j^L}  $$

Since the structure of the derivative goes through all the pre-activations of layer $L+1$, then the expression is simplifiable by summation:

$$ \frac{ \partial \mathcal{L} }{\partial a_j^L} = \sum_{i=1}^{n_{L+1}} \frac{ \partial \mathcal{L} }{\partial z_i^{L+1}} \cdot \frac{ \partial z_i^{L+1} }{\partial a_j^L}  $$

Notice how, using partial differentiation rules, $ \frac{ \partial z_i^{L+1} }{\partial a_j^L} = w_{ij}^{L+1}$ (similar to what happens with $\frac{\partial z_i^{L+1}}{\partial w_{ij}^{L+1}}). $ Also, as we had anticipated, the term $\delta_i^{L+1}$ appears, so we can substitute:

$$ \frac{ \partial \mathcal{L} }{\partial a_j^L} = \sum_{i=1}^{n_{L+1}} \delta_i^{L+1} \cdot w_{ij}^{L+1} $$

From the section where I talked to you about the term $\delta$, if $\delta_i^{L+1} = \frac{\partial \mathcal{L} }{\partial a_i^{L+1}} \cdot  \frac{\partial a_i^{L+1}}{\partial z_i^{L+1} }$, then, the same expression for layer $L$ should be $ \delta_j^{L} = \frac{\partial \mathcal{L} }{\partial a_j^{L}} \cdot  \frac{\partial a_j^{L}}{\partial z_j^{L} } $, and since we just obtained the first term:

$$ \delta_j^{L} = \left( \sum_{i=1}^{n_{L+1}} \delta_i^{L+1} \cdot w_{ij}^{L+1}  \right) \cdot f'(z_j^L)$$

> Remember how the derivative of the activation with respect to a pre-activation (think of it as a numeric value) is simply the derivative of the activation evaluated at that value, as we had already seen.

And finally, we have arrived at one of the most beautiful expressions of neural networks. You can see how we arrived at an expression where $\delta$ appears recursively, and it does so from forward to backward, resulting in that, the gradients of later layers influence the gradients of earlier layers, or said in another way, to calculate a gradient in layer $L$, the gradients of layer $L+1, L+2, \cdots, L_k$ are required, for these reasons, backward propagation is called as it is called.

---
### Activations and dead neurons

With all the tools that we have acquired through this introduction, we are finally ready to pose the problem with the formality it demands.

Our gradient of a weight (with the change in the layer notation that we declared) is:

$$ \frac{\partial \mathcal{L} }{\partial w_{ij}^{L}} = \frac{\partial \mathcal{L} }{\partial a_i^{L}} \cdot \frac{\partial a_i^{L} }{\partial z_i^{L} } \cdot \frac{\partial z_i^{L} }{\partial w_{ij}^{L} }  $$

We have derived each and every one of its terms:

$$ \frac{\partial \mathcal{L} }{\partial w_{ij}^{L}} = \delta_i^{L} \cdot  a_j^{L-1}$$

> We had said that we were only going to work with the pair $(L, L+1)$, but in this case mentioning $L-1$ is inevitable if we want to preserve the reasoning in the following chapters.

$$ \frac{\partial \mathcal{L} }{\partial w_{ij}^{L}} = \left( \sum_{i=1}^{n_{L+1}} \delta_i^{L+1} \cdot w_{ij}^{L+1}  \right) \cdot f'(z_j^L) \cdot a_j^{L-1} $$

In this way, we have managed to reduce all the unknowns to expressions whose behavior we can analyze, because they are in function of the data that we do know. We can make the following reasonings:

1. It is necessary to take care of the values in which the previous activations oscillate, because they directly influence the gradient of the next layer:

- If $a_j^{L-1} \to 0$, then $\frac{\partial \mathcal{L} }{\partial w_{ij}^{L}} \to 0$.

2. It is necessary to take care of the initialization and values of the weights in the network, especially in training, since the gradient is proportional to the accumulated product of the weights through the layers:

$$ \frac{\partial \mathcal{L} }{\partial w_{ij}^{L}} \propto \prod_{m=L+1}^{L_f} W^m$$

> Remember that we defined $W^l$ as a notation exclusive to this science. Besides, $L_f$ is the output layer.

* So that the smaller the values of the weights in a layer, the smaller the magnitude of the gradient will be, and vice versa.

3. It is necessary to take care of the values that the later activations take, since the gradient is proportional to the accumulated product of the derivatives of the activations of the neurons through the whole network:

$$\frac{\partial \mathcal{L} }{\partial w_{ij}^{L}} \propto \prod_{m=L}^{L_f} f'(z^{m})$$

> We do not include any subindex in $z^m$ so as not to generalize the indices $i,j$ through the whole network, because we are talking about an accumulated product that takes into account more than two layers.

* So that the smaller the values of the derivatives of the activations, the smaller our gradients will be, and vice versa.

If the values of the derivatives of the activations, the previous activations and the weights of a layer reduce with each step through the layers, then $ \frac{\partial \mathcal{L} }{\partial w_{ij}^{L}} \to 0$ (which is also known as "neuron saturation"). If the gradient is 0, it means that the value of the weight will not be modified and therefore, **no learning will be produced in the perceptron**, no matter if the loss function still indicates that there exists a high error.

This phenomenon is called **vanishing gradients** (*vanishing gradients*).

If the values of the derivatives of the activations, the previous activations and the weights of a layer increase with each step through the layers, then $ \frac{\partial \mathcal{L} }{\partial w_{ij}^{L}} \to \infty$. If the gradient is very large, the value of the weight will be modified massively, causing the loss function to move away from the minimum, with a greater distance in each training iteration, so in this case, in the long run, **no learning will be produced in the perceptron either**.

This phenomenon is called **exploding gradients** (*exploding gradients*).

> Think again about the mountain example. In the vanishing case, imagine that with each step you take, each time you advance less and less, until the point where you stop advancing, you never reach the summit. On the other hand, in the opposite case, with each step you take, each time you advance a greater distance, and at some point, you advance so much, that you pass the summit, and when you try to return, you overshoot the summit again, but you end up at a point even farther than the first time, you end up being incapable of reaching the summit.

If we take the 3 previous statements, and collapse them into a single expression that formally describes the problem:

$$\frac{\partial \mathcal{L} }{\partial w_{ij}^{L}}\propto  \left( \prod_{m=L+1}^{L_f} W^m \right)  \left( \prod_{m=L}^{L_f} f'(z^{m}) \right) a_j^{L-1} $$

Although this expression should not be interpreted as an exact equality in scalar terms, it reflects the fundamental structure of the gradient: its dependence on accumulated products of weights and activation derivatives through the layers.

But, how to face this problem?, Can we then alter the weights to solve it?, How do normal distributions influence? and what about "energy preservation"?.

We are going to answer these and more questions in the next chapter, where we will introduce **statistics** as a formal tool to address the problem. Besides, we will see how our objective will be to achieve a "preservation" of the ranges of values in which the weights can move, so that we avoid the problems of vanishing or gradient explosion as much as possible, making it possible to achieve a minimum desired training in the network.
