# A Journey into the Core of Xavier Initialization
[Read in English](README.md) | [Leer en Español](es/README.md)
---
## Introduction
Have you ever wondered why, when building a neural network, we can't simply initialize all weights to 1, or 0, or even giant random values as a starting point? After all, they’ll eventually tune themselves through training, right?

If you've spent any time with Deep Learning, you’ve likely heard about the "Vanishing Gradient" or "Exploding Gradient" problems. These phenomena cause your network to "die" prematurely: gradients vanish or neurons saturate, leaving the model unable to learn even the simplest patterns.

This document is more than just a theoretical explanation; it is a derivation from scratch. My goal is to walk you through these pages of logical and statistical reasoning that led Xavier Glorot and Yoshua Bengio to propose their famous initialization technique in 2010. We will not accept any formula as "given by divine intervention" we will build every single one of them using the fundamental tools of mathematics.

---
### What will we cover?

We will explore the flow of a neural network from the perspective of variance preservation. The study is divided into three main blocks:

  1. The Foundations: Intuitive definitions of random variables and why variance is our best tool for measuring a signal's "health."

  2. The Forward Pass: How to prevent information from vanishing or exploding as it travels from input to prediction.

  3. The Backward Pass: The hidden symmetry in error calculation and how to ensure the gradient reaches the early layers of the network alive.

---
### Is this for you? (Prerequisites)

I have designed this derivation to be as accessible as possible. However, to follow the thread of the proof without getting lost, you should have the following minimum prior knowledge:

**For the Forward Pass analysis:**
- MLP Architecture: Understanding neurons, layers, and their connections (the basic dot product $z = Wx + b$ for pre-activations).

- Descriptive Statistics: Feeling comfortable with the concepts of Mean (average) and Variance/Standard Deviation (data dispersion).
  
- Basic Algebra: Solving equations, powers, and square roots.

**For the Backward Pass analysis:**

- Differential Calculus: Understanding what a derivative is and, especially, the Chain Rule.

- Backpropagation: A basic notion of how error flows backward through a network. (Don't worry about complex gradient notation; we will cover an introduction to it).

- Partial Derivatives: Understanding that we can derive with respect to one variable while keeping others constant.

If you meet these requirements, we will build the rest here. We will define every new statistical property right before we need it for the next step of the proof. If you already master these concepts, feel free to skip directly to the sections of interest.


[Chapter 0: The problem](https://bastianru.github.io/xavier-init-from-scratch/en/Chapter-0)