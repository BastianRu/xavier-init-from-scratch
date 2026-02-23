---
layout: default
title: Capitulo 0
---

# Capitulo 0: El problema 
---

## *Vanishing/Exploding Gradients*: Los Enemigos del perceptron multicapa

Como te mencione en la introduccion, seguramente has oido mucho de los problemas de "gradientes que explotan" y "desvanecimiento de gradientes", en ingles, *Exploding Gradients* y *Vanishing Gradients*, que causan una reduccion del aprendizaje de la red neuronal y en los casos mas graves, la muerte del mismo. 

Pero, **¿que son realmente y que los causa?**

Bueno, para responder a esta pregunta tenemos que analizar el flujo de los datos a traves de la red, y entender las transformaciones que sufren, y como esto afecta en el aprendizaje, que sera nuestro foco de analisis en los siguientes capitulos. 

---
### Recordando la arquitectura MLP (Forward Pass)

#### Preactivaciones 

Bien, sabemos que la operacion fundamental que se ejecuta en un perceptron es la siguiente:

$$ y = w_1x_1 + w_2x_2 + w_3x_3 + \cdots + w_nx_n $$

> * Para efectos de simplicidad ignoraremos el sesgo $b$ (*bias*). Mas adelante abordaremos una explicacion detallada del porque.
> * $y$ tambien se conoce como la pre-activacion $z$.
 

Donde $w_i$ es el peso $i$ de la capa y $x_i$ la entrada $i$ a la capa, que viene de la anterior capa. Como cada salida de la capa anterior esta conectada con cada perceptron de la capa posterior, y cada conexion supone un peso, si tenemos $n$ perceptrones en la capa anterior, tendremos $n$ pesos para la capa posterior.

Y la preactivacion $y$ se define como la suma ponderada de los pesos $w_i$ de la capa y las entradas $x_i$ a la capa. 

> El Algebra Lineal tambien nos aporta otras dos definiciones fantasticas:
> 1. "$y $ es igual la **combinacion lineal** de las entradas $x_1, x_2, ..., x_n$, donde los pesos $w_1, w_2, ..., w_n$ son los escalares". Si revisas la definicion de combinacion lineal, te daras cuenta de que tiene mucho sentido, y de hecho, esta definicion nos da la respuesta a por qué necesitamos funciones de activacion en los perceptrones, ya que de no tenerlas, sin importar cuantas capas tengamos, solo estariamos efecutando una operacion lineal a traves de toda la red. Si el fenomeno que intentamos predecir en la realidad **no es lineal**, nuestro modelo jamas podra aproximarlo. 
> 2. "$y$ es el **producto punto** entre el vector de pesos $W$ y el vector de entradas $X$". Si agrupas los pesos $w_1, w_2, ..., w_n$ y las entradas $x_1, x_2, ..., x_n$ en vectores, al realizar su producto punto, obtendras exactamente $y$. Al ser la mas eficiente, esta es la forma en la que las computadoras la calculan. Y tambien es la logica y notacion que se usa en el codigo de Deep Learning real. 

Ahora bien, de aqui en adelante, podemos tambien usar la notacion simplificada con sumatoria, cuando convenga:

$$ y = \sum_{i = 1}^{n} w_ix_i $$

----
#### Activaciones

El siguiente paso en un perceptron despues de calcular la preactivacion, es aplicar una funcion no lineal a los datos, de modo que rompamos con la linealidad natural de las sumas ponderadas, y podamos darle al modelo la libertad de transformar los valores de los datos a gusto.

Si $$f$$ es una funcion no lineal, la activacion en un perceptron se define como:

$$ a = f(w_1x_1 + w_2x_2 + w_3x_3 + \cdots + w_nx_n) $$ 

> En la practica, no podemos usar cualquier funcion no lineal. Imagina que usamos la funcion $$x^2$$ como no-linealidad para nuestra red. Pero el fenomeno que queremos modelar aplica la transformacion $$ x^3$$ a los datos. Nuestro modelo jamas podra aproximar $$x^3$$ por medio de cuadrados de combinaciones lineales dado que las activaciones finales seran polinomios con terminos elevados a potencias pares $2, 4, 6, 8,...,2n $, y para obtener un valor de la forma $x^3$ necesitas al menos un  termino de la forma $x^3$ en tu activacion final.

Algunas de las funciones no lineales mas famosas (y mas usadas) hasta la fecha, por su capacidad de romper la linealidad y aproximar otras funciones de cualquier tipo son:

* Funcion Sigmoide: $$ \sigma(x) = \frac{1}{1+e^{-x}} $$
* Tangente Hiperbolica: $$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}} $$
* Funcion ReLU (Rectified Linear Unit): $$ ReLU(x) = max(0, x)  $$

Para este analisis, tomaremos como no-linealidad la funcion **Tangente Hiperbolica $$tanh$$**.

Ahora. El paso de la activacion tiene un impacto enorme (casi decisivo) en el proceso de aprendizaje de la red, llevado a cabo por el algoritmo de retro-propagacion (*backpropagation*).

Pero antes de poder hablar de propagacion hacia atras, es necesario recordar e introducir la notacion que estaremos usando el resto de la derivacion.

### Algo de notacion...

Hemos abordado como los perceptrones efectuan la transformacion de los datos. 
Pero para hablar de una red neuronal, o mas precisamente, de un perceptron multicapa (*Multi-Layer Perceptron*) debemos entender que este proceso se repite a lo largo de todas las capas de la red, hasta llegar a la prediccion. 

* A la primera capa de la red, la llamamos **capa de entrada** (*Input Layer*), que no contiene realmente perceptrones, si no, solo los datos de entrada, en bruto. Un valor por cada "neurona" de la capa. En mucho codigo real, encontraras los valores de entrada agrupados en un vector de entrada $$X$$.

>En las primeras arquitecturas de modelos de lenguaje, si una palabra se codificaba, por ejemplo, con un vector de 30 dimensiones (es decir, un valor para cada componente), y si la frase era de, por ejemplo, 3 palabras, habia que usar una capa de entrada con 90 valores.

* A todas las capas de la red que efectuen operaciones de activacion con base a las salidas (*outputs*) de la anterior capa, se les llama **capas ocultas** o intermedias (*Hidden Layers*).

> Una razon intuitiva de por que se les añadio el termino "oculto" viene del llamado **Problema de la Interpretabilidad**. Al deshacerse del determinismo y otorgarle tanta flexibilidad a la red para aprender a predecir todo tipo de fenomenos, perdemos la capacidad de interpretar los patrones intermedios o los significados reales de los valores que adquieren las neuronas en las capas intermedias. Hoy en dia, existe una rama del Machine Learning dedicada a esto, llamada Explainable AI (XAI).

* A la ultima capa intermedia, se le llama capa de salida (*Output Layer*), esta capa es especial porque es la prediccion final de toda la red, segun con que datos de entrenamiento entrenemos la red (y a diferencia de las capas ocultas),  los valores de esta capa adquiriran un significado u otro. 

> Hay un debate muy grande sobre si la ultima capa deberia ser considerada o no como la ultima capa intermedia, dado que, aunque efectua una operacion de activacion, sus valores si poseen un significado, no como las capas ocultas. Esto depende del contexto y muchas veces no afecta el analisis.  

El siguiente diagrama de ejemplo nos permite visualizar la estructura basica de un perceptron multicapa con una capa de entrada de 6 valores, dos capas intermedias con 4 y 3 perceptrones respectivamente, y una capa de salida con un unico valor de salida:
        
<br>
<p align="center">
  <img src="https://raw.githubusercontent.com/ledell/sldm4-h2o/master/mlp_network.png" width="480" height="200" alt="Perceptron-Multi-capa">
</p>

<br>
Puedes notar dos relaciones muy importante entre capas:
1.  **El numero de outputs (activaciones) de una capa, determina el numero de inputs para cada neurona de la siguiente capa, o lo que es lo mismo, el numero de inputs a la siguiente capa**.

> En el diagrama de ejemplo, la primera capa tiene 6 valores, que podemos considerar indistintamente inputs u outputs dado que no se aplica ninguna operacion, entonces, la primera capa oculta, recibe 6 inputs en cada neurona.

Y por consecuente:

2. **El numero de outpus de una capa, determina el numero de pesos para cada neurona de la siguiente capa**

> Cada neurona de la segunda capa, recibe 6 inputs, y como cada conexion de la primera capa hacia la segunda implica un peso, cada neurona en la segunda capa tiene 6 pesos. En total, la segunda capa tiene $$4 
 \text{ neuronas}\cdot 6 \text{ pesos } = 24 \text{ pesos} $$.

Ahora. Aterrizaremos todos los conceptos con notacion concreta, para referirnos a ellos de una forma mas rapida:
 
* Nos referiremos a una **pre-activacion** en la capa $L$ como: $z^{(L)}$
* Nos referiremos a una **activacion** en la capa $L$ como: $a^{(L)}$

> $L$ es alguna capa de la red exceptuando la primera.

#### ¿Que hay de los pesos?

Bueno, debido a que en la conexion entre neuronas de capas consecutivas existe un peso asociado, se vuelve necesario introducir dos indices mas.

> Debo advertirte que a partir de aqui la notacion se vuelve mas robusta, asi que no resultaria raro en asboluto si necesitas varias re-lecturas para asimilarla.

* A un peso especifico entre alguna neurona de la capa $L$ y alguna otra neurona de la capa siguiente $L+1$ lo denotamos como: $ w_{ij}  $

Donde el indice $i$ refiere la i-ésima neurona de la capa $L+1$ y $j$ la j-ésima neurona de la capa $L$.

> Nota como esta notacion parece estar "al reves", porque $i$ denota neuronas de la capa a la que **llegan** los datos de la propagacion, mientras que $j$ denota neuronas de la capa de la que **salen** los datos de la propagacion. Esto se debe a que la mayoria de los problemas que enfrentamos en las redes neuronales residen en la retropropagacion, donde la informacion fluye hacia atras, por lo que a la hora de analizarla, resulta mas sencillo invertir los indices.

#### ¿$L$?, ¿$L+1$?, ¿$L-1$? 

Considero necesario detenernos brevemente a reflexionar sobre los índices de las capas. Usualmente, solo analizamos el flujo de la información entre dos capas adyacentes, dado que este razonamiento es recursivo y lo podemos aplicar a cuantos pares de capas consecutivas queramos.

Si $L$ es la capa sobre la que estamos razonando:

* $L+1$ es la capa inmediatamente posterior a $L$.
* $L-1$ es la capa inmediatamente anterior a $L$.

Dado que nuestro análisis se limita a capas adyacentes, podemos elegir estudiar la pareja $(L, L+1)$ o bien $(L-1, L)$. En este articulo, utilizaremos unica y exclusivamente $(L, L+1)$, siendo $L$ la capa anterior a $L+1$.

Entonces, con todo esto en mente, podemos re-escribir nuestra definicion de activacion para una neurona $i$ de la capa $L+1$ como:

$$ a_i^{(L+1)} = f(w_{i1}^{(L+1)}a_1^{(L)} + w_{i2}^{(L+1)}a_2^{(L)} + \cdots + w_{ij}^{(L+1)} a_{j}^{(L)})  $$

Donde $j = 1, 2, 3, ..., n_{in}$ ($n_{in} = \text{numero de entradas a cada neurona de la capa } L+1 $)

> Recuerda las relaciones entre capas que te mencione!.

Es evidente como la notacion se vuelve cada vez mas engorrosa de leer, por eso, usualmente conviene usar sumatorias:

$$ a_i^{(L+1)} = f( \sum_{j = 1}^{n_{in}} w_{ij}^{(L+1)} a_j^{(L)} ) $$

> No olvides que la pre-activacion sigue existiendo: $ z_i^{(L+1)} = \sum_{j = 1}^{n_{in}} w_{ij}^{(L+1)} a_j^{(L)} $, por lo que, $ a_i^{(L+1)} = f( z_i^{(L+1)})$.

Bien, con todo esto estamos listos para adentrarnos en la propagacion hacia atras, que es donde verdaderamente se encuentra la explicacion a nuestros problemas. 

---
### Recordando la arquitectura MLP (Backward Pass)

#### La funcion de perdida (*Loss Function*)

La literatura plantea la necesidad de una medida de que tan erroneos son los datos predichos por nuestro modelo con relacion a los datos reales o esperados.
Y aqui es donde entran las funciones de coste o de perdida. Las hay de varios tipos pero basicamente todas entregan un resultado de interes, un valor que precisamos minimizar tanto como sea posible, ya que si este valor es cercano a 0, nos encontraremos con un modelo que aproxima muy bien los datos de entrada a los datos que nosotros consideramos como "reales" o "verdaderos". A este proceso se le llama, optimizacion de la funcion de perdida.

> Por muy eficaz que sea nuestro algoritmo de entrenamiento, en la practica, no siempre podremos lograr que la funcion de perdida sea 0. Por ejemplo, en modelos de lenguaje modernos, que, en base a una secuencia de palabras predicen la siguiente palabra de la frase, los datos de entrenamiento son secuencias de palabras junto con la **posible** siguente palabra. Pero en el lenguaje, una secuencia de palabras puede tener muchas posibles siguientes palabras, lo que resulta en que en los datos de entrenamiento exitan secuencias de entrada, que tienen como posible dato de salida muchas palabras diferentes. <br><br>
Le estariamos comunicando al modelo que hay varias "respuestas" validas a una sola pregunta. Esto no siempre es malo, al haber varias opciones posibles, nace la **creatividad**, aunque este termino es mas bien de modelos generativos, preferiria decir que nuestro modelo es capaz de **generalizar** sobre los datos. <br><br>
> Ironicamente...,la imposibilidad de que el error sea 0, es lo que hace que la IA generativa sea util en el mundo real.

Para esta derivacion, no sera necesario definir una funcion de perdida concreta, siendo que, las funciones de perdida comparten suficientes caracteristicas en comun que podemos extraer para el analisis:

* Una funcion de perdida es funcion de la activacion final $a^{(L+1)}$ (*output* de la red) y del valor esperado $y$ (No lo confundas con el $y$ del forward!).
>Debo aclarar que pueden existir MLPs en las que la ultima capa tenga mas de una neurona, es decir, mas de una activacion final, para esos casos la funcion de perdida es funcion de la **media** de las perdidas de cada neurona con respecto a los valores esperados $y_i$ por separado, es decir:
$$ L= \frac{1}{m} \sum_{i=1}^{m} l(a_i^{(L+1)}, y_i)$$. Para simplificar la notacion en este aritculo, asumiremos $m=1$.
* Una funcion de perdida es continua y diferenciable en el dominio relevante.

Como las activaciones de una capa son funciones de los pesos de la capa y de las activaciones de la capa anterior, y a su vez, esas activaciones son funciones de los pesos de la capa anterior, y asi sucesivamente, la funcion de costo termina siendo funcion de todos los pesos de la red, de las entradas de la red, y de los valores esperados.

Recordemos que **todos** los pesos a lo largo de la red son parametros entrenables, lo que quiere decir que su valor es alterable, a diferencia de los valores de entrada y los valores esperados, que no lo son. Por lo tanto, tratamos a la función de costo como una funcion cuyas unicas variables independientes son los pesos, ya que son los unicos valores que tenemos el poder de alterar para minimizar el error.

Si agrupamos todos los pesos de una capa en una matriz que llamaremos $W^{(L)}$, en donde cada columna $i$ contenga los pesos de una neurona $i$. 
Entonces podemos agrupar todos los pesos de la red neuronal de la siguiente manera:

$$ \theta = \mathrm{\{W^{(l)}}\}_{l=1}^{k}  $$

>Seguramente no conocias esta notacion, porque fue creada por la comunidad de Deep Learning para ser práctica.

Donde $\theta$ se conoce como el conjunto de parmetros y agrupa todas las matrices de pesos $W^{(L)}$ de la capa $1, 2, 3, \cdots, k$. Y $k$ representa la capa de salida de la red.

Entonces $L$ es funcion de $(\theta)$: $L(\theta)$.


#### El vector gradiente

Bien, esta claro que la funcion de costo mide que tan "mal" predice nuestra red. Pero, ¿y como minimizamos realmente la perdida?.

El calculo vectorial nos aporta la respuesta. Si tenemos una funcion $f$ de varias variables $x_1, x_2, x_3, \cdots, x_n$, el **vector gradiente** de esa funcion nos indica la direccion de maximo crecimiento de la funcion. Es decir, que forma tienen los cambios en $x_1, x_2, x_3, \cdots, x_i$ que producen la maxima taza de aumento en el valor de $f$. 

El vector gradiente de $f$ se define como:

$$ \nabla f = \left[ \frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2}, \frac{\partial f}{\partial x_3}, \cdots, \frac{\partial f}{\partial x_n}  \right] $$

Donde $\frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2}, \frac{\partial f}{\partial x_3}, \cdots, \frac{\partial f}{\partial x_n}$ son las **derivadas parciales** de $f$ con respecto a $x_1, x_2, x_3, \cdots, x_i$.

> Recuerda que, para el vector gradiente, al igual que en las derivadas normales, solo obtenemos un valor numerico cuando evaluamos la funcion en un punto concreto, es decir, asignamos valores a $x_1, x_2, x_3, \cdots, x_n$. Esto es importante porque la direccion de maximo crecimiento de la funcion depende de los puntos en los que este evaluada. Intuitivamente, imagina que estas subiendo una montaña desde el norte, y un amigo tuyo esta subiendola tambien pero desde el sur, si llamas a tu amigo y le preguntas hacia que direccion esta la cima, te respondera que norte, porque desde su perspectiva es correcto, sin embargo para ti, la cima se encuentra hacia el sur. Para el vector gradiente ocurre exactamente lo mismo.

Trabajo en progreso...































