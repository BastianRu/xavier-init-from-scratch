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

> Por muy eficaz que sea nuestro algoritmo de entrenamiento, en la practica, no siempre podremos lograr que la funcion de perdida sea 0. Por ejemplo, en modelos de lenguaje modernos, que, en base a una secuencia de palabras predicen la siguiente palabra de la frase, los datos de entrenamiento son secuencias de palabras junto con la **posible** siguente palabra. Pero en el lenguaje, una secuencia de palabras puede tener muchas posibles siguientes palabras, lo que resulta en que en los datos de entrenamiento existan secuencias de entrada, que tienen como posible dato de salida muchas palabras diferentes. <br><br>
Le estariamos comunicando al modelo que hay varias "respuestas" validas a una sola pregunta. Esto no siempre es malo, al haber varias opciones posibles, nace la **creatividad**, aunque este termino es mas bien de modelos generativos, preferiria decir que nuestro modelo es capaz de **generalizar** sobre los datos. <br><br>
> Ironicamente...,la imposibilidad de que el error sea 0, es lo que hace que la IA generativa sea util en el mundo real.

Para esta derivacion, no sera necesario definir una funcion de perdida concreta, siendo que, las funciones de perdida comparten suficientes caracteristicas en comun que podemos extraer para el analisis:

* Una funcion de perdida es funcion de la activacion final $a^{(L+1)}$ (*output* de la red) y del valor esperado $y$ (No lo confundas con el $y$ del forward!).
>Debo aclarar que pueden existir MLPs en las que la ultima capa tenga mas de una neurona, es decir, mas de una activacion final, para esos casos la funcion de perdida es funcion de la **media** de las perdidas de cada neurona con respecto a los valores esperados $y_i$ por separado, es decir:
$$ \mathcal{L} = \frac{1}{m} \sum_{i=1}^{m} l(a_i^{(L+1)}, y_i)$$. Para simplificar la notacion en este aritculo, asumiremos $m=1$.
* Una funcion de perdida es continua y diferenciable en el dominio relevante.

Como las activaciones de una capa son funciones de los pesos de la capa y de las activaciones de la capa anterior, y a su vez, esas activaciones son funciones de los pesos de la capa anterior, y asi sucesivamente, la funcion de costo termina siendo funcion de todos los pesos de la red, de las entradas de la red, y de los valores esperados.

Recordemos que **todos** los pesos a lo largo de la red son parametros entrenables, lo que quiere decir que su valor es alterable, a diferencia de los valores de entrada y los valores esperados, que no lo son. Por lo tanto, tratamos a la función de costo como una funcion cuyas unicas variables independientes son los pesos, ya que son los unicos valores que tenemos el poder de alterar para minimizar el error.

Si agrupamos todos los pesos de una capa en una matriz que llamaremos $W^{(L)}$, en donde cada columna $i$ contenga los pesos de una neurona $i$. 
Entonces podemos agrupar todos los pesos de la red neuronal de la siguiente manera:

$$ \theta = \mathrm{\{W^{(l)}}\}_{l=1}^{k}  $$

>Seguramente no conocias esta notacion, porque fue creada por la comunidad de Deep Learning para ser práctica.

Donde $\theta$ se conoce como el conjunto de parmetros y agrupa todas las matrices de pesos $W^{(L)}$ de la capa $1, 2, 3, \cdots, k$. Y $k$ representa la capa de salida de la red.

Entonces la perdida $\mathcal{L}$ es funcion de $(\theta)$: $\mathcal{L}(\theta)$.


#### El vector gradiente

Bien, esta claro que la funcion de costo mide que tan "mal" predice nuestra red. Pero, ¿y como minimizamos realmente la perdida?.

El calculo vectorial nos aporta la respuesta. Si tenemos una funcion $f$ de varias variables $x_1, x_2, x_3, \cdots, x_n$, el **vector gradiente** de esa funcion nos indica la direccion de maximo crecimiento de la funcion. Es decir, que forma tienen los cambios en $x_1, x_2, x_3, \cdots, x_i$ que producen la maxima taza de aumento en el valor de $f$. 

El vector gradiente de $f$ se define como:

$$ \nabla f = \left[ \frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2}, \frac{\partial f}{\partial x_3}, \cdots, \frac{\partial f}{\partial x_n}  \right] $$

Donde $\frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2}, \frac{\partial f}{\partial x_3}, \cdots, \frac{\partial f}{\partial x_n}$ son las **derivadas parciales** de $f$ con respecto a $x_1, x_2, x_3, \cdots, x_i$.

> Recuerda que, para el vector gradiente, al igual que en las derivadas normales, solo obtenemos un valor numerico cuando evaluamos la funcion en un punto concreto, es decir, asignamos valores a $x_1, x_2, x_3, \cdots, x_n$. Esto es importante porque la direccion de maximo crecimiento de la funcion depende de los puntos en los que este evaluada. Intuitivamente, imagina que estas subiendo una montaña desde el norte, y un amigo tuyo esta subiendola tambien pero desde el sur, si llamas a tu amigo y le preguntas hacia que direccion esta la cima, te respondera que norte, porque desde su perspectiva es correcto, sin embargo para ti, la cima se encuentra hacia el sur. Para el vector gradiente ocurre exactamente lo mismo.

Ahora. Como $\mathcal{L}$ es funcion de $\theta$, que es conjunto de todas las matrices de pesos de la red. Por lo que en efecto, podemos encontrar el vector gradiente para $\mathcal{L}$:

$$ \nabla \mathcal{L} = \left[  \frac{\partial \mathcal{L} }{\partial w_1}, \frac{\partial \mathcal{L}}{\partial w_2}, \frac{\partial \mathcal{L}}{\partial w_3}, \cdots, \frac{\partial \mathcal{L}}{\partial w_N}  \right]  $$

Con $N$ como el numero **total** de pesos en la red.

>Cada derivada parcial de este vector nos indica como cambia la funcion de perdida $\mathcal{L}$ con respecto a cada peso en especifico. Por ejemplo $\frac{\partial \mathcal{L}}{\partial w_1}$ nos indica que tan "sensible" es $\mathcal{L}$ con respecto a $w_1$, responde a la pregunta: ¿Si modifico un poco el valor de $w_1$, que tanto cambia la perdida?.

Donde $N$ es el numero de parametros en total de la red.

Pero, nota como hay algo extraño aqui, el vector gradiente nos indicara la direccion (es decir como modificar el valor de los pesos) para obtener el maximo crecimiento de la funcion, es decir aumentar la funcion de perdida, pero esto es precisamente lo contrario a lo que queremos!. Lo que buscamos es optimizar la funcion de perdida, es decir minimizarla, no aumentarla. Bueno, por ello el calculo vectorial viene otra vez al rescate y nos da la solucion:

**Si $\nabla f$ nos indica la direccion de maximo crecimiento de $f$, $-\nabla f$ nos da la direccion de maximo descenso de $f.$**

Interpolando, $-\nabla \mathcal{L}$ nos indica como debemos modificar los valores de los pesos para reducir nuestra funcion de perdida.

> * Viendolo algebraicamente, si pudieramos escribir la funcion de perdida como una ecuacion gigante en donde las constantes son los valores de entrada a la red mas los valores de salida esperados, y las incognitas fueran los pesos, estariamos tratando de encontrar todos los valores de esos pesos tales que al reemplazarlos en la ecuacion, sea 0. Y digo tratando, porque ya sabes que en la practica, tratamos de minimizar ese numero tanto como sea posible. 
> * Hace rato te mencione una de las razones intuitivas por las cuales en algunos modelos es posible que jamas se alcance un valor de 0 absoluto para la funcion de perdida. Bueno, con el gradiente como herramienta, hay otra justificacion para ello. Si recuerdas calculo diferencial, sabras que una funcion puede tener varios **valores minimos locales**, que son puntos en donde la funcion parece decrecer hasta cierto punto, y a partir de ahi, comienza a crecer otra vez, dando la falsa impresion de que ese es el valor minimo absoluto que puede tomar la funcion en todo su dominio. La consecuencia de esto en nuestro sistema es que nuestro vector gradiente nos guiara hasta el minimo local mas cercano que se encuentre, pero no es garantia de que ese sea el **minimo global** o absoluto de la funcion, y una vez llegados a ese punto (o muy cerca de el), como el negativo del gradiente solo apunta a la direccion de maximo decenso, y ya no podemos decender mas, el entramiento morira en ese punto. En altas dimensionalidades este problema es incluso mas activo.

##### La regla de la cadena 

Bien, para obtener nuestro vector gradiente, debemos calcular la **derivada parcial** de la funcion de perdida con respecto a **todo** peso en $\theta$.

Bien. Para una derivada parcial de una funcion de, por ejemplo dos variables,  $f(x, y)$, si $x$ y $y$ son directamente las variables independientes de $f$, es decir que no hay otras funciones de las que dependan $x$ y $y$, podemos calcular sus derivadas parciales $\frac{\partial f}{\partial x}$ y $\frac{\partial f}{\partial y}$ de manera directa. 

Sin embargo, hemos de recordar que en esta arquitectura, los pesos (sobre todo los de las primeras capas) no influyen directamente en $\mathcal{L}$ si no que lo hacen por medio de varias funciones a las cuales si afectan directamente (la preactivacion y la activacion).

Por lo que, analogicamente y retomando nuestro ejemplo $f(x, y)$. Si $x$ no es una independiente si no una funcion de otra variable, por ejemplo, $x = g(a)$, y lo mismo para $y$, y $y = h(b)$. Entonces ahora, si $a$ y $b$ son arbitrarias y nos interesa conocer $\frac{\partial f}{\partial a}$ y $\frac{\partial f}{\partial b}$, la cosa cambia.

Aqui es donde entra la regla de la cadena (*Chain Rule*), que nos indica la forma de obtener esas derivadas:

Sea $f(x, y)$ donde $x = g(a)$ y $y = h(b)$, entonces:

$$ \frac{\partial f}{\partial a} = \frac{\partial f}{\partial x} \cdot \frac{\partial x}{\partial a} $$ 

$$ \frac{\partial f}{\partial a} = \frac{\partial f}{\partial y} \cdot \frac{\partial y}{\partial b} $$ 

> * Esta regla me encanta porque es muy intuitiva!. Nota como, si queremos encontrar la derivada de $f$ con respecto a $a$, vamos de lo mas grande a lo mas pequeño, primero encontramos $f$ con respecto a $x$ y luego $x$ con respecto a $a$. Simple!. 
> * En la literatura, documentacion, codigo, y demas, podras encontrar los terminos de las derivadas en otro orden, porque la multiplicacion es conmutativa. 

Esta regla se aplica para tantas relaciones entre funciones y variables tengamos.

Bien, con esta herramienta. Podemos adentrarnos en el calculo de las derivadas del vector gradiente. 

Con nuestra activacion definida como:

$$ a_i^{(L+1)} = f(z_i^{(L+1)}) $$

$$ z_i^{(L+1)} = w_{i1}^{(L+1)}a_1^{(L)} + w_{i2}^{(L+1)}a_2^{(L)} + \cdots + w_{ij}^{(L+1)} a_{j}^{(L)} $$

> Antes de seguir leyendo, observa y analiza la dependencia: $\mathcal{L}$ depende de $a_i^{(L+1)}$, $a_i^{(L+1)}$ depende de $z_i^{(L+1)}$, $z_i^{(L+1)}$ depende de $w_{i1}^{(L+1)}a_1^{(L)} + w_{i2}^{(L+1)}a_2^{(L)} + \cdots + w_{ij}^{(L+1)} a_{j}^{(L)}$. En tu mente, visualiza a cada producto $w_{ij}^{(L+1)}a_j^{(L)}$ como una sola variable. Lo que vamos a encontrar es la derivada de $\mathcal{L}$ con respecto a una de esas variables.

La derivada parcial de la funcion de perdida $\mathcal{L}$ con respecto a cierto peso $w_{ij}$ de la capa $L$ es:

$$ \frac{\partial \mathcal{L} }{\partial w_{ij}^{L+1}} = \frac{\partial \mathcal{L} }{\partial a_i^{L+1}} \cdot \frac{\partial a_i^{L+1} }{\partial z_i^{L+1} } \cdot \frac{\partial z_i^{L+1} }{\partial w_{ij}^{L+1} }  $$

Ahora, hay algo que omiti al hablarte la funcion de perdida. En la realidad, los modelos se entrenan con base a un conjunto de datos de entrenamiento. Si queremos que nuestro modelo sea capaz de predecir correctamente sin importar la diversidad de los datos de entrada, necesitamos entrenarlo mostrandole tambien diversos ejemplos, tantos como sea necesario para cubrir la gran mayoria de casos posibles del fenomeno. Matematicamente, esto se logra optimizando sobre el promedio de todas las perdidas de cada uno de los ejemplos de entrenamiento.

Entonces si queremos obtener el vector gradiente "verdadero" de nuestro modelo, por cada derivada parcial con respecto a un peso, tenemos que obtener el promedio sobre todas las derivadas parciales de cada uno de los ejemplos por separado. Es decir:

$$ \frac{\partial \mathcal{L}_T }{\partial w_{ij}^{L+1}} = \frac{1}{N} \sum_{k=1}^{N} \frac{\partial \mathcal{L_k} }{\partial w_{ij}^{L+1}}  $$

Donde $N$ es el numero total de ejemplos en nuestro conjunto de datos de entrenamiento.

Nuestro vector gradiente total se veria asi:

$$ \nabla \mathcal{L} = \begin{bmatrix} \frac{\partial \mathcal{L}_T }{\partial w_{1}} = \frac{1}{N} \sum_{k=1}^{N} \frac{\partial \mathcal{L_k} }{\partial w_{1}} \\ \frac{\partial \mathcal{L}_T }{\partial w_{2}} = \frac{1}{N} \sum_{k=1}^{N} \frac{\partial \mathcal{L_k} }{\partial w_{2}}\\ \vdots \\ \frac{\partial \mathcal{L}_T }{\partial w_{n}} = \frac{1}{N} \sum_{k=1}^{N} \frac{\partial \mathcal{L_k} }{\partial w_{n}} \end{bmatrix} $$ 

$n$ es el numero total de pesos en la red.

Sin embargo, en el proceso de *backprop* de todos los ejemplos de entrenamiento, los gradientes se calculan exactamente igual, por eso para este analisis tomaremos $N = 1$. 


> * En realidad, este no es todo el algoritmo de backpropagation. Normalmente, despues de calcular el gradiente de la red, efectuariamos un **decenso por gradiente** (Gradient Descent), mediante el cual modificamos secuencialmente el valor de los pesos con base a la informacion del vector gradiente. Usualmente no usamos toda la magnitud del gradiente sino una parte, a este factor se lo conoce como la tasa de aprendizaje $ \alpha $, (**por lo general** $ 0 \lt \alpha \leqslant 1$), esto se debe a que el gradiente completo es demasiado "pesado", sobre todo al inicio del entrenamiento. El hecho de que apunte a la direccion de maximo decenso no significa que su magnitud sea la correcta para llegar directamente al punto minimo, a veces puede ser mucho mayor de la requerida, causando que $\mathcal{L}$ oscile al rededor del minimo, o muy pequeña, causando que avance muy lento. $\alpha$ es un **hiperparametro**, lo que significa que se escoje dependiendo del contexto, no es exactamente arbitrario, solo que se ajusta dependiendo del tipo de problema con el que estemos tratando.
> * Te has preguntado por que se dice que una red neuronal es un sistema **NO determinista**?. Es extraño, si lo piensas bien, si se conocen todos los ejemplos de entrenamiento, en teoria, podriamos calcular cada paso del entrenamiento, de principio a fin. Esto contradice al supuesto. Para que exista no determinismo, en algun punto del sistema tendriamos que ser incapaces de saber cual es su siguiente estado, y entonces? <br><br>
> Te comente que el gradiente "verdadero" se calcula sobre el promedio de todas las perdidas de cada ejemplo de entrenamiento por individual, asumiendo que usamos **todos** los datos del conjunto de datos de entrenamiento. En la practica, no usamos casi nunca el gradiente "real", basicamente, porque tiene mas desventajas que ventajas. Primero, computacionalmente es impensable, el hardware y tiempo requeridos para este calculo son mounstrosos, crecen significativamente con el tamaño de la red y el tamaño del conjunto de datos de entrenamiento. Podriamos tardar semanas para una sola iteracion del entrenamiento. Y segundo pero no menos importante, aveces usar la direccion "absoluta" hacia el minimo no es lo mas viable.  <br><br>
> Y aqui es donde entra el **descenso por gradiente estocastico** (Stochastic Gradient Descent, SGD). El SGD plantea: Por que en vez de usar todos los ejemplos de entrenamiento al mismo tiempo para calcular un gradiente "absoluto", no usamos solamente un grupo pequeño **aleatorio** de ellos por  cada iteracion?. A este grupo se le conoce como **mini-lote** (mini-batch), y claro, podrias pensar que al calcular el gradiente sobre un subconjunto del lote completo, no nos acercariamos al minimo, o que nos acercariamos muy lento. Pero, resulta ser que no es asi!. Experimentalmente se ha probado que esto funciona igual de bien e incluso mejor que el gradiente completo. Al usar un gradiente "aproximado", se introduce un ruido en el proceso que en ocaciones ayuda a que el gradiente pueda llegar a mejores minimos, o evitar que se quede estancado en minimos malos. Sin mencionar que la computacion para el SGD es mucho mas amigable y posible. Entonces, sacrificamos el determinismo y escogemos tener un gradiente menos exacto, a cambio de mayor rapidez, y algo de ruido beneficioso. Como los ejemplos de entrenamiento para cada iteracion son elegidos aleatoriamente, el proceso se convierte en estocastico.

Si has llegado hasta aqui y tienes todos los conceptos al dia. Ya estamos listos para adentrarnos en el problema sin preocuparnos discordancias de notacion. Vamos a ello!

---
### Activaciones y Neuronas muertas

Bien, ya sabemos como fluyen los datos a traves de la red, de ida y vuelta. En la jerga del campo estos flujos son mas conocidos como **inferencia** (hacia adelante) y **entrenamiento** (hacia atras), los cuales estan intimamente relacionados porque uno depende del otro en una relacion secuencial. La inferencia de un paso afecta el entrenamiento del siguiente, y el entrenamiento del siguiente afecta la inferencia del proximo, y asi sucesivamente. 

Pero vale la pena enfocarse particularmente en el algoritmo de entrenamiento, la retro-propagacion, que hace posible el "aprendizaje de la red", como ya vimos. 

Y es que, no es perfecto. O mas bien, existen algunos efectos secundarios de su implementacion que debemos considerar seriamente. 

> A partir de aqui nos sumergiremos en mucho terreno probabilistico, y podrias pensar que estos problemas son derivados de la entropia. Pero como veremos, no es asi, no ocurren por azar, sino por la naturaleza misma de las funciones que elegimos. Como dice el Merovingio de la saga Matrix: **"Donde otros ven casualidad, yo veo causalidad"**.

Con nuestra derivada de la perdida con respecto a un peso en especifico:

$$ \frac{\partial \mathcal{L} }{\partial w_{ij}^{L+1}} = \frac{\partial \mathcal{L} }{\partial a_i^{L+1}} \cdot \frac{\partial a_i^{L+1} }{\partial z_i^{L+1} } \cdot \frac{\partial z_i^{L+1} }{\partial w_{ij}^{L+1} }  $$

La derivada $\frac{\partial \mathcal{L} }{\partial a_i^{L+1}}$ se calcula dependiendo de la funcion de perdida que se asigne a la red, aun asi, debido a que todas las funciones de perdida son funcion de las activaciones finales, podemos analizar su comportamiento, pero lo dejaremos para secciones posteriores, por ahora nos centraremos en los ultimos dos terminos de esta expresion. En $\frac{\partial z_j^{L+1} }{\partial w_{ij}^{L+1}}$  como el peso es la variable independiente, y la activacion anterior una constante, entonces:

$$ \frac{\partial z_i^{L+1} }{\partial w_{ij}^{L+1}} = a_j^{L} $$

Y en $ \frac{\partial a_i^{L+1} }{\partial z_i^{L+1} } $, si $f = \tanh$, la derivada de la activacion con respecto a la preactivacion es (por regla de la cadena) simplemente la derivada de la tangente evaluada en esa misma preactivacion. Es decir:

$$ \frac{\partial a_i^{L+1} }{\partial z_i^{L+1} } = f'(z_i^{L+1}) = \tanh'(z_i^{L+1}) $$

Fijate como para ambos casos, $\frac{\partial z_i^{L+1} }{\partial w_{ij}^{L+1}}$ y $ \frac{\partial a_i^{L+1} }{\partial z_i^{L+1} }$ terminan dependiendo directamente de las activaciones anteriores y la preactivacion de la capa actual. Tambien, observa como $a_j^{L}$ es realidad en $\tanh(z_j^{L})$ (una preactivacion en L), lo que quiere decir que a fin de cuentas, la funcion de activacion determina absolutamente todo el comportamiento del flujo de la informacion a traves de la red, tanto en la propagacion hacia adelante como en la propagacion hacia atras.

> Por esta razon es muy importante elegir adecuadamente la funcion de activacion para cada modelo. Y como veremos mas adelante, existen soluciones propuestas que solventan el problema solo para algunas activaciones, no para todas.

Como hemos mostrado, toda la atencion se centra en la funcion de activacion, para nuestro caso, la tangente hiperbolica, y su derivada. Asi que vamos a investigar su curioso comportamiento.

> Puedes intentar aplicar el mismo analisis a las otras funciones de activacion, te daras cuenta como la gran mayoria sufren de problemas similares a los que mostraremos.

La grafica de $\tanh$ luce algo asi:

<br>
<p align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/8/87/Hyperbolic_Tangent.svg" width="480" height="200" alt="Perceptron-Multi-capa">
</p>

Esta funcion "comprime" los valores de entrada a valores entre -1 y 1. Teoricamente 1 y -1 solo se alcanzan cuando $x \to \pm\infty$, pero computacionalmente, basta con que las entradas se alejen un poco del origen para que el redondeo acerque los valores a 1 y -1. 

Por ejemplo, teoricamente:

$$ \tanh(2) \approx 0.96402758007 $$

**PyTorch**, la libreria por excelencia para el desarrollo de *Machine Learning*, tiene la funcion implementada, lista para usarse:

```python
torch.tensor(2).tanh()
# Salida: tensor(0.9640)
```
> en PyTorch, los **tensores** son la estructura fundamental, todos los valores son representados mediante tensores. Un tensor puede representar escalares, vectores y matrices de cualquier dimension.

Notese como solo se usan **4 cifras significativas**. Esto tiene mucho impacto. Veamos tambien otros ejemplos:

* $$ \tanh(5) \approx 0.99990920426 $$

```python
torch.tensor(5).tanh()
# Salida: tensor(0.9999)
```
> Como la quinta cifra significativa es 0, el redondeo deja el numero en 0.9999.

* $$\tanh(6) \approx 0.99998771165 $$

```python
torch.tensor(6).tanh()
# Salida: tensor(1.0000)
```

> Sin embargo aqui, la quinta cifra es 8, por lo que el redondeo aproxima el numero.

En realidad, estos ejemplo son solo conceptuales, para que comprendas que existe un limite en la capacidad de las maquinas para representar numeros irracionales (como los que produce $\tanh$). La precision de **visualizacion** por defecto de PyTorch es de 4 decimales, pero puede ser ajustada a discrecion. El limite real se encuentra en el tipo de dato que los tensores manejan, que es `float32` (32 bits), lo que significa que solo tendriamos aproximadamente 7 digitos de presicion para los valores de las activaciones:

* ```python
  torch.set_printoptions(precision=10)
  torch.tensor(9).tanh()
  # Salida: tensor(0.9999999404)   
  ```
Y

* ```python
  torch.set_printoptions(precision=10)
  torch.tensor(10).tanh()
  # Salida: tensor(1.)
  ```

Como se muestra, ajustar la precisión visual nos permite ver los dígitos que PyTorch normalmente oculta, ayudandonos a diferenciar valores que del 1 o -1 absolutos. Sin embargo, esto **es puramente visual**: para una entrada de 10, la diferencia con el 1 es tan pequeña que el float32 se queda sin capacidad para representarla. 

Por lo que, las preactivaciones con valores de aproximadamente 10 o mas (o -10), se traducen en activaciones puramente de 1 (o -1). En la practica, esto ocurre con frecuencia dependiendo de la inicialización de los pesos. Si no se controlan estos valores iniciales, las preactivaciones tienden a alcanzar estos extremos facilmente e incluso valores mayores en redes con muchas capas, lo veremos a detalle mas adelante, pero por ahora, llamaremos preativaciones "grandes" a los valores de aproximadamente 10 o mas, o aproximadamente -10 o menos.

Recordemos los terminos:

* $$ \frac{\partial z_i^{L+1} }{\partial w_{ij}^{L+1}} = \tanh(z_j^{L}) $$

>Recuerda que: $ a_j^L = \tanh(z_j^L)$

* $$ \frac{\partial a_i^{L+1} }{\partial z_i^{L+1} }  = \tanh'(z_i^{L+1})  $$ 

De lo analizado podemos sacar dos conclusiones:

 1. Si las preactivaciones $ z_j^{L} $ adquieren valores grandes, los terminos$ \frac{\partial z_i^{L+1} }{\partial w_{ij}^{L+1}}$ seran aproximadamente 1 o -1. Esto quiere decir que si las preactivaciones son muy grandes, en el proceso de propagacion hacia atras, a lo mucho, lo unico que aportaran sera un cambio de signo al gradiente de los pesos!. Y el aprendizaje que aporta un cambio binario es bastante precario en muchos casos.

> Cuidado, no siempre aporta solo un cambio de signo, esto solo es si las preactivaciones son los suficientemente grandes, si no lo son. Aporta un factor entre 1 y -1 que multiplica al resto del gradiente. 

 2. **La importante**: $$\tanh'(x) = 1 - \tanh^2(x)$$
. Si $ x \to \pm\infty $, entonces $\tanh'(x) \to 0$. Pero como ya vimos, computacionalmente, si las preativaciones $z_i^{L+1}$ tienden a ser grandes, los terminos $ \frac{\partial a_i^{L+1} }{\partial z_i^{L+1} }$ tenderan a 0!. 

Formalicemos estas dos reglas para los casos extremos:

   * Si $ z_j^{L} \to \pm\infty$, entonces, $ \frac{\partial z_i^{L+1} }{\partial w_{ij}^{L+1}} \to \pm1 $.

   * Si $z_i^{L+1} \to \pm\infty$, entonces $\frac{\partial a_i^{L+1} }     {\partial z_i^{L+1} } \to 0$.


Excelente, hemos analizado el comportamiento dos de los tres terminos de el gradiente para un peso de la red. Pero aun nos falta el mas importante. 

#### La propagacion del gradiente

Como vimos en las secciones anteriores, el gradiente de un peso depende tambien de un tercer termino. La derivada de la perdida con respecto a la activacion de la capa presente:

$$ \frac{\partial \mathcal{L} }{\partial a_i^{L+1}} $$

Este termino es especial porque aqui reside el llamado concepto de la propagacion del gradiente (*gradient propagation*). Y es que, la naturaleza de la regla de la cadena hace que el calculo de gradientes sea recursivo, sin esta caracteristica, no conseguiriamos ahorrarnos mucha computacion que imposibilitaria la optimizacion del entrenamiento.

---
**El termino** $\delta$

Antes de pasar al calculo de dicha expresion, considero prudente introducir otra notacion con la que vamos a trabajar y que muy probablemente encontraras en el material relacionado.

Como ya vimos $ \frac{\partial \mathcal{L} }{\partial a_i^{L+1}} $ nos dice que la perdida es funcion (en este caso directa) de la activacion $a_i^{L+1}$. Y $a_i^{L+1}$ es funcion de $z_i^{L+1}$. Esto significa que podemos encontrar la derivada de la perdida con respecto a $z_i^{L+1}$, asi que apliquemos la regla de la cadena otra vez:

$$ \frac{\partial \mathcal{L}}{\partial z_i^{L+1} } = \frac{\partial \mathcal{L} }{\partial a_i^{L+1}} \cdot  \frac{\partial a_i^{L+1}}{\partial z_i^{L+1} } $$

Este termino es particular porque aparecere de manera recursiva (como veremos en breves). Por ello, en la literatura, se le otorgo la siguiente notacion especial:

$$\delta_i^{L+1} = \frac{\partial \mathcal{L}}{\partial z_i^{L+1} }  $$

Ahora, nota como $\frac{\partial \mathcal{L} }{\partial a_i^{L+1}}$ y $\frac{\partial a_i^{L+1}}{\partial z_i^{L+1} }$ aparecen en $\frac{\partial \mathcal{L} }{\partial w_{ij}^{L+1}}$. Por lo que podemos sustituir expresiones:

$$ \frac{\partial \mathcal{L} }{\partial w_{ij}^{L+1}} = \delta_i^{L+1} \cdot \frac{\partial z_i^{L+1} }{\partial w_{ij}^{L+1} } $$

Ya vimos que $\frac{\partial z_i^{L+1} }{\partial w_{ij}^{L+1} } = a_j^L $, entonces:

$$ \frac{\partial \mathcal{L} }{\partial w_{ij}^{L+1}} = \delta_i^{L+1} \cdot a_j^L $$

> He aqui otra definicion del gradiente de un peso con la que podrias toparte en tu aprendizaje.

---
<br>

En $\frac{\partial \mathcal{L} }{\partial a_i^{L+1}}$, la activacion $a_i^{L+1}$ 












