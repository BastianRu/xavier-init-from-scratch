
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

$$ y = w_1x_1 + w_2x_2 + w_1x_1 + \cdots + w_nx_n $$

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

$$ a = f(w_1x_1 + w_2x_2 + w_1x_1 + \cdots + w_nx_n) $$ 

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
##
<p align="center">
  <img src="https://raw.githubusercontent.com/ledell/sldm4-h2o/master/mlp_network.png" width="480" height="200" alt="Perceptron-Multi-capa">
</p>

##
Puedes notar dos relaciones muy importante entre capas:
1.  **El numero de outputs (activaciones) de una capa, determina el numero de inputs para cada neurona de la siguiente capa, o lo que es lo mismo, el numero de inputs a la siguiente capa**.

> En el diagrama de ejemplo, la primera capa tiene 6 valores, que podemos considerar indistintamente inputs u outputs dado que no se aplica ninguna operacion, entonces, la primera capa oculta, recibe 6 inputs en cada neurona.

Y por consecuente:

2. **El numero de outpus de una capa, determina el numero de pesos para cada neurona de la siguiente capa**

> Cada neurona de la segunda capa, recibe 6 inputs, y como cada conexion de la primera capa hacia la segunda implica un peso, cada neurona en la segunda capa tiene 6 pesos. En total, la segunda capa tiene $$4 \cdot 6 = 24$$ pesos.

Ahora. Aterrizaremos todos los conceptos con notacion concreta, para referirnos a ellos de una forma mas rapida. 
 

----
### Recordando la arquitectura MLP (Backward Pass)
















