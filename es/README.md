# Un Viaje al Nucleo de la Inicialización de Xavier
---
## Introducción
¿Alguna vez te has preguntado por qué, al construir una red neuronal, no podemos simplemente inicializar todos los pesos en 1, o en 0, o incluso en valores aleatorios gigantes como punto de partida?, porque, bueno, al final, se van a ir afinando solos no?

Seguramente has escuchado mucho de los problemas de "desvanecimiento de gradientes" (Vanishing Gradients) o "gradientes que explotan" (Exploding Gradients) que causan que tu red "muera" rápidamente: los gradientes desaparecen o las neuronas se saturan, dejando a la red incapaz de aprender incluso el patrón más simple.

Este documento no es solo una explicación teórica; es una derivación desde cero. Mi objetivo es acompañarte a través de estas páginas de razonamiento lógico y estadístico que llevaron a Xavier Glorot y Yoshua Bengio a proponer su famosa técnica de inicialización en 2010. No aceptaremos ninguna "fórmula" como caída del cielo, construiremos cada una de ellas usando las herramientas fundamentales de la matemática.

---
### ¿Qué abordaremos?

Exploraremos el flujo de la red neuronal desde una perspectiva de preservación de la varianza. Dividiremos el estudio en tres grandes bloques:

1. Los Cimientos (Capitulo 0): Definiciones intuitivas de variables aleatorias y por qué la varianza es nuestra mejor herramienta para medir la "salud" de una señal

2. La Propagacion hacia adelante (Capitulo 1): Cómo evitar que la información se desvanezca o explote mientras viaja desde la entrada hasta la predicción.

3. La Propagacion hacia atrás (Capitulo 1): La simetría oculta en el cálculo del error y cómo garantizar que el gradiente llegue vivo a las primeras capas de la red.

---
### ¿Esto es para ti? (Nivel mínimo requerido)

He diseñado esta demostracion para que sea lo más accesible posible, pero para poder seguir el hilo de la derivación sin perderte, es necesario que traigas estos conocimientos minimos previos:

**Para el análisis del Forward pass:**
- Arquitectura MLP: Debes entender qué es una neurona, una capa y cómo se conectan (el producto punto básico $z = Wx + b$ (preactivacion).

- Estadística Descriptiva: Debes sentirte cómodo con el concepto de Media (el promedio de los valores) y Varianza/Desviación Estándar (qué tan dispersos están los datos).

- Álgebra Básica: Despeje de ecuaciones, potencias y raíces cuadradas.

**Para el análisis del Backward pass:**

- Cálculo Diferencial: Entender qué es una derivada.

- Backpropagation: Una noción básica de cómo el error fluye hacia atrás en una red neuronal. (No te preocupes por las notaciones de las expresiones para los gradientes, cubriremos una introduccion de ello)

- Derivadas Parciales: No necesitas ser un experto, pero sí entender que podemos derivar respecto a una variable manteniendo las otras constantes.

Si cumples con esto, el resto lo construiremos aqui. Definiremos cada propiedad estadística nueva justo antes de que la necesitemos para el siguiente paso de la demostración. Asimismo, si ya dominas los conceptos sientete libre de saltarte a las secciones de interes. 


[Capitulo 0: El problema](https://bastianru.github.io/xavier-init-from-scratch/es/Capitulo-0)