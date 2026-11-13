# Arquitecturas de aprendizaje automático

> Implementaciones de [arquitecturas neuronales](https://www.deeplearningbook.org/), de McCulloch-Pitts (1943) a Mamba-3 (2026), cada una como respuesta a la limitación de la anterior.

Santiago Ramón y Cajal, dibujos de la corteza cerebral humana (1899):

![Santiago Ramón y Cajal, dibujos de la corteza cerebral humana](https://upload.wikimedia.org/wikipedia/commons/5/5b/Cajal_cortex_drawings.png)

## Contenido

- [Los cimientos](capitulo-1/)
  - [La neurona de McCulloch-Pitts](capitulo-1/1.1-mcculloch-pitts/)
  - [El perceptrón de Rosenblatt](capitulo-1/1.2-perceptron/)
  - [Adaline y la regla delta](capitulo-1/1.3-adaline/)
  - [El problema XOR y los límites de una capa](capitulo-1/1.4-xor/)
- [Profundidad y estructura](capitulo-2/)
  - [Perceptrón multicapa y retropropagación](capitulo-2/2.1-perceptron-multicapa/)
  - [Redes convolucionales](capitulo-2/2.2-redes-convolucionales/)
  - [Redes de Hopfield: memoria asociativa](capitulo-2/2.3-hopfield/)
  - [Redes recurrentes de Elman y el gradiente que se desvanece](capitulo-2/2.4-rnn-elman/)
  - [LSTM: aprender a recordar](capitulo-2/2.5-lstm/)
- [La revolución profunda](capitulo-3/)
  - [Activaciones e inicialización](capitulo-3/3.1-activaciones-e-inicializacion/)
  - [Redes residuales](capitulo-3/3.2-redes-residuales/)
  - [Secuencia a secuencia y atención](capitulo-3/3.3-seq2seq-y-atencion/)
- [El Transformer](capitulo-4/)
  - [Autoatención y el Transformer](capitulo-4/4.1-autoatencion-y-transformer/)
  - [El costo de la atención y la caché KV](capitulo-4/4.2-costo-y-cache-kv/)
  - [Mezcla de expertos](capitulo-4/4.3-mezcla-de-expertos/)
- [El regreso de la recurrencia](capitulo-5/)
  - [Atención lineal: el Transformer era una RNN](capitulo-5/5.1-atencion-lineal/)
  - [Modelos de espacio de estados: HiPPO y S4](capitulo-5/5.2-espacios-de-estados-s4/)
  - [Mamba: espacios de estados selectivos](capitulo-5/5.3-mamba/)
  - [Mamba-2 y la dualidad entre espacios de estados y atención](capitulo-5/5.4-mamba2-dualidad/)
- [Hoy (2024 a 2026)](capitulo-6/)
  - [La regla delta vuelve: DeltaNet y Gated DeltaNet](capitulo-6/6.1-regla-delta/)
  - [Híbridos: cuánta atención hace falta](capitulo-6/6.2-hibridos/)
  - [Estados complejos y seguimiento de estado: la idea de Mamba-3](capitulo-6/6.3-estados-complejos/)

## Referencias

- Goodfellow, Bengio y Courville (2016). *Deep Learning*. https://www.deeplearningbook.org/
- Prince (2023). *Understanding Deep Learning*. https://udlbook.github.io/udlbook/
- Zhang, Lipton, Li y Smola. *Dive into Deep Learning*. https://d2l.ai/
- *Introduction to Neural Network Models of Cognition*. https://com-cog-book.github.io/com-cog-book/intro
- Gu (2025). On the tradeoffs of SSMs and transformers. https://goombalab.github.io/blog/2025/tradeoffs/
