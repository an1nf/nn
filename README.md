# Notas de curso: arquitecturas de aprendizaje automático

### Del perceptrón a Mamba-3

Implementaciones mínimas de arquitecturas neuronales, de McCulloch-Pitts (1943) a Mamba-3 (2026), cada una como respuesta a la limitación de la anterior. Capítulos 1–2 en NumPy con gradientes a mano; 3–6 en PyTorch.

* Capítulo 1: Los cimientos
  * 1.1: [La neurona de McCulloch-Pitts](capitulo-1/1.1-mcculloch-pitts/)
  * 1.2: [El perceptrón de Rosenblatt](capitulo-1/1.2-perceptron/)
  * 1.3: [Adaline y la regla delta](capitulo-1/1.3-adaline/)
  * 1.4: [El problema XOR y los límites de una capa](capitulo-1/1.4-xor/)
* Capítulo 2: Profundidad y estructura
  * 2.1: [Perceptrón multicapa y retropropagación](capitulo-2/2.1-perceptron-multicapa/)
  * 2.2: [Redes convolucionales](capitulo-2/2.2-redes-convolucionales/)
  * 2.3: [Redes de Hopfield: memoria asociativa](capitulo-2/2.3-hopfield/)
  * 2.4: [Redes recurrentes de Elman y el gradiente que se desvanece](capitulo-2/2.4-rnn-elman/)
  * 2.5: [LSTM: aprender a recordar](capitulo-2/2.5-lstm/)
* Capítulo 3: La revolución profunda
  * 3.1: [Activaciones e inicialización](capitulo-3/3.1-activaciones-e-inicializacion/)
  * 3.2: [Redes residuales](capitulo-3/3.2-redes-residuales/)
  * 3.3: [Secuencia a secuencia y atención](capitulo-3/3.3-seq2seq-y-atencion/)
* Capítulo 4: El Transformer
  * 4.1: [Autoatención y el Transformer](capitulo-4/4.1-autoatencion-y-transformer/)
  * 4.2: [El costo de la atención y la caché KV](capitulo-4/4.2-costo-y-cache-kv/)
  * 4.3: [Mezcla de expertos](capitulo-4/4.3-mezcla-de-expertos/)
* Capítulo 5: El regreso de la recurrencia
  * 5.1: [Atención lineal: el Transformer era una RNN](capitulo-5/5.1-atencion-lineal/)
  * 5.2: [Modelos de espacio de estados: HiPPO y S4](capitulo-5/5.2-espacios-de-estados-s4/)
  * 5.3: [Mamba: espacios de estados selectivos](capitulo-5/5.3-mamba/)
  * 5.4: [Mamba-2 y la dualidad entre espacios de estados y atención](capitulo-5/5.4-mamba2-dualidad/)
* Capítulo 6: Hoy (2024–2026)
  * 6.1: [La regla delta vuelve: DeltaNet y Gated DeltaNet](capitulo-6/6.1-regla-delta/)
  * 6.2: [Híbridos: cuánta atención hace falta](capitulo-6/6.2-hibridos/)
  * 6.3: [Estados complejos y seguimiento de estado: la idea de Mamba-3](capitulo-6/6.3-estados-complejos/)

## Mapa

```mermaid
flowchart TD
    A["1.1 Neurona de McCulloch-Pitts (1943)"] -->|"¿cómo aprender los pesos?"| B["1.2 Perceptrón (1958) · 1.3 Adaline (1960)"]
    B -->|"solo fronteras lineales: XOR (1.4)"| C["2.1 Perceptrón multicapa y retropropagación (1986)"]
    C -->|"imágenes: demasiados parámetros, sin invariancias"| D["2.2 Redes convolucionales (1989-1998)"]
    C -->|"recordar patrones completos"| H["2.3 Redes de Hopfield (1982)"]
    C -->|"secuencias de longitud variable"| E["2.4 RNN de Elman (1990)"]
    E -->|"el gradiente se desvanece"| F["2.5 LSTM (1997)"]
    D -->|"las redes profundas no entrenan"| G["3.1 ReLU, inicialización, GPU (2010-2012)"]
    G -->|"más capas, peor ajuste"| R["3.2 ResNet (2015)"]
    F -->|"toda la frase en un solo vector"| S["3.3 Seq2seq y atención (2014)"]
    S -->|"la recurrencia impide paralelizar"| T["4.1 Transformer (2017)"]
    R -->|"conexiones residuales"| T
    H -.->|"la atención es una Hopfield moderna"| T
    T -->|"costo cuadrático, caché KV que crece"| K["4.2 GQA, MQA, MLA · 4.3 Mezcla de expertos"]
    T -->|"costo cuadrático"| L["5.1 Atención lineal (2020) · 5.2 S4 (2021)"]
    L -->|"dinámica fija: no elige qué recordar"| M["5.3 Mamba (2023) · 5.4 Mamba-2 (2024)"]
    M -->|"olvida todo por igual"| N["6.1 DeltaNet y Gated DeltaNet (2024)"]
    N -->|"memoria finita"| O["6.2 Híbridos (2024-2026)"]
    K --> O
    M -->|"no puede contar: paridad"| P["6.3 Estados complejos, Mamba-3 (2026)"]
```

## Resultados

| | |
|---|---|
| ![XOR y espacio oculto](capitulo-2/2.1-perceptron-multicapa/figuras/xor_y_espacio_oculto.png) | ![Hopfield](capitulo-2/2.3-hopfield/figuras/recuperacion_de_letras.png) |
| La capa oculta vuelve separable XOR (2.1). | La red de Hopfield moderna recupera letras que la clásica confunde (2.3). |
| ![Flujo del gradiente en la LSTM](capitulo-2/2.5-lstm/figuras/flujo_y_compuertas.png) | ![Cabezas de inducción](capitulo-4/4.1-autoatencion-y-transformer/figuras/mapas_de_atencion.png) |
| La celda de la LSTM conserva el gradiente que la RNN pierde (2.5). | Cabezas de inducción en un Transformer de dos capas (4.1). |
| ![Mamba](capitulo-5/5.3-mamba/figuras/paso_selectivo.png) | ![Regla delta](capitulo-6/6.1-regla-delta/figuras/sobrescritura.png) |
| Mamba abre su paso $\Delta_t$ cuando llega un dato (5.3). | La regla delta reemplaza una asociación; la suma la mezcla (6.1). |

## Uso

```bash
git clone https://github.com/an1nf/nn.git && cd nn
python3 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python capitulo-1/1.2-perceptron/perceptron.py
```

CPU; semillas fijas. 3.3, 5.3 y 6.2 tardan varios minutos.

## Sitio

```bash
pip install markdown pygments pymdown-extensions
python sitio/construir.py && python -m http.server -d _sitio 8000
```

Se publica en GitHub Pages con `.github/workflows/pages.yml` en cada `push` a `main`.

## Referencias

- Goodfellow, Bengio y Courville (2016). *Deep Learning*. https://www.deeplearningbook.org/
- Prince (2023). *Understanding Deep Learning*. https://udlbook.github.io/udlbook/
- Zhang, Lipton, Li y Smola. *Dive into Deep Learning*. https://d2l.ai/
- *Introduction to Neural Network Models of Cognition*. https://com-cog-book.github.io/com-cog-book/intro
- Gu (2025). On the tradeoffs of SSMs and transformers. https://goombalab.github.io/blog/2025/tradeoffs/
