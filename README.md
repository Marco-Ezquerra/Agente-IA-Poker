# Poker AI – Agente inteligente para Poker Texas Hold'em

Proyecto personal de inteligencia artificial aplicada al póker, desarrollado en Python. Combina simulación estadística, teoría de juegos y algoritmos de decisión para construir un agente capaz de tomar decisiones estratégicas en entornos con información incompleta.

---

##Objetivo

Diseñar un agente capaz de jugar al póker de forma estratégica, utilizando algoritmos como **Monte Carlo Tree Search (MCTS)** y **Counterfactual Regret Minimization (CFR)**, junto con análisis de rangos, evaluación de equity y simulación de partidas completas desde cualquier estado del juego.

---

##Tecnologías y herramientas

- **Lenguaje principal:** Python 3  
- **Algoritmos:** MCTS, CFR  
- **Análisis de datos:** NumPy, pandas  
- **Simulación:** motor propio desarrollado desde cero  
- **Evaluación de manos:** uso de la librería externa [`poker-eval`](https://github.com/atinm/poker-eval) *(no desarrollada por mí)*, utilizada exclusivamente para calcular el valor de manos y equity de forma eficiente  
- **Procesamiento:** estructuras propias para logs, simulaciones silenciosas, análisis de decisiones y backpropagation

---

## Estructura del proyecto

 Este proyecto puede funcionar de forma independiente si `poker-eval` está instalado o accesible como dependencia. Localmengte tengo m arpeta dentro de la carpeta poker-eval que se puede obtener de la siguiente manera:
 git clone https://github.com/atinm/poker-eval.git
cd poker-eval
autoreconf --install
./configure
make
sudo make install


## Funcionalidades del agente

- Simulación completa desde cualquier fase: preflop, flop, turn, river  
- Evaluación de acciones mediante MCTS con nodos personalizados  
- Cálculo de equity en tiempo real mediante `poker-eval`  
- Generación de logs estructurados para análisis posterior  
- En desarrollo: implementación de CFR, filtrado dinámico de rangos y retropropagación adaptativa

---

## Estado actual

- ✅ Motor funcional con simulación desde cualquier punto del juego  
- ✅ Integración de MCTS con evaluación de equity  
- ✅ Sistema de logging y análisis  
- 🔄 En progreso: lógica completa de CFR + generación de datasets  
- 🔜 Próximo paso: filtrado de rangos coherente con historial de acciones y mejora del árbol de decisiones

---



