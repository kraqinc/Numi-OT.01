# Numination — Curriculum avanzado para pesos propios

Objetivo: que los pesos de Numi aprendan lenguaje, razonamiento, seguimiento de instrucciones, uso de herramientas y generación fiable, sin depender de frameworks externos como base de conocimiento.

Los enlaces son referencias de investigación e implementación para estudiar algoritmos, no dependencias que haya que instalar.

---

## 1. Preparación avanzada de los datos de entrenamiento

Objetivo: conseguir que los pesos aprendan información útil y no solamente memoricen texto.

| Tema | Detalle |
|------|--------|
| Distribución de calidad del corpus | Asignar diferentes pesos de muestreo a textos educativos, técnicos, científicos y conversaciones. |
| Deduplicación aproximada | Detectar documentos casi idénticos, incluso si cambian algunas palabras o el formato. |
| Detección de texto basura | Identificar HTML residual, texto incompleto, spam, caracteres corruptos y documentos sin contenido útil. |
| Filtrado por idioma | Separar español, inglés, portugués y otros idiomas, controlando la proporción de cada uno. |
| Balance de dominios | Evitar que una categoría o fuente domine el entrenamiento y perjudique otras capacidades. |
| Distribución de dificultad | Organizar ejemplos sencillos, intermedios y complejos para estudiar cómo cambia el aprendizaje. |
| Mezcla de datos durante el entrenamiento | Cambiar la proporción de dominios a lo largo de las etapas de entrenamiento. |
| Contaminación de evaluación | Evitar que las preguntas utilizadas para medir al modelo aparezcan previamente en el corpus de entrenamiento. |
| Procedencia de los datos | Registrar su origen, licencia, fecha, transformaciones y versión. |
| Detección de datos sintéticos repetitivos | Evitar alimentar el modelo con grandes cantidades de respuestas artificiales que repitan los mismos errores. |
| Densidad informativa | Medir cuánto contenido útil contiene un documento frente a su extensión. |
| Cobertura de casos poco frecuentes | Incluir ejemplos de caracteres especiales, errores tipográficos, nombres propios y situaciones excepcionales. |

Recursos:
- [FineWeb: preparación de grandes corpus de texto](https://arxiv.org/abs/2406.17557)
- [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)

---

## 2. Tokenización y representación interna del lenguaje

Objetivo: que el motor transforme texto en tokens y vuelva a producir texto correctamente.

| Tema | Detalle |
|------|--------|
| BPE desde cero | Aprender cómo se construye un vocabulario a partir de combinaciones frecuentes de símbolos. |
| Tokenización por bytes | Representar cualquier texto mediante bytes, con una estrategia definida para caracteres desconocidos. |
| Normalización Unicode | Manejar acentos, emojis, caracteres combinados y representaciones equivalentes sin corromper el texto. |
| Tokenización multilingüe | Investigar cómo dividir palabras de idiomas con diferentes alfabetos y estructuras morfológicas. |
| Tokenización de código | Preservar indentación, operadores, nombres de variables, comentarios y símbolos de programación. |
| Longitud de secuencia | Estudiar cuántos tokens necesita cada idioma o tipo de texto para expresar la misma información. |
| Identificadores especiales | Definir IDs reservados para terminar mensajes, separar roles y representar llamadas a herramientas. |
| Tokenización reversible | Comprobar que codificar y decodificar texto no introduce pérdidas inesperadas. |
| Compatibilidad vocabulario-pesos | Mantener el mismo orden de IDs entre el tokenizador, la matriz de embeddings y la capa de salida. |
| Análisis de tokens desconocidos | Detectar cuándo nombres propios, símbolos o idiomas nuevos producen secuencias poco eficientes. |

Recurso: [Neural Machine Translation of Rare Words with Subword Units](https://aclanthology.org/P16-1162/)

---

## 3. Arquitecturas internas para tus propios pesos

| Tema | Detalle |
|------|--------|
| Máscara causal | Impedir que un token utilice información de tokens futuros durante el entrenamiento autorregresivo. |
| Codificación posicional rotatoria (RoPE) | Incorporar información sobre la posición de los tokens mediante rotaciones. |
| RMSNorm | Normalizar activaciones con operación distinta de LayerNorm. |
| SwiGLU | Función de activación con compuerta en las capas feed-forward. |
| Conexiones residuales | Permitir que la información atraviese capas profundas. |
| Inicialización de parámetros | Distribuciones iniciales que ayuden a evitar activaciones o gradientes excesivos. |
| Embeddings compartidos | Estudiar si conviene compartir embeddings de entrada y de salida. |
| Grouped-Query Attention (GQA) | Compartir grupos de cabezas KV para reducir memoria de inferencia. |
| Multi-Query Attention (MQA) | Reducción de cabezas KV frente a atención multi-cabeza tradicional. |
| Mixture of Experts (MoE) | Distribuir tokens entre expertos y medir equilibrio de carga. |
| Escalado de residuales | Controlar la magnitud de señales que recorren muchas capas. |
| Arquitecturas de espacio de estados | Comparar alternativas a la atención tradicional. |
| Atención local y global | Cuándo atender a todo el contexto y cuándo limitar a ventanas. |
| Extrapolación posicional | Comportamiento con contextos superiores a los vistos en entrenamiento. |

Recursos:
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [GQA](https://aclanthology.org/2023.emnlp-main.298/)
- [Mamba](https://arxiv.org/abs/2312.00752)

---

## 4. Entrenamiento numérico desde cero

| Tema | Detalle |
|------|--------|
| Entropía cruzada por token | Diferencia entre predicciones y tokens esperados. |
| Máscaras de pérdida | Qué posiciones cuentan (especialmente respuestas del asistente). |
| Retropropagación | Gradientes de atención y proyecciones. |
| Comprobación numérica de gradientes | Comparar analíticos vs diferencias finitas. |
| Acumulación de gradientes | Simular lotes efectivos grandes. |
| Recorte de gradientes | Limitar gradientes excesivos. |
| Calentamiento / decaimiento de LR | Estabilizar inicio y final del entrenamiento. |
| Optimización por lotes | AdamW, SGD y variantes. |
| Precisión mixta | FP32, FP16, BF16. |
| Estabilidad del softmax | Evitar desbordamientos. |
| Detección de NaN / Inf | Fallar rápido ante actualizaciones inválidas. |
| Reanudación exacta | Guardar pesos, optimizador, step, RNG y posición de datos. |
| Sobreajuste de prueba | Verificar que el modelo puede aprender un set diminuto. |
| Curvas de aprendizaje | Diagnóstico de sub/sobreentrenamiento. |
| Presupuesto de cómputo | Parámetros × tokens × tiempo. |

---

## 5. Enseñar al modelo a conversar

| Tema | Detalle |
|------|--------|
| Supervised Fine-Tuning (SFT) | Pares instrucción → respuesta. |
| Conversaciones multivuelta | Respuestas que dependen del historial. |
| Formatos de conversación | Roles, límites, fin de turno, herramientas. |
| Seguimiento de instrucciones | Solicitud principal vs restricciones vs formato. |
| Respuestas negativas apropiadas | Rechazar lo no permitido sin dejar de contestar lo válido. |
| Preferencias humanas | Utilidad, claridad, precisión, seguridad. |
| Direct Preference Optimization (DPO) | Pares preferido / no preferido. |
| Aprendizaje mediante retroalimentación | Evaluaciones humanas o automáticas. |
| Supervisión del proceso | Evaluar pasos intermedios, no solo el final. |
| Destilación de capacidades | Transferir comportamientos de un modelo docente. |
| Consistencia conversacional | Mantener definiciones y compromisos. |
| Control del formato | Tablas, JSON, estructuras exigidas. |
| Ejemplos difíciles | Instrucciones ambiguas y restrictivas. |
| Adaptación sin olvido | Mejorar una habilidad sin deteriorar otras. |

Recursos:
- InstructGPT
- Direct Preference Optimization
- Let's Verify Step by Step

---

## 6. Algoritmos de generación e inferencia

| Tema | Detalle |
|------|--------|
| Decodificación greedy | Token de mayor puntuación. |
| Temperatura / top-k / top-p | Control de variabilidad. |
| Penalizaciones de repetición | Sin dañar nombres o código. |
| Secuencias de parada | Detener en delimitadores. |
| KV cache | Reutilizar claves/valores. |
| Gestión del contexto | Qué hacer cuando se supera el límite. |
| Streaming de tokens | Entregar fragmentos en vivo. |
| Generación reproducible | Semillas controladas. |
| Batching de solicitudes | Aprovechar hardware. |
| Decodificación especulativa | Modelo auxiliar + verificación. |
| Atención eficiente en memoria | FlashAttention y variantes. |
| Cuantización de inferencia | Menor precisión, medir calidad. |

---

## 7. Uso de herramientas en Numination

| Tema | Detalle |
|------|--------|
| Selección de herramientas | Cuándo llamar una función. |
| Generación y validación de argumentos | JSON correcto según esquema. |
| Ciclo de ejecución | Solicitud → resultado → respuesta final. |
| Manejo de errores | No inventar éxito. |
| Reintentos controlados | Solo operaciones idempotentes. |
| Salidas estructuradas | JSON válido consistente. |
| Calculadora / búsqueda / código aislado | Delegar cuando conviene. |
| Permisos y auditoría | Registrar llamadas y resultados. |

Recurso: Toolformer

---

## 8. Recuperación de información y respuestas verificables

| Tema | Detalle |
|------|--------|
| Segmentación de documentos | Fragmentos con contexto. |
| Recuperación híbrida | Semántica + keywords. |
| Reordenamiento y fusión | Priorizar lo útil. |
| Atribución de afirmaciones | Relacionar claim con evidencia. |
| Evidencia insuficiente | No inventar. |
| Conocimiento temporal | Histórico vs actual. |
| Resolución de contradicciones | Fechas, contexto, credibilidad. |
| Separación instrucciones / documentos | Contenido recuperado = datos, no system. |
| Control de alucinaciones | Detectar claims no respaldados. |
| Citas verificables | Fuentes reales. |
| Memoria por usuario | Aislamiento entre cuentas. |

---

## 9. Evaluación avanzada de tus pesos

| Tema | Detalle |
|------|--------|
| Perplejidad | Predicción en set no visto. |
| Evaluación por capacidad | Matemáticas, código, idiomas, lógica, instrucciones. |
| Pruebas de regresión | Detectar deterioros. |
| Conversaciones largas | Continuidad y referencias. |
| Herramientas | Selección, argumentos, errores. |
| Multilingüe | Cada idioma por separado. |
| Calibración de confianza | Certeza vs acierto. |
| Generalización | Ejemplos nuevos. |
| Fugas de evaluación | Overlap train/test. |
| Adversarial | Entradas confusas o maliciosas. |
| Comparación ciega | Sin revelar versión. |
| Análisis de errores | Agrupar fallos y generar nuevos ejemplos. |
| Rendimiento por recurso | Calidad + memoria + latencia. |
| Evaluación continua | Tras cada cambio. |

Recurso: EleutherAI LM Evaluation Harness

---

## 10. Implementación nativa sin dependencias de IA

| Tema | Detalle |
|------|--------|
| Motor tensorial propio | Representación interna de tensores. |
| Multiplicación de matrices | Operación dominante. |
| Núcleos de atención | QK, máscara, softmax, V. |
| Normalización / embeddings | Según arquitectura. |
| Retropropagación manual | Sin autograd externo. |
| Planificación de memoria | Reutilizar buffers. |
| SIMD / kernels GPU | Según hardware. |
| Verificación entre motores | Resultados intermedios. |
| Formato propio de checkpoints | Versiones, dims, integridad. |
| Carga segura de pesos | Rechazar archivos inválidos. |
| Determinismo y perfilado | Medir y documentar. |

Referencias educativas:
- karpathy/llm.c
- karpathy/build-nanogpt

---

## 11. Seguridad específica de modelos propios

| Tema | Detalle |
|------|--------|
| Inyección de prompts | Instrucciones hostiles. |
| Fugas de contexto interno | No exponer system/private. |
| Envenenamiento de datos | Ejemplos manipulados. |
| Divulgación de información sensible | Secretos / PII. |
| Exceso de autonomía | Límites de acción. |
| Validación de código | Antes de ejecutar. |
| Aislamiento entre peticiones | Estado separado. |
| Robustez | Entradas inesperadas. |
| Límites de generación | Tokens, tiempo, tamaño. |
| Seguridad del checkpoint | Integridad y acceso. |

Recurso: OWASP Top 10 for LLM Applications

---

## 12. Prioridad para Numination (orden de trabajo)

1. **Tokenizador y formato de conversación** — IDs, límites de mensajes y tokens de fin deben coincidir exactamente con los usados al entrenar.
2. **Motor de inferencia** — embeddings, atención, normalización, FFN, logits y generación token a token.
3. **Entrenamiento verificable** — pérdida, gradientes, actualizaciones, checkpoints y reanudación.
4. **Ajuste conversacional** — SFT con el formato real de Numination.
5. **Evaluación automatizada** — sets independientes que detecten mejoras y regresiones.
6. **Herramientas y recuperación** — llamadas y validaciones en el sistema que rodea los pesos.

> Conocer un tema durante el entrenamiento no garantiza dominarlo. Cada capacidad necesita ejemplos adecuados, datos de calidad y pruebas que demuestren que el modelo realmente la ha aprendido.
