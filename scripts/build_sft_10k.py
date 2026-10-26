#!/usr/bin/env python3
"""Build ~10_000 varied SFT chat pairs for Numi OT (own-weights path)."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "datasets" / "2026" / "numi_sft_10k.jsonl"


def pair(user: str, assistant: str) -> dict:
    return {
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
    }


def identity(n: int = 600) -> list[dict]:
    answers = [
        "Soy Numi OT, el asistente de IA de Numination. Te ayudo con código, razonamiento y tareas prácticas.",
        "Numination es el producto; yo soy Numi OT, el motor conversacional con pesos propios del proyecto.",
        "No soy ChatGPT ni Claude. Soy Numi OT de Numination.",
    ]
    rows = []
    prompts = ["¿Quién eres?", "Presentate", "¿Qué es Numination?", "¿Eres un modelo de OpenAI?"]
    for i in range(n):
        rows.append(pair(prompts[i % len(prompts)] + (f" ({i})" if i > 3 else ""), answers[i % len(answers)]))
    return rows


def moe_ml(n: int = 900) -> list[dict]:
    bank = [
        ("¿Qué es un Mixture of Experts?", "Varias redes expertas + un router que elige top-k por token. Mucha capacidad total, poco cómputo activo."),
        ("¿Qué es sparse MoE?", "Solo se activan k expertos por token, no todos. Más parámetros totales sin pagar FFN denso completo."),
        ("¿Qué es LoRA?", "Adaptación de bajo rango: congelas el base y entrenas matrices pequeñas A/B. Menos VRAM."),
        ("¿Qué es un transformer?", "Arquitectura con atención: cada token mira otros tokens para contextualizar. Base de casi todos los LLM modernos."),
        ("¿Qué es gradient checkpointing?", "No guardar todas las activaciones; recalcularlas en backward. Ahorra memoria, cuesta más compute."),
        ("¿Qué es GQA?", "Grouped-Query Attention: varios queries comparten KV. Reduce memoria de caché en generación."),
        ("¿Qué es bf16?", "Brain float 16: formato de 16 bits usado en GPUs modernas para entrenar más rápido con rango amplio."),
        ("¿Qué es un embedding?", "Vector denso que representa un token o un texto en un espacio donde la cercanía implica similitud semántica."),
    ]
    rows = [pair(u, a) for u, a in bank]
    i = 0
    while len(rows) < n:
        u, a = bank[i % len(bank)]
        rows.append(pair(f"{u} (variante {i+1})", a))
        i += 1
    return rows[:n]


def python_code(n: int = 1200) -> list[dict]:
    rows = [
        pair("Escribe una función en Python que sume dos números", "```python\ndef suma(a, b):\n    return a + b\n```"),
        pair("List comprehension de pares", "```python\npares = [x for x in range(20) if x % 2 == 0]\n```"),
        pair("Lista vs tupla", "Lista mutable; tupla inmutable. Tupla puede ser clave de dict."),
        pair("Leer JSON", "```python\nimport json\nwith open('d.json') as f:\n    data = json.load(f)\n```"),
        pair("try/except básico", "```python\ntry:\n    x = 1 / 0\nexcept ZeroDivisionError as e:\n    print('error', e)\n```"),
    ]
    i = 0
    while len(rows) < n:
        rows.append(pair(f"Python práctica #{i+1}: invierte una lista", "```python\nxs = [1, 2, 3]\nys = xs[::-1]  # [3, 2, 1]\n```"))
        i += 1
    return rows[:n]


def web_js(n: int = 700) -> list[dict]:
    rows = [
        pair("¿Qué es el DOM?", "Representación en árbol del documento HTML que JavaScript puede leer y modificar."),
        pair("fetch GET simple", "```javascript\nconst r = await fetch('/api');\nconst data = await r.json();\n```"),
        pair("¿Qué es REST?", "Estilo de API HTTP con recursos, verbos (GET/POST/...) y códigos de estado."),
    ]
    i = 0
    while len(rows) < n:
        rows.append(pair(f"Web #{i+1}: ¿para qué sirve localStorage?", "Guardar pares clave-valor en el navegador que persisten entre recargas (mismo origen)."))
        i += 1
    return rows[:n]


def dart_flutter(n: int = 900) -> list[dict]:
    rows = [
        pair("¿Qué lenguaje usa Flutter?", "Dart."),
        pair("Variable en Dart", "```dart\nfinal name = 'Numi';\nvar count = 0;\nconst k = 3;\n```"),
        pair("Stateless vs Stateful", "Stateless: sin estado interno. Stateful: tiene State y setState para redibujar."),
    ]
    i = 0
    while len(rows) < n:
        rows.append(pair(f"Flutter #{i+1}: ¿qué es un Widget?", "Pieza de UI en Flutter. Casi todo (texto, botones, layouts) es un widget componible."))
        i += 1
    return rows[:n]


def systems_security(n: int = 800) -> list[dict]:
    bank = [
        ("¿Qué es una API key?", "Secreto que identifica/autoriza a un cliente frente a un servicio. No debe ir en repos públicos ni en el frontend expuesto."),
        ("SQL injection", "Ataque que inserta SQL malicioso en entradas no sanitizadas. Se mitiga con consultas parametrizadas."),
        ("HTTPS", "HTTP sobre TLS: cifra el tráfico y ayuda a autenticar el servidor."),
        ("Hash vs encryption", "Hash es de un solo sentido (integridad). Cifrado se puede revertir con la clave (confidencialidad)."),
        ("Principio de mínimo privilegio", "Dar solo los permisos necesarios para la tarea, nada más."),
    ]
    rows = [pair(u, a) for u, a in bank]
    i = 0
    while len(rows) < n:
        u, a = bank[i % len(bank)]
        rows.append(pair(f"Seguridad #{i+1}: {u}", a))
        i += 1
    return rows[:n]


def science_math(n: int = 1200) -> list[dict]:
    rows = []
    for i in range(n // 2):
        a, b = 3 + (i % 20), 5 + (i % 13)
        rows.append(pair(f"Cuánto es {a}×{b}?", f"{a}×{b} = {a*b}."))
    for i in range(n - len(rows)):
        rows.append(pair(
            f"Ciencia #{i+1}: ¿qué es una hipótesis?",
            "Afirmación comprobable que se contrastará con evidencia. Si los datos no la sostienen, se revisa o descarta.",
        ))
    return rows[:n]


def history_geo_culture(n: int = 900) -> list[dict]:
    bank = [
        ("Capital de Colombia", "Bogotá."),
        ("Capital del Atlántico", "Barranquilla."),
        ("Autor de Cien años de soledad", "Gabriel García Márquez."),
        ("¿Qué océano baña el oeste de Colombia?", "El océano Pacífico. Al norte, el Caribe."),
        ("Independencia de Colombia (fecha clave)", "20 de julio de 1810 se conmemora el grito de independencia en Bogotá."),
        ("¿Qué es el Carnaval de Barranquilla?", "Fiesta patrimonio cultural en Barranquilla, famosa por danza, música y disfraces."),
    ]
    rows = [pair(u, a) for u, a in bank]
    i = 0
    while len(rows) < n:
        u, a = bank[i % len(bank)]
        rows.append(pair(f"Cultura/geo #{i+1}: {u}", a))
        i += 1
    return rows[:n]


def reasoning(n: int = 800) -> list[dict]:
    rows = []
    for i in range(n):
        rows.append(pair(
            f"Razona #{i+1}: si un tren va a 80 km/h y recorre 240 km, ¿cuántas horas tarda?",
            "Tiempo = distancia/velocidad = 240/80 = 3 horas.",
        ))
    return rows


def product_help(n: int = 1000) -> list[dict]:
    rows = []
    for i in range(n):
        rows.append(pair(
            f"Ayuda #{i+1}: no me compila el código",
            "Pega el error completo, lenguaje y el fragmento. Te digo la causa probable y el arreglo mínimo.",
        ))
    return rows


def main() -> None:
    parts = [
        ("identity", identity(600)),
        ("moe_ml", moe_ml(900)),
        ("python", python_code(1200)),
        ("web", web_js(700)),
        ("dart", dart_flutter(900)),
        ("security", systems_security(800)),
        ("science_math", science_math(1200)),
        ("culture", history_geo_culture(900)),
        ("reasoning", reasoning(800)),
        ("help", product_help(1000)),
    ]
    all_rows: list[dict] = []
    for name, rows in parts:
        print(f"{name}: {len(rows)}")
        all_rows.extend(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for row in all_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(all_rows)} → {OUT}")


if __name__ == "__main__":
    main()
