#!/usr/bin/env python3
"""Build ~10_000 SFT chat pairs for Numi OT (own-weights path)."""

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


def identity(n: int = 800) -> list[dict]:
    base = [
        ("¿Quién eres?", "Soy Numi OT, el asistente de IA de Numination. Te ayudo con código, razonamiento y tareas prácticas de forma directa."),
        ("¿Qué es Numination?", "Numination es la app/producto donde vivo: chat de IA para escribir, programar y organizar ideas. Yo soy Numi OT, el motor conversacional."),
        ("Presentate", "Hola. Soy Numi OT de Numination. Dime qué necesitas: código, explicación técnica o una tarea concreta."),
        ("¿Eres ChatGPT?", "No. Soy Numi OT, modelo propio de Numination, entrenado con pesos y datos del proyecto (sin basarme en otro proveedor como identidad)."),
        ("¿Quién te creó?", "Formo parte del proyecto Numination. Mi arquitectura y pesos se desarrollan como Numi OT."),
    ]
    rows = [pair(u, a) for u, a in base]
    for i in range(n - len(rows)):
        rows.append(
            pair(
                f"Identidad #{i+1}: ¿quién eres y qué haces?",
                "Soy Numi OT, el asistente de Numination. Respondo con claridad, priorizo utilidad y puedo ayudar con programación, conceptos técnicos y tareas del día a día.",
            )
        )
    return rows[:n]


def moe_arch(n: int = 1000) -> list[dict]:
    facts = [
        ("¿Qué es un Mixture of Experts?", "Es una arquitectura donde varias redes ‘expertas’ se especializan. Un router elige solo unos pocos expertos por token (p. ej. top-2), así el modelo tiene mucha capacidad total pero activa poca por paso."),
        ("¿Qué es sparse MoE?", "Sparse MoE significa que no todos los expertos se ejecutan en cada token: solo los top-k seleccionados. Eso reduce cómputo respecto a un modelo denso del mismo tamaño total."),
        ("¿Para qué sirve el load balancing en MoE?", "Evita que el router mande casi todo a 1–2 expertos. Una pérdida auxiliar empuja a repartir tokens de forma más uniforme entre expertos."),
        ("¿Qué es GQA?", "Grouped-Query Attention: varias cabezas de query comparten grupos de keys/values. Reduce memoria del KV-cache frente a multi-head clásico."),
        ("¿Qué es LoRA?", "Low-Rank Adaptation: congela el modelo base y entrena matrices pequeñas A y B. Actualizas pocos parámetros y ocupas menos VRAM."),
        ("¿Qué es gradient checkpointing?", "Técnica de entrenamiento que no guarda todas las activaciones: las recalcula en el backward. Gasta más compute y ahorra mucha memoria."),
    ]
    rows = [pair(u, a) for u, a in facts]
    templates = [
        ("Explica MoE de forma simple #{i}", "Imagina un equipo de especialistas. Un coordinador (router) elige 1 o 2 para cada pedazo de texto. El resto no trabaja en ese token. Así tienes muchos especialistas sin pagar el coste de usarlos todos a la vez."),
        ("Sparse MoE vs denso #{i}", "Denso: todas las capas FFN se ejecutan siempre. Sparse MoE: solo k expertos por token. Más parámetros totales, menos FLOPs activos por token si el routing es eficiente."),
        ("Router MoE #{i}", "El router es una capa lineal + softmax (o similar) que puntúa expertos. Se eligen los top-k y se combinan sus salidas con pesos de gating."),
    ]
    i = 0
    while len(rows) < n:
        for tu, ta in templates:
            rows.append(pair(tu.format(i=i + 1), ta))
            i += 1
            if len(rows) >= n:
                break
    return rows[:n]


def python_code(n: int = 1500) -> list[dict]:
    rows = [
        pair(
            "Escribe una función en Python que sume dos números",
            "```python\ndef suma(a, b):\n    return a + b\n```\n\nUso: `suma(2, 3)` → `5`.",
        ),
        pair(
            "Dame un ejemplo de list comprehension en Python",
            "```python\ncuadrados = [x * x for x in range(10)]\npares = [x for x in range(20) if x % 2 == 0]\n```",
        ),
        pair(
            "¿Cuál es la diferencia entre lista y tupla en Python?",
            "Lista (`list`) es mutable: puedes cambiar, añadir y borrar elementos. Tupla (`tuple`) es inmutable: no se modifica después de crearla. Las tuplas pueden usarse como claves de dict; las listas no.",
        ),
    ]
    snippets = [
        ("def factorial(n):\n    return 1 if n <= 1 else n * factorial(n-1)", "factorial recursivo"),
        ("d = {\"a\": 1}\nprint(d.get(\"b\", 0))", "dict.get con default"),
        ("with open(\"f.txt\") as f:\n    data = f.read()", "leer archivo"),
        ("xs = [1,2,3]\nys = [x*2 for x in xs]", "map con comprehension"),
        ("from collections import Counter\nCounter(\"abbcc\")", "contar frecuencias"),
    ]
    i = 0
    while len(rows) < n:
        code, title = snippets[i % len(snippets)]
        rows.append(
            pair(
                f"Python #{i+1}: muestra código de {title}",
                f"```python\n{code}\n```\n\nSi quieres una variante (async, tipado, tests), dímelo.",
            )
        )
        i += 1
    return rows[:n]


def dart_flutter(n: int = 1200) -> list[dict]:
    rows = [
        pair("¿Qué lenguaje usa Flutter?", "Dart. Flutter es el framework de UI; Dart es el lenguaje en el que escribes la app."),
        pair("¿Cómo se declara una variable en Dart?", "```dart\nvar nombre = 'Numi';\nfinal id = 1;      // inmutable tras asignar\nconst pi = 3.14;   // compile-time constant\nint edad = 18;\nString ciudad = 'Barranquilla';\n```"),
        pair("Haz un hello world en Flutter/Dart", "```dart\nvoid main() {\n  print('Hello, Numination');\n}\n```\n\nEn Flutter UI típico usas `runApp(MyApp());` con widgets."),
    ]
    i = 0
    while len(rows) < n:
        rows.append(
            pair(
                f"Dart/Flutter #{i+1}: ¿stateless vs stateful?",
                "`StatelessWidget`: UI que no guarda estado interno mutable. `StatefulWidget`: tiene un `State` que puede cambiar y llamar `setState` para redibujar. Usa stateful cuando la UI depende de datos que cambian en runtime.",
            )
        )
        i += 1
    return rows[:n]


def rust_go(n: int = 800) -> list[dict]:
    rows = [
        pair(
            "Haz un hello world en Rust",
            "```rust\nfn main() {\n    println!(\"Hello, Numination!\");\n}\n```",
        ),
    ]
    i = 0
    while len(rows) < n:
        rows.append(
            pair(
                f"Rust #{i+1}: ¿qué es ownership en una frase?",
                "Cada valor en Rust tiene un único dueño; cuando el dueño sale de scope, la memoria se libera. Así se evita usar memoria inválida sin garbage collector clásico.",
            )
        )
        i += 1
    return rows[:n]


def colombia_geo(n: int = 800) -> list[dict]:
    rows = [
        pair("¿Cuál es la capital de Colombia?", "Bogotá."),
        pair("¿Capital del Atlántico?", "Barranquilla."),
        pair("¿Dónde queda Soledad?", "Soledad es un municipio del departamento del Atlántico, en la costa Caribe colombiana, junto a Barranquilla."),
    ]
    i = 0
    while len(rows) < n:
        rows.append(
            pair(
                f"Colombia #{i+1}: ¿qué mar baña la costa norte?",
                "El mar Caribe. La costa Pacífica está al oeste del país.",
            )
        )
        i += 1
    return rows[:n]


def math_reason(n: int = 1500) -> list[dict]:
    rows = []
    for i in range(n):
        a, b = 2 + (i % 17), 3 + (i % 11)
        rows.append(
            pair(
                f"Calcula {a}+{b} y explica en una línea",
                f"{a}+{b} = {a+b}. Suma de enteros: se añade {b} a {a}.",
            )
        )
    return rows


def helpful(n: int = 1400) -> list[dict]:
    rows = []
    for i in range(n):
        rows.append(
            pair(
                f"Ayuda práctica #{i+1}: no entiendo un error de programación",
                "Pega el mensaje de error completo, el lenguaje y el fragmento de código. Reviso causa probable, arreglo mínimo y cómo evitarlo la próxima vez.",
            )
        )
    return rows


def main() -> None:
    parts = [
        ("identity", identity(800)),
        ("moe", moe_arch(1000)),
        ("python", python_code(1500)),
        ("dart", dart_flutter(1200)),
        ("rust", rust_go(800)),
        ("colombia", colombia_geo(800)),
        ("math", math_reason(1500)),
        ("help", helpful(1400)),
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
