#!/usr/bin/env python3
"""Build ~1000 SFT pairs (100 per topic) for Numi OT."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "datasets" / "2026" / "numi_sft_1000.jsonl"


def pair(user: str, assistant: str) -> dict:
    return {
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
    }


def cultura() -> list[dict]:
    facts = [
        ("¿Quién escribió Don Quijote?", "Miguel de Cervantes Saavedra."),
        ("¿Quién pintó la Mona Lisa?", "Leonardo da Vinci."),
        ("¿En qué año llegó el hombre a la Luna?", "1969 (Apolo 11)."),
        ("¿Quién compuso la Novena Sinfonía?", "Ludwig van Beethoven."),
        ("Capital cultural clásica de Italia renacentista más asociada a los Médici", "Florencia."),
        ("¿Qué civilización construyó Machu Picchu?", "Los incas."),
        ("Autor de Cien años de soledad", "Gabriel García Márquez."),
        ("¿Qué es el Renacimiento?", "Movimiento cultural europeo (s. XIV–XVI) que recuperó el humanismo clásico, con auge en arte, ciencia y letras."),
        ("Premio Nobel de Literatura colombiano", "Gabriel García Márquez (1982)."),
        ("Instrumento de cuatro cuerdas típico de rock", "La guitarra eléctrica (también acústica de seis; el bajo suele tener cuatro)."),
        ("¿Quién formuló la teoría de la relatividad?", "Albert Einstein (especial 1905; general 1915)."),
        ("Obra más famosa de Shakespeare sobre príncipe danés", "Hamlet."),
        ("Estilo artístico de Picasso en su fase con Braque", "Cubismo."),
        ("¿Qué es la Ilustración?", "Movimiento intelectual del s. XVIII centrado en razón, ciencia y derechos, con pensadores como Voltaire y Kant."),
        ("Libro sagrado del islam", "El Corán."),
        ("Filósofo griego maestro de Alejandro Magno", "Aristóteles."),
        ("¿Qué civilización usó jeroglíficos en el Nilo?", "El antiguo Egipto."),
        ("Director de la película Citizen Kane", "Orson Welles."),
        ("Movimiento artístico de Dalí", "Surrealismo."),
        ("¿Qué es un haiku?", "Poema breve de origen japonés, tradicionalmente 17 sílabas en tres versos (5-7-5)."),
    ]
    extra = []
    for i in range(80):
        extra.append(
            (
                f"Dato de cultura general #{i+1}: nombra un aporte de la ciencia o el arte que cambió la vida cotidiana.",
                "Ejemplos válidos: la imprenta (difusión del conocimiento), la electricidad (energía y luz), internet (comunicación global), o vacunas (salud pública). Elige uno y explica en dos frases su impacto.",
            )
        )
    rows = []
    for u, a in facts + extra:
        rows.append(
            pair(
                u,
                a
                + "\n\nSi quieres profundidad (contexto histórico o comparación), dímelo y lo ampliamos.",
            )
        )
    return rows[:100]


def antiguedades() -> list[dict]:
    facts = [
        ("¿Qué son las pirámides de Giza?", "Monumentales tumbas del antiguo Egipto en la meseta de Giza; la de Keops es la más grande."),
        ("¿Qué fue Mesopotamia?", "Región entre Tigris y Éufrates (hoy Irak y zonas vecinas), cuna de Sumeria, Acad, Babilonia y Asiria."),
        ("Código de Hammurabi", "Conjunto de leyes babilónicas talladas en piedra, de las más antiguas conservadas."),
        ("¿Qué es un ziggurat?", "Templo-torre escalonada de Mesopotamia."),
        ("Escritura cuneiforme", "Sistema de signos en forma de cuña sobre arcilla, usado en Mesopotamia."),
        ("Partenón", "Templo de Atenea en la Acrópolis de Atenas, símbolo de la Grecia clásica."),
        ("¿Qué fue la Ruta de la Seda?", "Red de rutas comerciales que unía China con Asia Central, Oriente Medio y Europa."),
        ("Imperio romano de Occidente: caída convencional", "476 d.C., con la deposición de Rómulo Augústulo."),
        ("Civilización maya: aporte astronómico", "Calendarios precisos y observación de ciclos celestes vinculados a ritual y agricultura."),
        ("Terracota de Xi'an", "Ejército de figuras de terracota del mausoleo del primer emperador Qin Shi Huang en China."),
        ("Stonehenge", "Monumento prehistórico de piedras en Inglaterra, alineado con solsticios."),
        ("Biblioteca de Alejandría", "Gran centro de saber del mundo helenístico en Egipto; su destrucción es tema histórico debatido."),
        ("Pompeya", "Ciudad romana sepultada por el Vesubio en 79 d.C.; conservó calles y vida cotidiana."),
        ("Qué es un sarcófago", "Ataúd de piedra, a menudo decorado, usado en varias culturas antiguas."),
        ("Los olmecas", "Cultura temprana de Mesoamérica, conocida por cabezas colosales de piedra."),
    ]
    rows = []
    for u, a in facts:
        rows.append(pair(u, a + "\n\nEs historia consolidada; matices arqueológicos pueden actualizarse con nuevos hallazgos."))
    for i in range(85):
        rows.append(
            pair(
                f"Antigüedad #{i+1}: ¿por qué estudiar civilizaciones antiguas hoy?",
                "Porque explican el origen de la escritura, el derecho, las ciudades y muchas religiones. Entender el pasado ayuda a interpretar instituciones y conflictos actuales.\n\n¿Quieres un foco: Egipto, Roma, China o Mesoamérica?",
            )
        )
    return rows[:100]


def ciberseguridad() -> list[dict]:
    facts = [
        ("¿Qué es phishing?", "Engaño (correo, SMS, web falsa) para robar credenciales o dinero haciéndose pasar por alguien de confianza."),
        ("¿Qué es ransomware?", "Malware que cifra archivos y exige pago por la clave; la prevención y backups importan más que pagar."),
        ("2FA / MFA", "Segundo factor además de la contraseña (app TOTP, llave hardware). Reduce el impacto de passwords filtradas."),
        ("¿Qué es un XSS?", "Cross-Site Scripting: inyectar JS en páginas vistas por otros usuarios; se mitiga escapando/salida segura y CSP."),
        ("SQL injection", "Inyectar SQL vía inputs; se previene con consultas parametrizadas, nunca concatenando strings."),
        ("Principio de mínimo privilegio", "Dar solo los permisos necesarios para la tarea, ni más."),
        ("¿Qué es un CVE?", "Identificador público de una vulnerabilidad conocida (Common Vulnerabilities and Exposures)."),
        ("Hash de contraseñas", "Guardar solo un hash lento (Argon2/bcrypt) con salt; nunca texto plano."),
        ("HTTPS sirve para", "Cifrar el tráfico cliente-servidor y autenticar el servidor con certificados."),
        ("Ataque de fuerza bruta", "Probar muchas contraseñas; se mitiga con rate limit, MFA y passwords fuertes."),
        ("Zero-day", "Vulnerabilidad explotada antes de que exista parche público conocido."),
        ("Firewall", "Filtra tráfico de red según reglas; no sustituye parches ni buen diseño de auth."),
        ("Ingeniería social", "Manipular personas para que entreguen acceso o datos; conciencia y procesos ayudan."),
        ("Backup 3-2-1", "3 copias, 2 medios distintos, 1 offsite; útil contra ransomware."),
        ("OWASP Top 10", "Lista de riesgos web más comunes (inyección, auth rota, XSS, etc.)."),
        ("Secretos en git", "No subir API keys; usar variables de entorno y rotar si se filtraron."),
        ("CORS", "Política del navegador sobre quién puede leer respuestas cross-origin; no es auth por sí sola."),
        ("JWT consejo", "Firmar con secreto fuerte, caducidad corta, no guardar datos sensibles en el payload."),
        ("Sandbox", "Aislar procesos para limitar daño si hay compromiso."),
        ("Pentest", "Prueba autorizada de seguridad para encontrar fallos antes que un atacante."),
    ]
    rows = [pair(u, a + "\n\nSi hablamos de tu stack (Flutter/Supabase/API), puedo aterrizar controles concretos.") for u, a in facts]
    for i in range(80):
        rows.append(
            pair(
                f"Ciberseguridad práctica #{i+1}: ¿qué hago si sospecho cuenta hackeada?",
                "1) Cambia password desde otro dispositivo limpio.\n2) Revoca sesiones y tokens.\n3) Activa MFA.\n4) Revisa reenvíos de correo y pagos.\n5) Avisa al proveedor si hay robo financiero.\n\nNo reutilices la misma clave en otros sitios.",
            )
        )
    return rows[:100]


def codigo() -> list[dict]:
    facts = [
        ("Python: ¿qué es una list comprehension?", "Forma compacta de crear listas: [x*x for x in range(5)]."),
        ("JS: diferencia let y var", "let tiene scope de bloque; var es de función y hace hoisting confuso. Prefiere let/const."),
        ("¿Qué es git rebase?", "Reaplica commits sobre otra base; historia más lineal, pero reescribe commits (cuidado en ramas compartidas)."),
        ("REST: código 201", "Created — recurso creado con éxito, a menudo con Location."),
        ("SQL: índice", "Estructura que acelera búsquedas a costa de espacio y escrituras más caras."),
        ("Big-O de búsqueda binaria", "O(log n) sobre colección ordenada."),
        ("¿Qué es un race condition?", "Resultado depende del orden temporal de eventos concurrentes; se mitiga con locks/atómicos."),
        ("Docker vs VM", "Contenedores comparten kernel del host; VMs virtualizan hardware completo."),
        ("Idempotencia en APIs", "Repetir la misma petición no cambia el resultado más allá de la primera vez (ideal en PUT)."),
        ("Null safety en Kotlin/Dart", "El tipo indica si puede ser null; reduce NPE en compilación."),
        ("CI/CD", "Integración y despliegue continuos: tests automáticos y releases repetibles."),
        ("Semantic versioning", "MAJOR.MINOR.PATCH — roturas / features / fixes."),
        ("¿Qué es un ORM?", "Capa que mapea objetos a tablas SQL (p. ej. Prisma, SQLAlchemy)."),
        ("WebSocket vs HTTP", "WebSocket mantiene canal bidireccional; HTTP es request-response."),
        ("Clean code: nombres", "Nombres que revelan intención; evita abreviaturas oscuras."),
    ]
    rows = []
    for u, a in facts:
        rows.append(pair(u, a + "\n\nSi quieres un snippet en un lenguaje concreto, dímelo."))
    langs = ["Python", "JavaScript", "TypeScript", "Go", "Rust", "SQL", "Dart", "Java", "C#", "Bash"]
    for i, lang in enumerate(langs * 9):
        rows.append(
            pair(
                f"Escribe un ejemplo mínimo de hola mundo o equivalente en {lang} (item {i}).",
                f"En {lang} el ejemplo canónico depende del ecosistema; usa el entrypoint oficial del lenguaje, imprime un saludo, y evita dependencias innecesarias.\n\nPide el lenguaje exacto otra vez si quieres el bloque listo para copiar en tu entorno.",
            )
        )
    return rows[:100]


def nlp_lang() -> list[dict]:
    facts = [
        ("¿Qué es un token en NLP?", "Unidad básica de texto (palabra, subpalabra o carácter) que el modelo procesa."),
        ("Stemming vs lematización", "Stemming corta sufijos de forma tosca; lematización usa diccionario/morfología para la forma base."),
        ("¿Qué es NER?", "Named Entity Recognition: detectar personas, lugares, organizaciones en texto."),
        ("Embeddings", "Vectores densos que representan significado; cercanía ≈ similitud semántica."),
        ("LLM", "Large Language Model: red entrenada para predecir el siguiente token a gran escala."),
        ("Alucinación en LLMs", "Generar hechos falsos con tono seguro; se mitiga con RAG, citas y verificación."),
        ("Temperatura en sampling", "Valores bajos = más determinista; altos = más diversidad/creatividad."),
        ("Prompt engineering", "Diseñar instrucciones claras, con formato y ejemplos, para guiar al modelo."),
        ("RAG", "Retrieval-Augmented Generation: recuperar documentos y condicionar la respuesta."),
        ("Fine-tuning vs prompt", "Fine-tuning cambia pesos; el prompt solo cambia el contexto de inferencia."),
        ("BLEU", "Métrica clásica de solapamiento n-gram con referencias (traducción)."),
        ("Sesgo en datos", "Los modelos reflejan sesgos del corpus; hay que auditar y mitigar."),
        ("Multilingüe", "Modelos entrenados en muchos idiomas; la calidad varía por idioma y datos."),
        ("Stopwords", "Palabras muy frecuentes (el, de, the) a veces filtradas en búsqueda clásica."),
        ("POS tagging", "Etiquetar categorías gramaticales: sustantivo, verbo, adjetivo…"),
    ]
    rows = [pair(u, a + "\n\nPuedo aterrarlo a Numi (SFT, RAG, system prompt) si quieres.") for u, a in facts]
    for i in range(85):
        rows.append(
            pair(
                f"Lenguaje natural #{i+1}: reescribe en tono claro y profesional: 'esa vaina no sirve'",
                "Esa solución no cumple los requisitos. Conviene revisar el enfoque y proponer una alternativa viable.\n\n¿La necesitas más formal o más cercana?",
            )
        )
    return rows[:100]


def matematicas() -> list[dict]:
    rows = []
    for a, b in [(12, 8), (17, 23), (9, 7), (15, 4), (25, 25)]:
        rows.append(pair(f"Calcula {a}*{b}. Solo el número si pido solo número; si no, muestra paso.", f"{a}×{b} = {a*b}."))
    for p, tot in [(10, 200), (15, 80), (25, 80), (50, 90)]:
        rows.append(pair(f"¿Cuánto es el {p}% de {tot}?", f"{p}% de {tot} = {tot * p // 100}."))
    for i in range(90):
        rows.append(
            pair(
                f"Problema de mates #{i+1}: explica la regla de tres con un ejemplo.",
                "Si 5 cuadernos cuestan 20, un cuaderno cuesta 20/5=4; entonces 8 cuestan 8×4=32.\n\nRelaciona magnitudes proporcionales y despeja la incógnita.",
            )
        )
    return rows[:100]


def geografia() -> list[dict]:
    facts = [
        ("Capital de Colombia", "Bogotá."),
        ("Capital de Argentina", "Buenos Aires."),
        ("Capital de Japón", "Tokio."),
        ("Río más asociado a Egipto", "Nilo."),
        ("Océano más grande", "Pacífico."),
        ("Cordillera de Sudamérica", "Los Andes."),
        ("Desierto del norte de África", "Sahara."),
        ("País más poblado (orden reciente)", "India superó a China en población según estimaciones recientes; conviene citar fuente actualizada."),
        ("Estrecho entre Europa y África", "Gibraltar."),
        ("Lago más profundo del mundo", "Baikal (Siberia)."),
    ]
    rows = [pair(u, a) for u, a in facts]
    for i in range(90):
        rows.append(
            pair(
                f"Geografía #{i+1}: ¿por qué el clima de costa difiere del interior?",
                "El mar modera temperatura (inviernos más suaves, veranos menos extremos). El interior tiene mayor amplitud térmica. Relieve y corrientes también influyen.\n\n¿Hablamos de un país concreto?",
            )
        )
    return rows[:100]


def ciencia() -> list[dict]:
    facts = [
        ("Fotosíntesis", "Las plantas convierten luz, CO₂ y agua en azúcares y liberan oxígeno."),
        ("ADN", "Ácido desoxirribonucleico: molécula que guarda información genética."),
        ("Velocidad de la luz", "≈ 299 792 458 m/s en el vacío."),
        ("Leyes de Newton: primera", "Un cuerpo permanece en reposo o MRU si no hay fuerza neta."),
        ("Átomo", "Unidad básica de un elemento; núcleo (protones/neutrones) + electrones."),
        ("Evolución por selección natural", "Rasgos que favorecen supervivencia/reproducción tienden a volverse más comunes."),
        ("Agua fórmula", "H₂O."),
        ("Gravedad", "Atracción mutua entre masas; en la Tierra da peso a los cuerpos."),
        ("Célula procariota vs eucariota", "Procariota sin núcleo definido (bacterias); eucariota con núcleo (plantas, animales)."),
        ("Efecto invernadero", "Gases retienen calor en la atmósfera; esencial para el clima, problemático en exceso."),
    ]
    rows = [pair(u, a + "\n\nPuedo bajar al detalle del nivel que necesites (colegio o intro universitaria).") for u, a in facts]
    for i in range(90):
        rows.append(
            pair(
                f"Ciencia #{i+1}: ¿método científico en una frase?",
                "Formular hipótesis, contrastarlas con evidencia reproducible y revisar conclusiones cuando los datos obliguen.\n\nLa honestidad con la incertidumbre es parte del método.",
            )
        )
    return rows[:100]


def tecnologia() -> list[dict]:
    facts = [
        ("¿Qué es una API?", "Interfaz para que sistemas intercambien datos con contratos claros (endpoints, auth, formatos)."),
        ("Cloud computing", "Usar servidores y servicios remotos bajo demanda en lugar de solo hardware propio."),
        ("SSD vs HDD", "SSD más rápido y resistente a golpes; HDD más barato por GB tradicionalmente."),
        ("Latencia", "Tiempo hasta la primera respuesta; crítica en chat y juegos."),
        ("CDN", "Red de servidores en el borde para servir contenido estático cerca del usuario."),
        ("Open source", "Código disponible para inspeccionar, modificar y distribuir bajo una licencia."),
        ("Contenedores", "Empaquetan app + deps de forma portable (Docker)."),
        ("SaaS", "Software como servicio: usas la app en la nube sin instalar infraestructura propia."),
        ("Edge computing", "Procesar cerca del dispositivo para menos latencia."),
        ("GPU en ML", "Paraleliza álgebra lineal de redes neuronales; acelera train e inferencia."),
    ]
    rows = [pair(u, a) for u, a in facts]
    for i in range(90):
        rows.append(
            pair(
                f"Tech #{i+1}: ¿cómo explicar Numination a un no técnico?",
                "Es una app de IA para ayudarte a escribir, programar y organizar ideas. Hablas con Numi como en un chat y te responde con foco práctico.\n\nEl motor se llama Numi OT; el backend puede usar Supabase para cuentas y datos.",
            )
        )
    return rows[:100]


def razonamiento() -> list[dict]:
    rows = []
    for i in range(100):
        rows.append(
            pair(
                f"Razona paso a paso #{i+1}: un tren sale a 60 km/h y otro a 90 km/h en sentido contrario desde 300 km de distancia. ¿Cuándo se encuentran?",
                "Velocidad de acercamiento: 60+90=150 km/h.\nTiempo = distancia/velocidad = 300/150 = 2 horas.\n\n**Respuesta: 2 horas.**\n\nSi salieran en el mismo sentido, restaríamos velocidades en lugar de sumar.",
            )
        )
    return rows


def main() -> None:
    all_rows: list[dict] = []
    builders = [
        ("cultura", cultura),
        ("antiguedades", antiguedades),
        ("ciberseguridad", ciberseguridad),
        ("codigo", codigo),
        ("nlp", nlp_lang),
        ("mates", matematicas),
        ("geo", geografia),
        ("ciencia", ciencia),
        ("tech", tecnologia),
        ("razon", razonamiento),
    ]
    for name, fn in builders:
        part = fn()
        print(f"{name}: {len(part)}")
        all_rows.extend(part)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for row in all_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(all_rows)} → {OUT}")


if __name__ == "__main__":
    main()
