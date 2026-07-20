# 🤖 BimBam Buy — Asistente Virtual Inteligente

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=flat&logo=langchain)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat&logo=docker&logoColor=white)

Un chatbot corporativo potenciado por Inteligencia Artificial diseñado para responder automáticamente las preguntas más frecuentes de los colaboradores de **BimBam Buy**, una tienda online de confianza. 

El agente utiliza **RAG (Retrieval-Augmented Generation)** para buscar información de manera semántica en los documentos oficiales de la empresa y generar respuestas precisas, contextualizadas y actualizadas.

---

## 🏗️ Arquitectura de la Solución

```text
┌──────────────────┐      ┌──────────────────┐      ┌──────────────────┐
│                  │      │                  │      │                  │
│    Streamlit     │─────►│     FastAPI      │─────►│  LangChain RAG   │
│    (Frontend)    │ HTTP │    (Backend)     │      │   (Pipeline)     │
│                  │      │                  │      │                  │
└──────────────────┘      └──────────────────┘      └────────┬─────────┘
                                                             │
                                              ┌──────────────┴──────────────┐
                                              │                             │
                                     ┌────────▼────────┐          ┌─────────▼───────┐
                                     │                 │          │                 │
                                     │    ChromaDB     │          │  Groq + GPT-OSS │
                                     │  (Vector Store) │          │     (120B)      │
                                     │                 │          │                 │
                                     └─────────────────┘          └─────────────────┘
```

### Flujo de trabajo
1. El colaborador escribe una consulta en la interfaz de chat.
2. El sistema vectoriza la pregunta y busca los fragmentos más relevantes en la base de datos documental.
3. El LLM (Large Language Model) genera una respuesta conversacional basada estrictamente en la información encontrada.
4. Se muestra la respuesta final junto con las fuentes consultadas para garantizar la trazabilidad.

---

## 💻 Stack Tecnológico

| Componente | Tecnología | Propósito |
|---|---|---|
| **Lenguaje** | Python 3.12 | Base y lógica del proyecto. |
| **Frontend** | Streamlit | Interfaz web interactiva e intuitiva para el chat. |
| **Backend** | FastAPI | Creación de la API REST de alta velocidad. |
| **Motor RAG** | LangChain | Orquestación del pipeline entre búsqueda y generación. |
| **Base de Datos** | ChromaDB | Almacenamiento y recuperación eficiente de vectores. |
| **Embeddings** | HuggingFace | Transformación de texto en representaciones vectoriales. |
| **IA Generativa**| Groq (GPT-OSS-120B)| Inferencia ultrarrápida para la generación de respuestas. |
| **Deploy** | Docker + Oracle Cloud | Contenerización y despliegue en infraestructura cloud. |

---

## 🚀 Guía de Instalación y Ejecución

### Requisitos Previos
* [Git](https://git-scm.com/)
* [Docker](https://www.docker.com/) y Docker Compose (Recomendado)
* [Python 3.12+](https://www.python.org/) (Si decides ejecutarlo localmente)
* Una API Key gratuita de [Groq](https://console.groq.com/keys)

### Opción 1: Con Docker (Recomendado)

1. **Clonar el repositorio:**
   ```bash
   git clone https://github.com/NicolasParadaA/challenge-alura-one.git
   cd challenge-alura-one
   ```

2. **Configurar variables de entorno:**
   ```bash
   cp .env.example .env
   ```
   *Abre el archivo `.env` y añade tu `GROQ_API_KEY`.*

3. **Construir y levantar los contenedores:**
   ```bash
   docker-compose up -d --build
   ```
4. **Acceso:** Abre `http://localhost:8501` en tu navegador.

### Opción 2: Ejecución Local (Sin Docker)

1. **Instalar dependencias:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Indexar los documentos en ChromaDB:**
   ```bash
   python ingest.py
   ```
3. **Ejecutar el Backend (Terminal 1):**
   ```bash
   uvicorn api:app --reload --port 8000
   ```
4. **Ejecutar el Frontend (Terminal 2):**
   ```bash
   streamlit run app.py
   ```

---

## 📚 Base Documental

El agente ha sido entrenado e indexado utilizando los siguientes documentos oficiales de BimBam Buy:

| Documento | Tema Principal |
|---|---|
| 📋 **Política de reembolsos** | Procesos de devoluciones y reembolsos. |
| 💳 **Métodos de pago** | Opciones aceptadas y preguntas frecuentes. |
| 🚚 **Guía de envíos** | Tiempos de entrega y costos por zona geográfica. |
| 🛡️ **Manual de garantía** | Períodos de cobertura y resolución de fallas. |
| 🤝 **Programa de afiliados** | Reglas de comisiones y campañas permitidas. |

---

## 💬 Ejemplos de Interacción

<details>
<summary><strong>📦 Pregunta: "¿Cuánto cuesta el envío?"</strong> <i>(Haz clic para expandir)</i></summary>
<br>

> El costo del envío no es un valor fijo, sino que se determina en función de varios factores:
>
> - Destino del pedido (país, región y ciudad).
> - Peso volumétrico y tamaño del paquete.
> - Servicio de entrega seleccionado (estándar, exprés, etc.).
> - Promociones vigentes y umbral de envío gratis que pueda aplicar en el país o campaña.
>
> El importe final del envío se muestra antes de confirmar la compra, de modo que el cliente siempre sabe cuánto pagará.
>
> 📄 **Fuente:** `guia-de-tiempos-y-costos-de-envio-de-bimbambuy.pdf`

</details>

<details>
<summary><strong>🔄 Pregunta: "¿Cuál es la política de devoluciones?"</strong> <i>(Haz clic para expandir)</i></summary>
<br>

> **Política de devoluciones de BimBam Buy**
>
> **Momento de la cancelación:**
> - Si la orden se cancela antes del despacho, normalmente no se genera costo de envío.
> - Si la orden ya fue despachada o entregada, los costos logísticos pueden considerarse al resolver la devolución.
>
> **Reembolsos por garantía:**
> - Cuando el caso es por falla de producto, primero se verifica el Manual de Garantía de Productos.
> - Si la garantía aplica, el reembolso se procesa según lo indicado en el manual.
>
> 📄 **Fuente:** `politica-de-reembolsos-y-devoluciones-de-bimbambuy.pdf`

</details>

<details>
<summary><strong>💳 Pregunta: "¿Qué métodos de pago aceptan?"</strong> <i>(Haz clic para expandir)</i></summary>
<br>

> Los métodos de pago que BimBam Buy acepta son:
>
> - Tarjeta de crédito y débito.
> - Transferencia bancaria.
> - Pago en efectivo en los puntos habilitados.
> - Billeteras digitales (según el país).
> - Cuotas o financiamiento, cuando corresponda.
>
> *Nota: la disponibilidad de cada método puede variar según el país, la categoría del producto y el monto de la compra.*
>
> 📄 **Fuente:** `preguntas-frecuentes-sobre-metodos-de-pago-de-bimbambuy.pdf`

</details>

<details>
<summary><strong>📈 Pregunta: "¿Cómo funciona el programa de afiliados?"</strong> <i>(Haz clic para expandir)</i></summary>
<br>

> **Funcionamiento del Programa de Afiliados de BimBam Buy**
>
> - **Participantes:** Solo afiliados aprobados que operen en los países habilitados de LATAM.
> - **Generación de comisión:** Se genera cuando la compra se origina a través del enlace del afiliado, el pedido queda confirmado, el pago es aprobado y cumple las condiciones.
> - **Actividades permitidas:** Promoción digital, contenidos editoriales y campañas utilizando el código asignado.
>
> 📄 **Fuente:** `programa-de-afiliados-de-bimbambuy.pdf`

</details>

---

## ☁️ Despliegue (Deployment)

La aplicación se encuentra desplegada en producción a través de **Oracle Cloud Infrastructure (OCI)**.

**Especificaciones del servidor:**
* **Proveedor:** Oracle Cloud Infrastructure (OCI)
* **Instancia:** VM.Standard.A1.Flex (Arquitectura ARM64)
* **Recursos:** 1 OCPU, 6GB RAM
* **Sistema Operativo:** Ubuntu 24.04