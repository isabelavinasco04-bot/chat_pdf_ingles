# Interfaz de pregunta
st.subheader("Escribe qué quieres saber sobre el documento")
user_question = st.text_area(" ", placeholder="Escribe tu pregunta aquí...")

# Selector de idioma de respuesta
response_lang = st.selectbox("Idioma de respuesta", ["Inglés 🇬🇧", "Italiano 🇮🇹"], index=0)
lang_instruction = "Please answer in English." if response_lang == "Inglés 🇬🇧" else "Rispondi in italiano."

# Procesar pregunta
if user_question:
    docs = knowledge_base.similarity_search(user_question)
    
    # Modelo de lenguaje
    llm = OpenAI(temperature=0, model_name="gpt-4o")

    # Cargar cadena de QA
    chain = load_qa_chain(llm, chain_type="stuff")
    
    # Agregar instrucción de idioma
    full_question = f"{user_question}\n\n{lang_instruction}"

    # Ejecutar el chain
    response = chain.run(input_documents=docs, question=full_question)

    # Mostrar resultado
    st.markdown("### Respuesta:")
    st.markdown(response)
🌍 ¿Qué hace esto?
Muestra un menú con las opciones “Inglés 🇬🇧” o “Italiano 🇮🇹”

Según la opción, el prompt agrega automáticamente:

"Please answer in English." o

"Rispondi in italiano."

El modelo (GPT-4o) responderá en ese idioma, sin importar el idioma del PDF o de la pregunta.

¿Quieres que además te lo deje con banderitas en los botones o color rosado en el selector para mantener la estética de tus otras apps? 🎀







Tú dijiste:
pero dame todo el codigo


ChatGPT Plus
Perfecto Isa 💖✨ aquí tienes todo tu código completo, ya actualizado para que:

La interfaz siga en español,

Pero el modelo responda en inglés o en italiano, según lo que elijas en el menú,

Todo perfectamente integrado y con tu estructura original intacta.

🎧 Chat PDF Multilenguaje (Español UI, inglés/italiano output)
python
Copiar código
import os
import streamlit as st
from PIL import Image
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
import platform

# ---------- Configuración de la App ----------
st.set_page_config(page_title="RAG Multilenguaje 💬", page_icon="📚", layout="centered")

# Título e info general
st.title('Generación Aumentada por Recuperación (RAG) 💬')
st.write("Versión de Python:", platform.python_version())

# Imagen decorativa
try:
    image = Image.open('Chat_pdf.png')
    st.image(image, width=350)
except Exception as e:
    st.warning(f"No se pudo cargar la imagen: {e}")

# Sidebar informativa
with st.sidebar:
    st.subheader("Asistente de análisis de PDF")
    st.write("Este agente te ayudará a realizar análisis sobre el PDF cargado.")
    st.caption("💡 Ahora puedes elegir que las respuestas sean en inglés o italiano.")

# Clave API
ke = st.text_input('Ingresa tu Clave de OpenAI', type="password")
if ke:
    os.environ['OPENAI_API_KEY'] = ke
else:
    st.warning("Por favor ingresa tu clave de API de OpenAI para continuar")

# Carga del PDF
pdf = st.file_uploader("Carga el archivo PDF", type="pdf")

# Procesamiento del PDF
if pdf is not None and ke:
    try:
        # Extraer texto del PDF
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        
        st.info(f"Texto extraído: {len(text)} caracteres")
        
        # Dividir texto en fragmentos
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=500,
            chunk_overlap=20,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        st.success(f"Documento dividido en {len(chunks)} fragmentos")
        
        # Crear embeddings y base de conocimiento
        embeddings = OpenAIEmbeddings()
        knowledge_base = FAISS.from_texts(chunks, embeddings)
        
        # Sección de pregunta
        st.subheader("Escribe qué quieres saber sobre el documento")
        user_question = st.text_area(" ", placeholder="Escribe tu pregunta aquí...")

        # Selección de idioma de respuesta
        response_lang = st.selectbox("Idioma de respuesta", ["Inglés 🇬🇧", "Italiano 🇮🇹"], index=0)
        lang_instruction = "Please answer in English." if response_lang == "Inglés 🇬🇧" else "Rispondi in italiano."

        # Procesar pregunta
        if user_question:
            docs = knowledge_base.similarity_search(user_question)
            
            # Crear modelo (GPT-4o o compatible)
            llm = OpenAI(temperature=0, model_name="gpt-4o")

            # Cargar cadena QA
            chain = load_qa_chain(llm, chain_type="stuff")
            
            # Agregar instrucción de idioma al prompt
            full_question = f"{user_question}\n\n{lang_instruction}"

            with st.spinner("Analizando el documento... ✨"):
                response = chain.run(input_documents=docs, question=full_question)

            # Mostrar respuesta
            st.markdown("### Respuesta:")
            st.markdown(response)
                
    except Exception as e:
        st.error(f"Error al procesar el PDF: {str(e)}")
        import traceback
        st.error(traceback.format_exc())

elif pdf is not None and not ke:
    st.warning("Por favor ingresa tu clave de API de OpenAI para continuar")
else:
    st.info("Por favor carga un archivo PDF para comenzar")
