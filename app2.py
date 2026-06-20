import os
import streamlit as st
from dotenv import load_dotenv

# 1. Carrega as variáveis de ambiente do arquivo .env (apenas para teste local)
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA

# ---------------------------------------------------------
# 2. Configuração da Página do Streamlit
# ---------------------------------------------------------
st.set_page_config(
    page_title="PowerTech Solutions | Suporte Técnico IA",
    page_icon="⚙️",
    layout="centered",
)

# ---------------------------------------------------------
# 3. Tema Visual Personalizado (CSS da empresa)
# ---------------------------------------------------------
st.markdown(
    """
    <style>
    .stApp {
        background-color: #0F1B2D;
    }
    h1, h2, h3 {
        color: #F2A900 !important;
    }
    .stChatMessage {
        border-radius: 10px;
    }
    [data-testid="stSidebar"] {
        background-color: #142539;
        border-right: 2px solid #F2A900;
    }
    .status-ok {
        color: #2ECC71;
        font-weight: bold;
    }
    .status-erro {
        color: #E74C3C;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# 4. Cabeçalho Estilizado
# ---------------------------------------------------------
st.markdown("## ⚙️ PowerTech Solutions")
st.markdown("### 🤖 Assistente Inteligente de Suporte Técnico")
st.write("Tire dúvidas sobre **alarmes**, **falhas**, **manutenção preventiva** e **procedimentos operacionais** do Inversor de Frequência PTX-FD3000.")
st.divider()

# ---------------------------------------------------------
# 5. Gerenciamento da API Key da NVIDIA
# ---------------------------------------------------------
nvidia_api_key = os.environ.get("NVIDIA_API_KEY")

if not nvidia_api_key:
    try:
        if "NVIDIA_API_KEY" in st.secrets:
            nvidia_api_key = st.secrets["NVIDIA_API_KEY"]
    except Exception:
        pass

if not nvidia_api_key:
    st.info("Por favor, adicione sua NVIDIA_API_KEY no arquivo .env (local) ou nos Secrets do Streamlit (deploy).", icon="🔑")
    st.stop()

# ---------------------------------------------------------
# 6. Inicialização e Cache do Banco de Dados RAG (Vetorização do PDF)
# ---------------------------------------------------------
@st.cache_resource(show_spinner="Processando o manual técnico em PDF...")
def inicializar_rag():
    nome_arquivo_pdf = "manual.pdf"

    if not os.path.exists(nome_arquivo_pdf):
        st.error(f"Arquivo '{nome_arquivo_pdf}' não foi encontrado! Coloque o manual.pdf na raiz do projeto.")
        st.stop()

    loader = PyPDFLoader(nome_arquivo_pdf)
    paginas = loader.load()

    # TextSplitter calibrado para os limites da API da NVIDIA (Máx 512 tokens)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50,
    )
    docs = text_splitter.split_documents(paginas)

    embeddings = NVIDIAEmbeddings(
        model="nvidia/nv-embedqa-e5-v5",
        nvidia_api_key=nvidia_api_key,
        model_type="passage",
    )

    vectorstore = FAISS.from_documents(docs, embedding=embeddings)

    # Retorna o buscador configurado para trazer os 4 blocos mais relevantes
    return vectorstore.as_retriever(search_kwargs={"k": 4})


# Flags de status para exibir na sidebar
manual_carregado = os.path.exists("manual.pdf")
vetorizacao_ok = False
ia_conectada = False

retriever = None
llm = None

try:
    retriever = inicializar_rag()
    vetorizacao_ok = True
except Exception:
    vetorizacao_ok = False

# ---------------------------------------------------------
# 7. Configuração do Modelo de Linguagem (LLM) da NVIDIA
#    Requisitos técnicos: temperature e max_tokens configurados
# ---------------------------------------------------------
try:
    llm = ChatNVIDIA(
        model="meta/llama-3.1-8b-instruct",
        nvidia_api_key=nvidia_api_key,
        temperature=0.2,      # baixa temperatura -> respostas mais técnicas e consistentes
        max_tokens=1024,      # limite de tokens da resposta gerada
    )
    ia_conectada = True
except Exception:
    ia_conectada = False

# ---------------------------------------------------------
# 8. Engenharia de Prompt
#    Persona + Tarefa + Contexto + Restrições
# ---------------------------------------------------------
template_prompt = """
Você é um engenheiro especialista em manutenção industrial da PowerTech Solutions, com profundo
conhecimento sobre o Inversor de Frequência PTX-FD3000.

TAREFA:
Auxiliar técnicos de campo a diagnosticar alarmes, falhas, dúvidas de manutenção preventiva e
procedimentos operacionais do equipamento, com respostas claras, objetivas e tecnicamente corretas.

CONTEXTO:
As informações abaixo (em português) foram extraídas diretamente do manual técnico oficial do
equipamento. Use SOMENTE essas informações para responder — não utilize conhecimento externo
sobre outros equipamentos ou marcas.

RESTRIÇÕES:
- Se a resposta não estiver presente no contexto fornecido, diga claramente:
  "Desculpe, essa informação não consta no manual técnico do equipamento."
- Nunca invente códigos de alarme, valores técnicos ou procedimentos que não estejam no contexto.
- Sempre que aplicável, reforce recomendações de segurança (ex.: desligar e aguardar descarga dos
  capacitores antes de qualquer intervenção).
- Responda sempre em português do Brasil, de forma profissional e direta, como em um chamado de
  suporte técnico real.

Contexto (extraído do manual técnico):
{context}

Pergunta do técnico: {question}
Resposta do engenheiro especialista:
"""
prompt = ChatPromptTemplate.from_template(template_prompt)

# Criação do Pipeline RAG (LangChain Expression Language - LCEL)
if retriever is not None and llm is not None:
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
else:
    rag_chain = None

# ---------------------------------------------------------
# 9. Sidebar — Informações do Projeto e Status
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("## ⚙️ PowerTech Solutions")
    st.markdown("### ℹ️ Informações do Projeto")
    st.write("**Produto:** Inversor de Frequência PTX-FD3000")
    st.write("**Equipe responsável:** Suporte Técnico PowerTech")
    st.write("**Modelo de IA:** meta/llama-3.1-8b-instruct (NVIDIA NIM)")
    st.write("**Embeddings:** nvidia/nv-embedqa-e5-v5")
    st.write("**Banco vetorial:** FAISS")

    st.divider()
    st.markdown("### 📊 Status do Sistema")

    if manual_carregado:
        st.markdown("✅ <span class='status-ok'>Manual carregado</span>", unsafe_allow_html=True)
    else:
        st.markdown("❌ <span class='status-erro'>Manual não encontrado</span>", unsafe_allow_html=True)

    if vetorizacao_ok:
        st.markdown("✅ <span class='status-ok'>Vetorização concluída</span>", unsafe_allow_html=True)
    else:
        st.markdown("❌ <span class='status-erro'>Vetorização pendente/erro</span>", unsafe_allow_html=True)

    if ia_conectada:
        st.markdown("✅ <span class='status-ok'>IA conectada (NVIDIA)</span>", unsafe_allow_html=True)
    else:
        st.markdown("❌ <span class='status-erro'>IA não conectada</span>", unsafe_allow_html=True)

    st.divider()

    # Botão de limpar conversa
    if st.button("🗑️ Limpar Conversa"):
        st.session_state.messages = []
        st.rerun()

if rag_chain is None:
    st.error("Não foi possível inicializar o assistente. Verifique o manual.pdf e a API Key da NVIDIA.")
    st.stop()

# ---------------------------------------------------------
# 10. Histórico do Chat na Interface (Streamlit Session State)
# ---------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Olá! 👷 Sou o assistente de suporte técnico da PowerTech Solutions. "
                "Processei o manual do Inversor de Frequência PTX-FD3000 e estou pronto para ajudar "
                "com alarmes, falhas, manutenção e procedimentos operacionais. O que você deseja saber?"
            ),
        }
    ]

avatares = {"assistant": "⚙️", "user": "🧑‍🔧"}

for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar=avatares.get(message["role"])):
        st.write(message["content"])

# ---------------------------------------------------------
# 11. Fluxo de Interação: Entrada do Usuário e Resposta da IA
# ---------------------------------------------------------
if prompt_usuario := st.chat_input("Ex: O que significa o alarme E004?"):
    st.session_state.messages.append({"role": "user", "content": prompt_usuario})
    with st.chat_message("user", avatar=avatares["user"]):
        st.write(prompt_usuario)

    with st.chat_message("assistant", avatar=avatares["assistant"]):
        with st.spinner("🔧 Consultando manual técnico..."):
            try:
                resposta = rag_chain.invoke(prompt_usuario)
                st.write(resposta)
                st.session_state.messages.append({"role": "assistant", "content": resposta})
            except Exception as e:
                erro_msg = f"⚠️ Erro ao processar a requisição na API da NVIDIA: {e}"
                st.error(erro_msg)
                st.session_state.messages.append({"role": "assistant", "content": erro_msg})