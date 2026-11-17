# PowerTech Solutions — Suporte Técnico com IA para PTX-FD3000

**Autor:** Kaiky Yoshio Ono Marins  
**Categoria:** Python · Streamlit · RAG · Suporte técnico  
**Status:** Protótipo de estudo

## Visão geral

Assistente de suporte técnico em português para consulta ao manual do inversor de frequência PTX-FD3000. A aplicação usa uma abordagem RAG: recupera trechos relevantes do manual e os utiliza como contexto para responder dúvidas sobre alarmes, falhas, manutenção preventiva e procedimentos operacionais.

## Recursos implementados

- Interface web em Streamlit.
- Leitura do manual em PDF.
- Divisão do documento em trechos e vetorização.
- Recuperação semântica com FAISS.
- Embeddings NVIDIA `nv-embedqa-e5-v5`.
- Modelo `meta/llama-3.1-8b-instruct` via NVIDIA NIM.
- Indicadores de carregamento do manual, vetorização e conexão com a IA.

## Tecnologias

- Python e Streamlit
- LangChain
- PyPDF
- FAISS
- NVIDIA NIM

## Como executar

1. Instale as dependências listadas em `requirements.txt`.
2. Configure `NVIDIA_API_KEY` localmente em `.env` ou nos Secrets do Streamlit.
3. Execute `streamlit run app2.py`.
4. Mantenha o manual PDF disponível no caminho esperado pelo aplicativo.

**Observação:** respostas geradas por IA devem ser conferidas com o manual oficial e por profissionais qualificados. Não utilize o protótipo como única fonte para decisões de operação ou segurança industrial.

© 2026 Kaiky Yoshio Ono Marins.