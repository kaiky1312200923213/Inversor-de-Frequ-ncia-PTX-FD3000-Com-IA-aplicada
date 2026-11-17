# PowerTech Solutions — Assistente técnico para PTX-FD3000

**Autor:** Kaiky Yoshio Ono Marins  
**Portfólio:** [kaikysec.vercel.app](https://kaikysec.vercel.app/)  
**Categoria:** Python · Streamlit · RAG · Suporte técnico  
**Status:** Protótipo de estudo

## Visão geral

Assistente de suporte técnico em português para consulta ao manual do inversor de frequência PTX-FD3000. A aplicação utiliza uma abordagem RAG: recupera trechos relevantes do manual e os fornece como contexto para ajudar a responder dúvidas sobre alarmes, falhas, manutenção preventiva e procedimentos operacionais.

## Problema e proposta

Localizar uma informação específica em um manual técnico pode exigir a leitura de várias páginas. O protótipo explora a recuperação semântica para tornar a consulta ao documento mais direta.

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

## Segurança e limitações

Respostas geradas por IA podem conter erros ou deixar de recuperar informações relevantes. Confira qualquer orientação no manual oficial e com profissionais qualificados. Não utilize o protótipo como única fonte para decisões de operação, manutenção ou segurança industrial.

Nunca publique chaves de acesso ou arquivos `.env`.

## O que este projeto demonstra

- Aplicação de RAG a um manual técnico específico.
- Integração de busca semântica com um modelo de linguagem.
- Organização de uma interface de consulta para um domínio técnico.

© 2026 Kaiky Yoshio Ono Marins.
