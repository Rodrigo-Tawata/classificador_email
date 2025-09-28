# Classificador de Emails

Um sistema de classificação de emails e resposta automática em português. Ele classifica emails em produtivo ou improdutivo e sugere respostas automáticas apropriadas.

# Funcionalidades

- Classificação de Emails: Determina se o conteúdo de um email é produtivo ou improdutivo.

- Geração de Resposta Automática: Sugere uma resposta adequada com base na categoria do email.

- Interface Web: Disponível via Streamlit para testes rápidos e intuitivos.

# 🛠️ Como Rodar Localmente
-  1º - Clone o repositório:

```
git clone https://github.com/Rodrigo-Tawata/classificador_email.git
cd classificador_email
```

- 2º - Crie e ative um ambiente virtual (opcional, mas recomendado):

```
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows
```

- 3º - Instale as dependências 
```
pip install -r requirements.txt
```

- 4º - Treine o modelo (ou use o modelo pré-treinado)
```
python train.py
```

- 5º - Execute a aplicação
```
streamlit run app.py
```

Abra o navegador no link fornecido pelo Streamlit e teste a classificação de emails.

