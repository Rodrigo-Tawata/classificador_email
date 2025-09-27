from huggingface_hub import hf_hub_download
import gradio as gr
import joblib
import os
import pdfplumber

# Repo do Hugging Face Hub onde está o modelo
REPO_ID = "RodrigoTKT/desafio-autou"
FILENAME = "pipeline.pkl"

# Caminho local do modelo
MODEL_PATH = f"models/{FILENAME}"
os.makedirs("models", exist_ok=True)

# Tenta carregar o modelo local; se não existir, baixa do Hub
model = None
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    try:
        print("Baixando modelo do Hugging Face Hub...")
        model_path_hf = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
        model = joblib.load(model_path_hf)
        # opcional: salvar localmente para uso futuro
        joblib.dump(model, MODEL_PATH)
        print("Modelo carregado com sucesso!")
    except Exception as e:
        print("Não foi possível carregar o modelo:", e)

# Função para gerar resposta automática
def gerar_resposta(categoria, texto):
    cat = categoria.lower()
    if 'produtivo' in cat:
        return (
            "Recebemos sua solicitação e já encaminhamos ao time responsável. "
            "Por favor, aguarde nossa resposta em até 48 horas. Se precisar, informe o número do chamado ou mais detalhes."
        )
    else:
        return "Obrigado pelo contato! Sua mensagem foi recebida."

# Função principal para Gradio (texto ou arquivo)
def analisar_email(email_text, email_file):
    text_content = ""
    
    # Se houver upload de arquivo, processa ele
    if email_file is not None:
        filename = email_file.name
        if filename.endswith(".pdf"):
            with pdfplumber.open(email_file) as pdf:
                text_content = "\n".join([page.extract_text() or "" for page in pdf.pages])
        elif filename.endswith(".txt"):
            text_content = email_file.read().decode("utf-8")
        else:
            return "Formato de arquivo não suportado", ""
    else:
        text_content = email_text

    if not text_content.strip():
        return "Nenhum texto fornecido", ""

    # Classificação
    if model:
        categoria = model.predict([text_content])[0]
        try:
            proba = model.predict_proba([text_content])[0]
            confianca = float(max(proba))
        except Exception:
            confianca = None
        resposta = gerar_resposta(categoria, text_content)
        if confianca:
            categoria += f' (confiança: {confianca*100:.2f}%)'
        return categoria, resposta
    else:
        # fallback simples baseado em palavras-chave
        keywords = ['preciso','erro','ajuda','solicito','contrato','documento','recurso','suporte','cancel']
        if any(k in text_content.lower() for k in keywords):
            categoria = 'produtivo'
        else:
            categoria = 'improdutivo'
        resposta = gerar_resposta(categoria, text_content)
        return categoria, resposta

# Interface Gradio
with gr.Blocks() as demo:
    gr.Markdown("# AutoU — Classificação e Resposta Automática de Emails")
    with gr.Row():
        with gr.Column():
            email_input = gr.Textbox(label="Cole o texto do email", placeholder="Digite ou cole o email aqui...")
            email_file = gr.File(label="Ou envie um arquivo (.pdf, .txt)", file_types=[".pdf", ".txt"])
            analyze_button = gr.Button("Analisar Email")
        with gr.Column():
            categoria_output = gr.Textbox(label="Categoria")
            resposta_output = gr.Textbox(label="Resposta Sugerida")

    analyze_button.click(fn=analisar_email, inputs=[email_input, email_file], outputs=[categoria_output, resposta_output])

if __name__ == '__main__':
    demo.launch()
