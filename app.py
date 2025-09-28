# app.py
import os
import joblib
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename

# OpenAI
from openai import OpenAI

# Constantes
UPLOAD_FOLDER = "uploads"
MODEL_PATH = "models/pipeline.pkl"
ALLOWED_EXTENSIONS = {"txt", "pdf"}
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")  # defina sua chave no ambiente

# Garante que a pasta de uploads existe
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Inicializa Flask
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Inicializa cliente OpenAI
client = OpenAI(api_key=OPENAI_API_KEY)

# Carregar modelo se existir
model = None
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    print(f"Modelo carregado de {MODEL_PATH}")
else:
    print("Aviso: modelo não encontrado em", MODEL_PATH)

# Função para leitura de PDF
def read_pdf(path):
    try:
        from PyPDF2 import PdfReader
        reader = PdfReader(path)
        text = ""
        for p in reader.pages:
            text += p.extract_text() or ""
        return text
    except Exception as e:
        print("Erro ao ler PDF:", e)
        return ""

# Verifica se a extensão é permitida
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# Gera resposta automática baseada na IA com logs detalhados
def gerar_resposta_ia(categoria, texto):
    """
    Gera resposta usando OpenAI GPT.
    Se houver erro, retorna mensagem neutra e registra o motivo.
    Retorna também se a resposta veio da IA ou do fallback.
    """
    prompt = (
        f"Você é um assistente que responde emails de clientes. "
        f"A categoria do email é '{categoria}'. "
        f"Responda de forma clara, profissional e adequada ao contexto do email:\n\n"
        f"Email:\n{texto}\n\nResposta:"
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=200
        )
        resposta_ia = response.choices[0].message.content.strip()
        origem = "IA"
        return resposta_ia, origem, None  # nenhum erro
    except Exception as e:
        # Captura e imprime detalhes completos do erro
        print("Erro ao gerar resposta com IA:", repr(e))
        origem = "Fallback"
        mensagem_erro = repr(e)  # detalhe do erro
        return "Recebemos seu e-mail. Não foi possível gerar a resposta automática.", origem, mensagem_erro

@app.route("/", methods=["GET", "POST"])
def index():
    resultado = None
    resposta = None
    origem = None
    mensagem_erro = None
    confiança = None

    if request.method == "POST":
        # Texto digitado no formulário
        email_text = request.form.get("email_text", "").strip()

        # Arquivo enviado
        file = request.files.get("file")
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(path)
            ext = filename.rsplit(".", 1)[1].lower()
            if ext == "txt":
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    email_text = f.read()
            elif ext == "pdf":
                email_text = read_pdf(path)

        # Classificação
        if not email_text:
            resultado = "Nenhum texto fornecido."
            resposta = "Envie algum texto ou arquivo para processar."
            origem = "Nenhum"
        else:
            if model:
                pred = model.predict([email_text])[0]
                resultado = pred
                try:
                    proba = model.predict_proba([email_text])[0]
                    confiança = float(max(proba))
                except Exception:
                    confiança = None
                # Gera resposta com IA
                resposta, origem, mensagem_erro = gerar_resposta_ia(resultado, email_text)
            else:
                # fallback simples de classificação
                keywords = ["preciso", "erro", "ajuda", "solicito", "contrato", "documento", "recurso", "suporte", "cancel"]
                if any(k in email_text.lower() for k in keywords):
                    resultado = "produtivo"
                else:
                    resultado = "improdutivo"
                resposta, origem, mensagem_erro = gerar_resposta_ia(resultado, email_text)

    return render_template(
        "index.html",
        categoria=resultado,
        resposta=resposta,
        origem_resposta=origem,
        mensagem_erro=mensagem_erro,
        confiança=confiança
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
