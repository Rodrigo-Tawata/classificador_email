# app.py
import os
import joblib
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename

# Constantes
UPLOAD_FOLDER = "uploads"
MODEL_PATH = "models/pipeline.pkl"
ALLOWED_EXTENSIONS = {"txt", "pdf"}

# Garante que a pasta de uploads existe
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Inicializa Flask
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

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

# Gera resposta automática baseada na categoria
def gerar_resposta(categoria, texto):
    cat = categoria.lower()
    if "produtivo" in cat:
        return (
            "Recebemos sua solicitação e já encaminhamos ao time responsável. "
            "Por favor, aguarde nossa resposta em até 48 horas. "
            "Se precisar, informe o número do chamado ou mais detalhes."
        )
    else:
        return "Obrigado pelo contato! Sua mensagem foi recebida."

@app.route("/", methods=["GET", "POST"])
def index():
    resultado = None
    resposta = None
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
        else:
            if model:
                pred = model.predict([email_text])[0]
                resultado = pred
                try:
                    proba = model.predict_proba([email_text])[0]
                    confiança = float(max(proba))
                except Exception:
                    confiança = None
                resposta = gerar_resposta(resultado, email_text)
            else:
                # fallback simples
                keywords = ["preciso", "erro", "ajuda", "solicito", "contrato", "documento", "recurso", "suporte", "cancel"]
                if any(k in email_text.lower() for k in keywords):
                    resultado = "produtivo"
                else:
                    resultado = "improdutivo"
                resposta = gerar_resposta(resultado, email_text)

    return render_template("index.html", categoria=resultado, resposta=resposta, confiança=confiança)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
