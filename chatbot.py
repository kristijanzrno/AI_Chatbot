import aiml
from flask import Flask, render_template, request

kernel = aiml.Kernel()
kernel.setTextEncoding(None)
kernel.bootstrap(learnFiles="rules.xml")

app = Flask(__name__)

@app.route("/")
def home():
   return render_template("index.html")

@app.route("/get")
def get_bot_response():
    textInput = request.args.get('msg')
    print(textInput)
    return str(kernel.respond(textInput))

if __name__ == "__main__":
    app.run()
