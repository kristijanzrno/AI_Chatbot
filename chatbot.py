import aiml
from flask import Flask, render_template, request

kernel = aiml.Kernel()
kernel.setTextEncoding(None)
kernel.bootstrap(learnFiles='rules.xml')

app = Flask(__name__)

@app.route('/')
def home():
   return render_template('index.html')

@app.route('/get')
def get_bot_response():
    textInput = request.args.get('msg')
    return str(processQuery(textInput))

def processQuery(userInput):
    responseAgent = 'aiml'
    if responseAgent == 'aiml':
        answer = kernel.respond(userInput)
    #post-process the answer for commands
    if answer[0] == '#':
        params = answer[1:].split('$')
        cmd = int(params[0])
        if cmd == 0:
            return params[1]
        elif cmd == 1:
            return "wikipedia article"
        elif cmd == 2:
            return ""
        elif cmd == 99:
            return "I did not get that, please try again."
    else:
        return answer


if __name__ == '__main__':
    app.run()