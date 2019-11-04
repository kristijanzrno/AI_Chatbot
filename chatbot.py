import aiml
import nltk
import string
from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

sent_tokens = []
word_tokens = []
lemmer = nltk.stem.WordNetLemmatizer()

kernel = aiml.Kernel()
kernel.setTextEncoding(None)
kernel.bootstrap(learnFiles='rules.xml')


@app.route('/')
def home():
   return render_template('index.html')

@app.route('/get')
def get_bot_response():
    text_input = request.args.get('msg')
    return str(process_query(text_input))

def process_query(user_input):
    response_agent = 'aiml'
    if response_agent == 'aiml':
        answer = kernel.respond(user_input)
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
            #similarity component
            return check_similarity(user_input)
    else:
        return answer

def load_data():
    data = open('data.txt', 'r')
    raw = data.read().lower()
    nltk.download('punkt') # first-time use only
    nltk.download('wordnet') # first-time use only
    sent_tokens = nltk.sent_tokenize(raw)
    word_tokens = nltk.word_tokenize(raw)

def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]

remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)

def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))


def check_similarity(user_input):
    result = ''
    sent_tokens.append(user_input)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfdif = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfdif[-1], tfdif)
    idx = vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfdif = flat[-2]
    if(req_tfdif == 0):
        result = result + "I'm sorry, I couldnt understand you."
    else:
        result = result + sent_tokens[idx]
    return result

load_data()

if __name__ == '__main__':
    app.run()