import aiml
import nltk
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, render_template, request

#Uncomment on first use to download the library
#nltk.download('wordnet')

app = Flask(__name__)

questions = []
answers = []

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

lemmer = nltk.stem.WordNetLemmatizer()
def lem_tokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]

remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)

def lem_tokenizer(text):
    return lem_tokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))


def load_data():
    data = open('data.txt', 'r')
    for line in data:
        questions.append(line.split('::')[0].lower())
        answers.append(line.split('::')[1])

load_data()

def check_similarity(user_input):
    questions.append(user_input)
    tfidf_vectorizer = TfidfVectorizer(tokenizer=lem_tokenizer, stop_words='english')
    tf_matrix = tfidf_vectorizer.fit_transform(questions)
    vals = cosine_similarity(tf_matrix[-1], tf_matrix)
    idx = vals.argsort()[0][-2]
    print(vals)
    flat = vals.flatten()
    print(flat)
    flat.sort()
    print(flat)
    req_tfidf = flat[-2]
    if req_tfidf < 0.8:
        response = 'Couldnt figure it out'
    else:
        response = answers[idx]
    questions.pop()
    return response

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


if __name__ == '__main__':
    app.run()