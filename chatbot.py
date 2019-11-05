import aiml
import nltk
import string
import json, requests
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, render_template, request

#Uncomment on first use to download the library
#nltk.download('wordnet')

ipgelocation_api_key = '064b4149765b496eb89050790ff10c68'
ipgeolocation_api_url = 'https://api.ipgeolocation.io/astronomy?apiKey='

nasa_api_key = 'MvHhdCMNgq2VF1Tcu1UJYKemsPyFPnGE7U9dbXtn'
nasa_api_url = 'https://api.nasa.gov/planetary/apod?api_key='

astrobin_api_key = '6dd236d199c8b291509e52ffed9761d79ee305ec'
astrobin_api_secret = '69077fef211155bbc60d98207dafa27830221bc0'
astrobin_api_url = 'https://www.astrobin.com/api/v1/image/?title__icontains='

error_msg = 'Sorry, I could not find the answer... Please try again.'

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
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if req_tfidf < 0.8:
        response = error_msg;
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
            return fetch_pic_of_the_day()[int(params[1])]
        elif cmd == 2:
            return find_geolocation_info(attribute=params[1], address=params[2])
        elif cmd == 3:
            return find_astrophotography(params[1])
        elif cmd == 99:
            #similarity component
            return check_similarity(user_input)
    else:
        return answer

def fetch_json(url):
    response = requests.get(url)
    if response.status_code == 200:
        json_response = json.loads(response.content)
        if(json_response):
            return json_response
    return None

def find_geolocation_info(address, attribute):
            geolocator = Nominatim(user_agent="ai_chatbot_ntu", timeout=None)
            location = geolocator.geocode(address)
            if(location):
                try:
                    url = ipgeolocation_api_url + ipgelocation_api_key + "&lat="+str(location.latitude) +"&long="+str(location.longitude)
                    json_response = fetch_json(url)
                    if(json_response):
                        data = json_response[attribute]
                        return('The ' + attribute + ' at ' + address + ' is at ' + data)
                except:
                    return error_msg
            return error_msg

def fetch_pic_of_the_day():
    url = nasa_api_url+nasa_api_key
    json_response = fetch_json(url)
    try:
        if(json_response):
            return [json_response['explanation'],'img='+json_response['hdurl'], json_response['copyright']]
    except:
        return error_msg

def find_astrophotography(search_term):
    url = astrobin_api_url + search_term + '&limit=1&api_key=' + astrobin_api_key + '&api_secret='+astrobin_api_secret+'&format=json'
    json_response = fetch_json(url)
    try:
        if(json_response):
            image = json_response['objects'][0]['url_hd']
            return('img='+image)
    except:
        return error_msg
    return error_msg


if __name__ == '__main__':
    app.run()