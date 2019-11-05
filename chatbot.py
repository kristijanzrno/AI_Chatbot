import aiml
import nltk
import string
import json, requests
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, render_template, request

# Uncomment on first use to download the library
# nltk.download('wordnet')

# IP-GEOLOCATION API used to fetch sunset, sunrise, moonset and moonrise for a specific location
ipgelocation_api_key = '064b4149765b496eb89050790ff10c68'
ipgeolocation_api_url = 'https://api.ipgeolocation.io/astronomy?apiKey='

# NASA API used to fetch the image of the day and its info
nasa_api_key = 'MvHhdCMNgq2VF1Tcu1UJYKemsPyFPnGE7U9dbXtn'
nasa_api_url = 'https://api.nasa.gov/planetary/apod?api_key='

# ASTROBIN API used to fetch online astrophotography photos nad their info
astrobin_api_key = '504261c21dd060d57ba869d7e46d742f5dceed27'
astrobin_api_secret = '921adc5c7dda876ae7ec0173a5a346273b70e866'
astrobin_api_url = 'https://www.astrobin.com/api/v1/image/?title__icontains='

# Default error message
error_msg = 'Sorry, I did not get that... Please try again.'

# Flask is used as a Chatbot UI provider
app = Flask(__name__)

# Lists of questions and anwers, which will be the same length
# meaning that the answer for e.g. question[3] will be in the answer list at the same index
questions = []
answers = []
# Initializing aiml kernel
kernel = aiml.Kernel()
kernel.setTextEncoding(None)
kernel.bootstrap(learnFiles='rules.xml')

# Flask route that handles the initial '/' request that loads the index webpage
@app.route('/')
def home():
   return render_template('index.html')

# Flask route which handles GET requests received from chatbot UI
# It Processes the query and returns the result
@app.route('/get')
def get_bot_response():
    text_input = request.args.get('msg').lower()
    return str(process_query(text_input))

# Using nltk lemmatizer to normalise inputs
# Normalising will be done on questions list once we want to compare the user input with the questions
lemmer = nltk.stem.WordNetLemmatizer()
def lem_tokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]

remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
def lem_tokenizer(text):
    return lem_tokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

# Data loading function which loads questions and answers into two separate lists
# Questions and answers are dividied by '::' symbol within a 'data.txt' file
def load_data():
    data = open('data.txt', 'r')
    for line in data:
        questions.append(line.split('::')[0].lower())
        answers.append(line.split('::')[1])

# Implementing similarity component using bag of words, tfidf and cosine similarity
def check_similarity(user_input):
    # Adding the user input at the end of the questions list
    questions.append(user_input)
    # Initialize TF-IDF vectorizer which will be used to generate 
    # a list of tf-idf matrices (one for each question)
    tfidf_vectorizer = TfidfVectorizer(tokenizer=lem_tokenizer, stop_words='english')
    tf_matrix = tfidf_vectorizer.fit_transform(questions)
    # Compare the last item (which is the user input matrix) in the list of matrices
    # with all the other items in the list of matrices
    # result is a ndarray of similarity values ranging from 0 to 1
    similarity_values = cosine_similarity(tf_matrix[-1], tf_matrix)
    # Flattening the ndarray to make it one dimensional and then
    # sorting it to order values from lowest to highest
    flat = similarity_values.flatten()
    flat.sort()
    # Once the array has been sorted, the second item from the back is our highest value
    # Note that, technically, last item in the array is always the highest because it is user input compared with itself which gives result 1.0
    # But we take the second last item which is the highest cosine similarity value between user input and questions 
    highest_value = flat[-2]
    # If the similarity value higher than 0.8, the question is similar enough and return an answer to it
    # If not, return an error_msg
    if highest_value > 0.8:
         # Finding the highest value index, which will be at the
         # second position from the back (last one will always be 1.0 because of
         # user-input matrix being compared with itself at the end of the list)
        highest_value_index = similarity_values.argsort()[0][-2]
        response = answers[highest_value_index]
    else:
        response = error_msg
    # Removing the, previously appended user_input from the questions list
    # Leaving it in would eventually make the chatbot crash,
    # cause the question list would be growing but the answer list would stay the same length
    questions.pop()
    return response

# User query processing function
def process_query(user_input):
    # This was left here for any future pre-processing
    response_agent = 'aiml'
    if response_agent == 'aiml':
        answer = kernel.respond(user_input)
    # Post-process the answer for commands
    if answer[0] == '#':
        params = answer[1:].split('$')
        cmd = int(params[0])
        if cmd == 0:
            # Since the chatbot is web based, when the user types bye, it will only print a bye message and not quit
            # But this space is left for any future upgrades
            return params[1]
        elif cmd == 1:
            return fetch_pic_of_the_day()[int(params[1])]
        elif cmd == 2:
            return find_geolocation_info(attribute=params[1], address=params[2])
        elif cmd == 3:
            return find_astrophotography(params[1])
        elif cmd == 99:
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

def extract_single_json_object(json_data, json_object_name, return_error):
    try:
        return json_data[json_object_name]
    except:
        if return_error:
            return error_msg
    return None

def find_geolocation_info(address, attribute):
            geolocator = Nominatim(user_agent="ai_chatbot_ntu", timeout=None, scheme='http')
            location = geolocator.geocode(address)
            if(location):
                try:
                    url = ipgeolocation_api_url + ipgelocation_api_key + "&lat="+str(location.latitude) +"&long="+str(location.longitude)
                    json_response = fetch_json(url)
                    if(json_response):
                        data = extract_single_json_object(json_response, attribute, False)
                        if(data):
                            return('The ' + attribute + ' at ' + address + ' is at ' + data)
                        return error_msg
                except:
                    return error_msg

def fetch_pic_of_the_day():
    url = nasa_api_url+nasa_api_key
    json_response = fetch_json(url)
    result = [error_msg, error_msg, error_msg]
    if(json_response):
        result[0] = extract_single_json_object(json_response, 'explanation', True)
        img = extract_single_json_object(json_response, 'hdurl', True)
        if img != error_msg:
            result[1] = 'img='+img
        result[2] = extract_single_json_object(json_response, 'copyright', True)
    return result

def find_astrophotography(search_term):
    url = astrobin_api_url + search_term + '&limit=1&api_key=' + astrobin_api_key + '&api_secret='+astrobin_api_secret+'&format=json'
    json_response = fetch_json(url)
    try:
        if(json_response):
            image = json_response['objects'][0]['url_hd']
            return('img='+image)
    except:
        return error_msg

if __name__ == '__main__':
    load_data()
    app.run()