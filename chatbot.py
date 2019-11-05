
import aiml
import nltk
import string
import json, requests
from random import randrange
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, render_template, request

# Important Note; The console will output the link where the Chatbot can be accessed in (Usually: http://127.0.0.1:5000/)
# Please enter that link to interact with the chatbot 

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
    # Preprocessing the user input to remove punctuation
    user_input.translate(str.maketrans('', '', string.punctuation))

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
            # When parsed aiml response starts with #1, fetch NASA picture of the day and display it
            # Created fetch_pic_of_the_day function gives a result in format [description, picture, author]
            # And based on the question parameters, one of those 3 will be displayed
            return fetch_pic_of_the_day()[int(params[1])]
        elif cmd == 2:
            # When parsed aiml respone starts with #2, it means that the user wants one of the following information:
            # sunrise time, sunset time, moonrise time, moonset time at a specific location
            return find_geolocation_info(attribute=params[1], address=params[2])
        elif cmd == 3:
            # When parsed aiml respone starts with #2, it means that the user is searching for astrophotography photos
            return find_astrophotography(search_term=params[1])
        elif cmd == 99:
            # If the user question doesnt match any of the above queries, check the similarity of the query
            # with the questions loaded from 'data.txt' file; if any of them are matching enough (>0.8) 
            # return the answer to it
            return check_similarity(user_input)
    else:
        return answer

# Function created to fetch json data
# and return a null object if the fetching has failed
def fetch_json(url):
    response = requests.get(url)
    if response.status_code == 200:
        json_response = json.loads(response.content)
        if(json_response):
            return json_response
    return None

# Function created to extract json data by object name
# and return either an error message or an null object depending on the case
def extract_single_json_object(json_data, json_object_name, return_error):
    try:
        return json_data[json_object_name]
    except:
        if return_error:
            return error_msg
    return None

# Function created to find sunset, sunrise, moonset, moonrise data for a
# user specified location
def find_geolocation_info(address, attribute):
            # Using geopy.geolocator Nominatim to convert user query address into latitude and longitude
            geolocator = Nominatim(user_agent="ai_chatbot_ntu", timeout=None, scheme='http')
            location = geolocator.geocode(address)
            if(location):
                try:
                    # Constructing an ipgeolocator request url using the above found latitude and longitude values, along with api_key
                    url = ipgeolocation_api_url + ipgelocation_api_key + "&lat="+str(location.latitude) +"&long="+str(location.longitude)
                    json_response = fetch_json(url)
                    if(json_response):
                        # Extracting the data based on the attribute (extracted from the user query, e.g. when is the SUNSET at Nottingham)
                        # user had wanted (moonrise, moonset, sunrise sunset)
                        data = extract_single_json_object(json_response, attribute, False)
                        if(data):
                            # If the json object has been successfully extracted, return the result to chatbot UI
                            return('The ' + attribute + ' at ' + address + ' is at ' + data)
                        return error_msg
                except:
                    return error_msg

# Function created for NASA picture of the day information retreival
def fetch_pic_of_the_day():
    # Constructing url with the specified API key
    url = nasa_api_url+nasa_api_key
    json_response = fetch_json(url)
    # Function returns information formatted as a list of [description, image_url, author]
    # Sometimes some of these information are unavailable on the API (e.g. author)
    # so we are setting error messages as default outputs
    result = [error_msg, error_msg, error_msg]
    if(json_response):
        # Use created extract json object function in order to extract explanation (description),
        # hdurl (image link), and copyright (author) and add them into return array
        result[0] = extract_single_json_object(json_response, 'explanation', True)
        img = extract_single_json_object(json_response, 'hdurl', True)
        if img != error_msg:
            # Images are returned to the web ui as img=img_url, so when it sees that the 
            # returned answer starts with 'img=', it knows it should load an image in the chat box
            result[1] = 'img='+img
        result[2] = extract_single_json_object(json_response, 'copyright', True)
    return result

# Function created to fetch astronomy objects photos (e.g. nebulas, galaxies, planets...)
def find_astrophotography(search_term):
    # Constructing url of a specified API key, and a search term extracted from the user query
    # Limited the query to 10 objects
    url = astrobin_api_url + search_term + '&limit=10&api_key=' + astrobin_api_key + '&api_secret='+astrobin_api_secret+'&format=json'
    json_response = fetch_json(url)
    try:
        if(json_response):
            # Fetch the image object from the result json query and return it as img=img_url (just like in the NASA example)
            # The image is randomised between the fetched (max 10) objects to not show the same image
            # for the same query all the time
            image_objects = json_response['objects']
            rand_image = randrange(len(image_objects)-1)
            image = image_objects[rand_image]['url_hd']
            return('img='+image)
    except:
        return error_msg

if __name__ == '__main__':
    # Load the data from the text file and start the flask website
    # Note, the console will output the link where the Chatbot can be accessed in (Usually: http://127.0.0.1:5000/)
    # Please enter that link to interact with the chatbot 
    load_data()
    app.run()