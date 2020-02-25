import os
import aiml
import nltk
import string
import json, requests
from random import randrange
from geopy.geocoders import Nominatim
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from keras.models import load_model
from PIL import Image
from skimage import transform
import numpy as np
from shutil import copyfile
from keras.preprocessing.image import ImageDataGenerator

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

# ASTROBIN API used to fetch online astrophotography photos and their info
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

# Initializing FOL agent
v = """
planets => {}
galaxies => {}
nebulaes => {}
stars => {}
moons => {}
favourite_galaxies => fg
favourite_planets => fp
favourite_nebulaes => fn
favourite_stars => fs
favourite_moons => fm
be_in => {} """
# v = """ planets => {} galaxies => {} stars => {} nebulaes => {} moons => {} favourite galaxies => fg field2 => f2 field3 => f3 field4 => f4 be_in => {} """
folval = nltk.Valuation.fromstring(v)
grammar_file = "sem-cp.fcfg"
#grammar = nltk.data.load('simple-sem.fcfg')
val_assumptions = []
fc = 5
oc = 1
# Could not build Mace due to macOS CPP related errors (Mace makefile g++ incompatible with XCode clang++11) 
# model_builder = nltk.Mace()
# read_expr = nltk.sem.Expression.fromstring

# Image Clasiffication Question order resembling the GalaxyZoo flowchart
# There are 11 questions and 37 answers in total (37 classes)
# Defining 11 Questions
classification_questions=[
    'Is the galaxy simply smooth and rounded, with no sign of a disk?',
    'Could this be a disk viewed edge-on?',
    'Is there a sign of a bar feature through the centre of the galaxy?',
    'Is there any sign of a spiral arm pattern?',
    'How prominent is the central bulge, compared with the rest of the galaxy?',
    'Is there anything odd?',
    'How rounded is it?',
    'Is the odd feature a ring, or is the galaxy disturbed or irregular?',
    'Does the galaxy have a buldge at its centre? If so, what shape?',
    'How tightly wound do the spiral arms appear?',
    'How many spiral arms are there?']
#Defining 37 answers (classes)
classification_answers=['Smooth', 'Features or disk', 'Star of artifact', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No', 
    'No bulge', 'Just noticable', 'Obvious', 'Dominant', 'Yes', 'No', 'Completely round', 'In between', 'Cigar-shaped',
    'Ring', 'Lens or arc', 'Disturbed', 'Irregular', 'Other', 'Merger', 'Dust lane', 'Rounded', 'Boxy', 'No bulge', 'Tight',
    'Medium', 'Loose', 'One', 'Two', 'Three', 'Four', 'More than four', 'Can not tell']
# Question - answers range (e.g. first question has three answers, therefore 
# question at index 0 (1st question) has answers in range of [0,3] in the final predicted result)
classification_answer_ranges = [[0,3], [3,5], [5,7], [7,9], [9,13], [13,15], [15,18], [18,25], [25,28], [28,31], [31,37]]
# Format: AnswerNo [index] = NextQuestionNo
# NextQuestionNo = -1 marks the stop point
next_question = [7, 2, -1, 9, 3, 4, 4, 10, 5, 6, 6, 6, 6, 8, -1, 6, 6, 6, -1, -1, -1, -1, -1, -1, -1, 6, 6, 6, 11, 11, 11, 5, 5, 5, 5, 5, 5]

# Confirmation message
confirmation_message = 'I will note that.'

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

# Flask route which handles the image upload for the image classification
# Photos are uploaded with an asynchronous AJAX POST request to avoid page redirect
@app.route('/upload', methods = ['POST'])
def upload_file():
    # Checking if the file is in the post request
    if 'file' in request.files:
        # If yes, save it within the uploaded/data/ folder
        photo = request.files['file']
        photo.save('uploaded/data/'+photo.filename)
        # Right after this function ends, the classification function will be called

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
# Making temp grammar data as-well
def load_data():
    data = open('data.txt', 'r')
    for line in data:
        questions.append(line.split('::')[0].lower())
        answers.append(line.split('::')[1])
    copyfile('simple-sem.fcfg', grammar_file)  

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
    # Check if an image has been uploaded for classification
    if(user_input == '__csf__'):
        # If yes, classify the last uploaded image and return the classified object name
        return classify();
    # else, continue with the regular chat handling
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
        elif cmd == 4:
            return
        # FOL - "My favourite * is * "
        elif cmd == 6: 

            return 
        # ONE OF MY FAVOURITE * IS *
        elif cmd == 7: 
            global oc
            o = params[2]
            add_knowledge(pos=o, a='o'+str(oc), b=None, obj=True, updating=False)
            fav_str = 'favourite_'+params[1]
            if params[1] not in folval:
                update_valuation(params[1], False)
                update_grammar(params[1], True)
            if fav_str not in folval:
                update_valuation(fav_str, True)
                update_grammar(fav_str, False)
            add_knowledge(pos = params[1], a = o, b=None, obj = False, updating=False)
            add_knowledge(pos ='be_in', a = o, b=folval[fav_str], obj = False, updating=False)
            #print(folval)
            return confirmation_message
        # FOL - "What are my favourite *"
        elif cmd == 11:
            g = nltk.Assignment(folval.domain)
            m = nltk.Model(folval.domain, folval)
            fav_str = 'favourite_'+params[1]
            e = nltk.Expression.fromstring("be_in(x," + fav_str + ")")
            print(folval)
            #sent =  'all ' + 'planets ' + 'are_in ' + 'favourite_'+params[1]
            #testResult = nltk.evaluate_sents([sent], grammar_file, m, g)[0][0]
            #print(testResult)
            print(fav_str)
            sat = m.satisfiers(e, "x", g)
            print(sat)
            res = ''
            if len(sat) == 0:
                 res = 'None'
            else:
                for i in sat:
                    res += i + ', '
                res = res[:-2]
            return res
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
            try:
                geolocator = Nominatim(user_agent="ai_chatbot_astronomy", timeout=None)
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

# Function created to perform image classification service
def classify():
    # Although the ImageDataGenerator is not required here, because each time one image will be provided,
    # it is still being used for easy image pre-processing
    # Defining the data generator
    test_data_generator = ImageDataGenerator(rescale=1. / 255)
    # Defining the batch generator, with color mode set to RGB (same as the images for training),
    # and resizing them to target size 224x224, no need to shuffle because we are taking only 1 image
    test_batch_generator = test_data_generator.flow_from_directory(
        "./uploaded/",
        class_mode=None,
        color_mode="rgb",
        batch_size=1,
        target_size=(224, 224),
        shuffle=False)
    test_batch_generator.reset()
    # Classifying the given photo here
    predictions = model.predict_generator(test_batch_generator, steps=1)
    # Classified photo will result with an np array floating point numbers from 0 to 1 for 37 classes 
    # Putting those values in a float array so they are easier to process
    data = []
    for x in predictions[0]:
        data.append(float(x))
    # Remove the given picture after the classification, this can be modified and removed if someone wants to keep
    # the uploaded photos
    os.remove('./uploaded/' + test_batch_generator.filenames[0])
    # Processing the classification data, starting from the first question
    return(classification_answer(data, '', 0))

# Function created to process the actual classification prediction
# This function will resemble the functionality required by the GalaxyZoo challenge
# There are 11 questions with 37 answers (classes) in total
# The order of question that needs to be asked is specified in the next_question array
# For e.g., if a spiral is detected, then the next question will be how many arms are there
# Another example is, if an star or artifact is detected, dont ask any more questions
def classification_answer(data, response, question_number):
    # Checking which of the 37 answers are the answers for this specific question
    answers_range = classification_answer_ranges[question_number]
    # Cutting those answers from the rest of the list, so we can get the final answer to the question
    q_answers = data[answers_range[0]:answers_range[1]]
    # The final answer is the answer with the highest score of the answers for this question
    final_answer = max(q_answers)
    # Getting the index of this specific answer in the total array of 37 answers
    answer_index = answers_range[0] + q_answers.index(final_answer)
    # Finding out which is the next question to ask, and if there is one, do a recursive call to this function
    # and get the response for that question aswell
    if next_question[answer_index] != -1:
        response = classification_answer(data, response, next_question[answer_index]-1)
    # Setting the response to the response to this question + the response of all the previous questions asked
    # by the recursive calls
    response = classification_questions[question_number] + '<br> - ' + classification_answers[answer_index] + '<br>' + response 
    return response


def update_valuation(word, field):
    global v, folval, fc
    if field:
        word += ' => f'+ str(fc)
        fc+=1
    else:
        word += ' => {}'
    v += '\n'+word;
    folval = nltk.Valuation.fromstring(v)
    for apt in val_assumptions:
        add_knowledge(apt[0], apt[1], apt[2], apt[3], True)

def update_grammar(word, num):
    global grammar_data, grammar
    if num:
        lhs = nltk.grammar.Nonterminal(r'N[NUM=pl,SEM=<\x.'+word+'(x)>]')
    else:
        lhs = nltk.grammar.Nonterminal('PropN[-LOC,NUM=pl,SEM=<\P.P('+word+')>]')
    rhs = nltk.grammar.Nonterminal("""'"""+word+"""'""")
    new_production = nltk.grammar.Production(lhs, [rhs])
    with open(grammar_file, 'a') as f:
        f.write('\n'+str(new_production))

def clean_object(pos):
    if len(folval[pos]) == 1:
        if('',) in folval[pos]:
            folval[pos].clear()

def add_knowledge(pos, a, b, obj, updating):
    global v, folval, oc
    if obj:
        folval[pos] = a
        oc+=1
    else:
        clean_object(pos)
        if b == None:
            folval[pos].add((a))
        else:
            folval[pos].add((a,b)) 

    if not updating:
        val_assumptions.append((pos, a, b, obj))


if __name__ == '__main__':
    # Load the data from the text file and start the flask website
    # Note, the console will output the link where the Chatbot can be accessed in (Usually: http://127.0.0.1:5000/)
    # Please enter that link to interact with the chatbot 
    load_data()
    # Loading the trained model based on the vgg-16 architecture
    model = load_model('trained_model.h5')
    app.run()