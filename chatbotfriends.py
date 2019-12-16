import random
import string  # to process standard python strings
import nltk
from nltk.stem import WordNetLemmatizer

# nltk.download('punkt')  # first-time use only
# nltk.download('wordnet')  # first-time use only

flag = True
print("Hi there, I can be any of the six Friends, who do you want me to be?")
friend = input().lower()

f = open(friend + '.txt', 'r')
raw = f.read()
raw = raw.lower()  # converts to lowercase

sent_tokens = raw.split('\n')
# sent_tokens = nltk.sent_tokenize(raw)  # Segmenting the text into sentences. Output: converted list of sentences
filtered_tokens = [x for x in sent_tokens if str(friend.lower()) + ":" not in x]

word_tokens = nltk.word_tokenize(raw)  # Tokenize a string to split off punctuation other than periods. (don't)
# Output: converted list of words

lemmer = nltk.stem.WordNetLemmatizer()  # Lemma: basic word form


# WordNet is a semantically-oriented dictionary of English included in NLTK.
def LemTokens(tokens):  # creates normalized tokens
    return [lemmer.lemmatize(token) for token in tokens]


remove_punkt_dict = dict((ord(punkt), None) for punkt in string.punctuation)  # removes punctuation


def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punkt_dict)))  # ??


GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up", "hey", "How you doin'?")
GREETING_RESPONSES = ["hi", "hey", "*nods*", "hi there", "hello", "I am glad! You are talking to me", "How you doin'?"]

if friend == "ross":
    FRIEND_QUOTES = ("PIVOT!", "We were on a break!", "Unagi!")
elif friend == "monica":
    FRIEND_QUOTES = ("Seven!", "I have to clean.")
elif friend == "rachel":
    FRIEND_QUOTES = ("Sorry I'm late, but I left late. ...", "I want to get one of those job things.")
elif friend == "phoebe":
    FRIEND_QUOTES = ("That is brand new information.", "She's your lobster. ...")
elif friend == "chandler":
    FRIEND_QUOTES = ("Tell him to email me at www-dot-haha-not-so-much-dot-com!", "")
elif friend == "joey":
    FRIEND_QUOTES = ("How you doin'?", "Va' fa Napoli!", "Here come the meat sweats...")


def greeting(sentence):
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)


from sklearn.feature_extraction.text import TfidfVectorizer

# to extract features in a format supported by machine learning algorithms from datasets consisting of formats such as
# text and image. Convert a collection of raw documents to a matrix of TF-IDF features.
# TFIDF =  term frequency–inverse document frequency. used as a weighting factor in searches of information retrieval,
# text mining, and user modeling.
# The tf–idf value increases proportionally to the number of times a word appears in the document

from sklearn.metrics.pairwise import cosine_similarity


# is a measure of similarity between two non-zero vectors of an inner product space that measures the
# cosine of the angle between them


def response(user_response):
    robo_response = ''
    filtered_tokens.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize)
    tfidf = TfidfVec.fit_transform(filtered_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)  # -1 : last element of the array (here: user_response)
    idx = vals.argsort()[0][-2]
    flat = vals.flatten()  # one dimensional
    flat.sort()
    req_tfidf = flat[-2]  # -2 because it sorts the array and the first one is the user_response itself
    idxResponse = sent_tokens.index(filtered_tokens[idx])
    if req_tfidf == 0:
        robo_response = robo_response + "I am sorry! I don't understand you"
    else:
        robo_response = robo_response + sent_tokens[idxResponse+1]  # +1
    return robo_response


print(friend.upper() + ": Hi I am " + friend.upper() + "! " + random.choice(FRIEND_QUOTES))
while flag:
    user_response = input()
    user_response = user_response.lower()
    if user_response != 'bye':
        if user_response == 'thanks' or user_response == 'thank you':
            flag = False
            print(friend + ": You are welcome..")
        else:
            if greeting(user_response) is not None:
                print(friend + ": " + greeting(user_response))
            else:
                print(response(user_response))
                filtered_tokens.remove(user_response)
    else:
        flag = False
        print(friend + ": Bye! take care..")
