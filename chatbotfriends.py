import random
import string  # to process standard python strings
import nltk

#nltk.download('punkt')  # first-time use only
#nltk.download('wordnet')  # first-time use only

flag = True
print("Hi there, I can be any of the six Friends, who do you want me to be?")
friend = input()

f = open(friend + '.txt', 'r')
raw = f.read()
raw = raw.lower()  # converts to lowercase

sent_tokens = nltk.sent_tokenize(raw)  # converts to list of sentences
word_tokens = nltk.word_tokenize(raw)  # converts to list of words

lemmer = nltk.stem.WordNetLemmatizer()


# WordNet is a semantically-oriented dictionary of English included in NLTK.
def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]


remove_punkt_dict = dict((ord(punkt), None) for punkt in string.punctuation)

def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punkt_dict)))


GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up", "hey", "How you doin'?")
GREETING_RESPONSES = ["hi", "hey", "*nods*", "hi there", "hello", "I am glad! You are talking to me", "How you doin'?"]
if friend == "Ross":
    FRIEND_QUOTES = ("PIVOT!", "We were on a break!", "Unagi!")
elif friend == "Monica":
    FRIEND_QUOTES = ("Seven!", "I have to clean.")
elif friend == "Rachel":
    FRIEND_QUOTES = ("Sorry I'm late, but I left late. ...","")
elif friend == "Phoebe":
    FRIEND_QUOTES = ("That is brand new information.", "She's your lobster. ...")
elif friend == "Chandler":
    FRIEND_QUOTES = ("Tell him to email me at www-dot-haha-not-so-much-dot-com!","")
elif friend == "Joey":
    FRIEND_QUOTES = ("How you doin'?", "Va' fa Napoli!", "Here come the meat sweats...")

def greeting(sentence):
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)


from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics.pairwise import cosine_similarity


def response(user_response):
    robo_response = ''
    sent_tokens.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize)
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx = vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if req_tfidf == 0:
        robo_response = robo_response + "I am sorry! I don't understand you"
    else:
        robo_response = robo_response + sent_tokens[idx]
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

                print(friend + ":" + response(user_response))
                sent_tokens.remove(user_response)
    else:
        flag = False
        print(friend + ": Bye! take care..")
