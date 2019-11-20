import random
import string  # to process standard python strings

import nltk

f = open('advice.txt', 'r')
raw = f.read()
raw = raw.lower()  # converts to lowercase
nltk.download('punkt')  # first-time use only
nltk.download('wordnet')  # first-time use only
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


def greeting(sentence):
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)


from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics.pairwise import cosine_similarity


def response(user_response):
    robo_response = ''
    sent_tokens.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx = vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if req_tfidf == 0:
        robo_response = robo_response + "I am sorry! I don't understand you"
        return robo_response
    else:
        robo_response = robo_response + sent_tokens[idx]
        return robo_response

flag = True
print("ROBO: My name is Robo. I will give you excellent advice. If you want to exit, type Bye!")
while flag:
    user_response = input()
    user_response = user_response.lower()
    if user_response != 'bye':
        if user_response == 'thanks' or user_response == 'thank you':
            flag = False
            print("ROBO: You are welcome..")
        else:
            if greeting(user_response) is not None:
                print("ROBO: " + greeting(user_response))
            else:
                if user_response == 'this was bad advice':
                    print("ROBO: Then talk to a real friend and not to a robot!")
                else:
                    print('ROBO: ')
                    print(response(user_response))
                    sent_tokens.remove(user_response)
    else:
        flag = False
        print("ROBO: Bye! take care..")