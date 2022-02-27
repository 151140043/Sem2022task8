import json
import os.path
import pandas as pd
from bs4 import BeautifulSoup
import re

def review_to_wordlist( review ):
    # Function to convert a document to a sequence of words,
    # optionally removing stop words.  Returns a list of words.
    #
    # 1. Remove HTML
    review_text = BeautifulSoup(review, "lxml").get_text()
    #
    # 2. Remove non-letters
    # review_text = re.sub("[^a-zA-Z]", " ", review_text)
    #

    return review_text

review_to_wordlist("afahasodoa")

def json_preprocess(path, file_name):
    mid_dir = str(file_name % 100)
    mid_dir = mid_dir.zfill(2)
    path_name = str(file_name) + ".json"
    json_file = os.path.join(path, mid_dir, path_name)
    try:
        with open(json_file, "r") as load_f:
            load_dict = json.load(load_f)
        return review_to_wordlist(load_dict["text"])
    except FileNotFoundError:
        print("Sorry!The file " + str(file_name) + " can't find.")
        return None

train = pd.read_csv("data/semeval-2022_task8_train-data_batch2.csv", header=0)
trail = pd.read_csv("data/semeval-2022_task8_trial-data.csv",header=0)
print(len(trail))
path = "E:\\data\\data"
trial_path = "E:\\train"
news_text1 = []
news_text2 = []
language = []
processed_pair_id = []
Geography = []
Time = []
Entities = []
Narrative = []
Overall = []
Style = []
Tone = []

for i in range(0, len(train)):
    if i%1000 == 0:
        print("Review %d of %d\n" % (i + 1, len(train)))
    # pair_id = str(train["pair_id"][i]).split("_")
    # text1 = json_preprocess(path, int(pair_id[0]))
    # text2 = json_preprocess(path, int(pair_id[1]))
    pair_id = str(train["pair_id"][i]).split("_")
    text1 = json_preprocess(trial_path, int(pair_id[0]))
    text2 = json_preprocess(trial_path, int(pair_id[1]))
    if text2 is None or text1 is None:
        continue
    processed_pair_id.append(train["pair_id"][i])
    news_text1.append(text1)
    news_text2.append(text2)
    language.append(train["url1_lang"][i])
    Geography.append(train["Geography"][i])
    Entities.append(train["Entities"][i])
    Time.append(train["Time"][i])
    Narrative.append(train['Narrative'][i])
    Overall.append(train['Overall'][i])
    Style.append(train['Style'][i])
    Tone.append(train['Tone'][i])

output = pd.DataFrame(data={"pair_id": processed_pair_id, "text1": news_text1, "text2": news_text2, \
                            "lang": language, "Geography": Geography, "Entities": Entities, "Time": Time, "Narrative": Narrative, \
                            "Overall": Overall, "Style": Style, "Tone": Tone })
output.to_csv("mid_data\\mid_train_data3.csv")
print(output)
