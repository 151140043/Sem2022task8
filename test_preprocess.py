import json
import os.path
import pandas as pd
from bs4 import BeautifulSoup

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


test = pd.read_csv("data/semeval-2022_task8_eval_data_202201.csv", header=0)
test_path = "E:\\test"
news_text1 = []
news_text2 = []
processed_pair_id = []

for i in range(0, len(test)):
    if i%1000 == 0:
        print("Review %d of %d\n" % (i + 1, len(test)))
    pair_id = str(test["pair_id"][i]).split("_")
    text1 = json_preprocess(test_path, int(pair_id[0]))
    text2 = json_preprocess(test_path, int(pair_id[1]))
    # if text2 is None or text1 is None:
    #     continue
    processed_pair_id.append(test["pair_id"][i])
    news_text1.append(text1)
    news_text2.append(text2)

output = pd.DataFrame(data={"pair_id": processed_pair_id, "text1": news_text1, "text2": news_text2})
output.to_csv("mid_data\\mid_test_data.csv")
print(output)