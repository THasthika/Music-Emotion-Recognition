import re
import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
import time
import sys
import random

# ip_addresses = [
#     "51.158.68.68:8761",
#     "51.83.193.220:80",
#     "51.158.186.242:8811"
# ]

NOT_FOUND_TIME = 20
BLOCKED_TIME = 60 * 60
SLEEP_TIME = 20

TEST_CSV_FILE = "./datasets/deezer_mood_detection_dataset/test.csv"
TEST_LYRICS_OUTPUT = "./datasets/deezer_mood_detection_dataset/lyrics/test/"

VALIDATION_CSV_FILE = "./datasets/deezer_mood_detection_dataset/validation.csv"
VALIDATION_LYRICS_OUTPUT = "./datasets/deezer_mood_detection_dataset/lyrics/validation/"

TRAIN_CSV_FILE = "./datasets/deezer_mood_detection_dataset/train.csv"
TRAIN_LYRICS_OUTPUT = "./datasets/deezer_mood_detection_dataset/lyrics/train/"

def get_lyrics(artist, song_title, tries=0):
    artist = artist.lower()
    song_title = song_title.lower()
    # remove all except alphanumeric characters from artist and song_title
    artist = re.sub('[^A-Za-z0-9]+', "", artist)
    song_title = re.sub('[^A-Za-z0-9]+', "", song_title)
    # remove starting 'the' from artist e.g. the who -> who
    if artist.startswith("the"):
        artist = artist[3:]
    url = "http://azlyrics.com/lyrics/"+artist+"/"+song_title+".html"

    try:
        # proxy_index = random.randint(0, len(ip_addresses) - 1)
        # proxy = {"http": ip_addresses[proxy_index], "https": ip_addresses[proxy_index]}
        # content = requests.get(url, proxies=proxy)
        response = requests.get(url)
        if response.status_code == 404:
            time.sleep(NOT_FOUND_TIME)
            return 404
        if not response.ok:
            if tries < 3:
                print("waiting...{}".format(tries), end=" ", flush=True)
                time.sleep(BLOCKED_TIME)
                return get_lyrics(artist, song_title, tries+1)
            else:
                raise Exception("max tries exceeded")
        content = response.content

        soup = BeautifulSoup(content, 'html.parser')
        lyrics = str(soup)
        # lyrics lies between up_partition and down_partition
        up_partition = '<!-- Usage of azlyrics.com content by any third-party lyrics provider is prohibited by our licensing agreement. Sorry about that. -->'
        down_partition = '<!-- MxM banner -->'
        lyrics = lyrics.split(up_partition)[1]
        lyrics = lyrics.split(down_partition)[0]
        lyrics = lyrics.replace('<br>', '').replace(
            '</br>', '').replace('</div>', '').replace(
            '<br/>', '').strip()
        return lyrics
    except Exception as e:
        raise e

def download_lyrics(csv_file, output_folder):
    df = pd.read_csv(csv_file)
    count = len(df)
    for index, r in df.iterrows():
        artist_name = r["artist_name"]
        track_name = r["track_name"]
        ofname = os.path.join(output_folder, "{}.txt".format(r["dzr_sng_id"]))
        ofname_notfound = os.path.join(output_folder, "{}_404".format(r["dzr_sng_id"]))
        print("{}/{}... ".format(index, count), end="", flush=True)
        if os.path.exists(ofname_notfound):
            print("not found... skipped")
            continue
        if os.path.exists(ofname):
            print("skipped")
            continue
        lyrics = ""
        try:
            lyrics = get_lyrics(artist_name, track_name)
        except Exception as e:
            print("exception " + str(e))
            continue
        if lyrics == 404:
            with open(ofname_notfound, "w+") as f:
                f.write("404")
                f.flush()
            print("lyrics not found")
            continue
        with open(ofname, "w+") as f:
            f.write(lyrics)
            f.flush()
        print("downloaded")
        time.sleep(SLEEP_TIME)

download_lyrics(TEST_CSV_FILE, TEST_LYRICS_OUTPUT)

download_lyrics(VALIDATION_CSV_FILE, VALIDATION_LYRICS_OUTPUT)

download_lyrics(TRAIN_CSV_FILE, TRAIN_LYRICS_OUTPUT)