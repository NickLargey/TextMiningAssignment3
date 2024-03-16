"""
Sarah Lawrence
Mar 20,2024
Behrooz Mansouri
470 Text Mining and Analytics
"""
# TODO: fix Comments & replace varable

import requests
from bs4 import BeautifulSoup
import os
import re
import concurrent.futures

# pip install beautifulsoup4

# Stores the number of songs the genre with the least amount has
def find_min(url,headers):
    response = requests.get(url,headers=headers)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        h2_tags = soup.find_all('h2')
        for h2 in h2_tags:
            if 'Displaying' in h2.text and 'lyrics' in h2.text:
                count_text = h2.text.strip().split(' ')[1]
                count = count_text.replace(',', '')  # Remove commas from the count
                return int(count)

# Puts the lyrics in to txt documents
def scrape_lyrics(all_links,headers,genre,filepath,count):
    for link in all_links:
        song_response = requests.get(link,headers=headers)
        if song_response.status_code == 200:
            song_soup = BeautifulSoup(song_response.content, "html.parser")
            
            song_title = song_soup.find('h1', class_='lyric-title')
            # Finding the title of the song
            if song_title is None:
                print(f"{count}Warning: Could not find song title.")
                count += 1
                continue
            song_title = re.sub(r'[/""]', '',(song_title.text.strip()))
            txt_filepath = os.path.join(filepath, genre,song_title + ".txt")
            # Checking if lyrics exist
            lyrics_check = song_soup.find("pre", {"id": "lyric-body-text"})
            if lyrics_check is not None:
                with open(txt_filepath, "w", encoding="utf-8") as file:
                    # Putting lyrics in a txt document
                    lyrics = lyrics_check.text.strip()
                    file.write(lyrics)
                    print(f"{count} Saved {song_title} in {genre} folder")
                    count += 1
            else:
                print(f"{count} Not Saved: had no lyrics")
                count += 1
        else:
            print(f"Failed to get {link}. Status code: {song_response.status_code}")
            count += 1
    return count

# Gets all the song links on the page
def get_all_genre_songs(website,headers):
    song_response = requests.get(website,headers=headers)
    if song_response.status_code == 200:
        soup = BeautifulSoup(song_response.content, 'html.parser')
    
        lyric_divs = soup.find_all('div', class_='sec-lyric sec-center clearfix row')

        all_links = []
        for div in lyric_divs:
            title_tags = div.find_all('p', class_='lyric-meta-title')
            for title_tag in title_tags:
                link = title_tag.find('a')
                if link:
                    href = link.get('href')
                    all_links.append("https://www.lyrics.com" + href)
    return all_links

# scraping the lyrics and putting them in to txt
def songs_in_genre_files(genre, url, filepath, headers, min_count):
    count = 0
    i = 0
    # TODO: (replace varable) 500 is only temp -> real val: min_count
    while count < 500:
        if i > 0:
            all_links = get_all_genre_songs(os.path.join(url, str(i)), headers)
        else:
            all_links = get_all_genre_songs(url, headers)
        count = scrape_lyrics(all_links, headers, genre, filepath, count)
        i += 1 

# making sure traing and test folders dont have simlar songs
def compare_folders(test_folder, train_folder, genre):
    test_blues_folder = os.path.join(test_folder,genre)
    train_blues_folder = os.path.join(train_folder, genre)

    # List all text file titles in the Test folder
    test_files = [file.lower() for file in os.listdir(test_blues_folder) if file.endswith(".txt")]

    # List all text file titles in the Training folder
    train_files = [file.lower() for file in os.listdir(train_blues_folder) if file.endswith(".txt")]

    for test_file in test_files:
        if test_file in train_files:
            # Remove the matched file in Test Songs
            print(f"Match found: {test_file} in Test Songs matches {test_file} in Training Songs")
            training_file_path = os.path.join(train_blues_folder, test_file)
            os.remove(training_file_path)
            print(f"Removed {test_file} from Training Songs {genre}")
        else:
            print(f"No match found for {test_file} in Training Songs {genre}")
            
def main():
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}
    # Current file path
    find_directory = os.path.dirname(os.path.abspath(__file__))
    # Createing the main folder
    train_filepath = os.path.join(find_directory, "Training Songs")
    os.makedirs(train_filepath, exist_ok=True)
    # Starting genre links
    genres_dict = {
    'Rock': 'https://www.lyrics.com/genre/Rock',
    'Rap': 'https://www.lyrics.com/genre/Hip+Hop',
    'Pop': 'https://www.lyrics.com/genre/Pop',
    'Metal': 'https://www.lyrics.com/style/Heavy+Metal',
    'Country': 'https://www.lyrics.com/style/Country',
    'Blues': 'https://www.lyrics.com/genre/Blues'
    }
    
    min_count = float('inf')
    for genre, url in genres_dict.items():
        # Making genre folders
        subfolder_path = os.path.join(train_filepath, genre)
        os.makedirs(subfolder_path, exist_ok=True)
        # Gets the number of songs form the genres that has the lowest
        count = find_min(url,headers)
        if count < min_count:
            min_count = count
    
    # Running each file tasks independently 
        # File tasks = putting lyrcs that are in txt document into the correct genre folder
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Creating the tasks/threads
        futures = []
        for genre, url in genres_dict.items():
            futures.append(executor.submit(songs_in_genre_files, genre, url, train_filepath, headers, min_count))     

    # Removing training songs that match with test songs 
    test_folder = os.path.join(find_directory, "Test Songs")
    for genre, _ in genres_dict.items():
        compare_folders(test_folder, train_filepath, genre)  

if __name__ == "__main__":
    main()