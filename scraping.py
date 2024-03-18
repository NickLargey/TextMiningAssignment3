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
# threads / processes 
import concurrent.futures
import time
import string
from difflib import SequenceMatcher

# pip install beautifulsoup4

# Stores the number of songs the genre with the least amount
def find_min(url, headers):
    time.sleep(0.06)
    response = requests.get(url,headers=headers)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        h2_tags = soup.find_all('h2')
        for h2 in h2_tags:
            if 'Displaying' in h2.text and 'lyrics' in h2.text:
                count_text = h2.text.strip().split(' ')[1]
                count = count_text.replace(',', '')  # Remove commas from the count
                return int(count)

# Geting lyrics and titles
def scrape_lyrics(all_links, headers, genre, num, songs, count, min_genre):
    for link in all_links:
        song_response = requests.get(link,headers=headers)     
        if song_response.status_code == 200:
            song_soup = BeautifulSoup(song_response.content, "html.parser")
            
            # Finding the title of the song
            song_title = song_soup.find('h1', class_='lyric-title')
            if song_title is None:
                print(f"Warning: Could not find song title.")
                if genre == min_genre:
                 num -= 1
                if len(songs) == num:
                    return songs,num
                continue
            song_title = re.sub(r'[./""]', '',(song_title.text.strip()))

            # Finding the lyrics of the song
            lyrics_check = song_soup.find("pre", {"id": "lyric-body-text"})
            if lyrics_check is not None:
                # Adds count to the song title if it repeat
                while song_title in songs.values():
                    song_title = f"{song_title} {count}"
                    count += 1
                
                # Put in set to get rid of duplicates
                songs[lyrics_check.text.strip()] = song_title
                print(f"{len(songs)} Saved {song_title} in {genre} folder")
                # Fakes being a human to bypasses robot protection on website 
                # TODO fip between two sleeps 0.1 and 0.2
                time.sleep(0.2)
                if len(songs) == num:
                    return songs,num
            else:
                print(f"Not Saved: had no lyrics")
                if genre == min_genre:
                 num -= 1
                if len(songs) == num:
                    return songs,num
        else:
            print(f"Failed to get {link}. Status code: {song_response.status_code}")
            if genre == min_genre:
                 num -= 1
            if len(songs) == num:
                return songs,num
    
    return songs,num

# Gets all the song links on the page
def get_all_genre_songs(website,headers):
    song_response = requests.get(website, headers=headers)
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

# Scraping the lyrics and putting them in to txt
def songs_in_genre_files(genre, url, filepath, headers, min_count, min_genre):
    i,num = 1,min_count
    songs = {}
    count = 1
    while len(songs) != num:
        if i > 0:
            all_links = get_all_genre_songs(os.path.join(url, str(i)), headers)
        else:
            all_links = get_all_genre_songs(url, headers)
        local_songs,num = scrape_lyrics(all_links, headers, genre, num, songs, count, min_genre)
        songs.update(local_songs)
        i += 1 
    # Putting lyrics in a txt document
    for lyrics, title in songs.items():
        txt_filepath = os.path.join(filepath, genre, title + ".txt")
        with open(txt_filepath, "w", encoding="utf-8") as file:
            file.write(lyrics)
# Function to read the contents of a text file
def read_file_contents(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

# Making sure traing and test folders dont have simlar songs
def compare_folders(test_file, train_files, test_folder, train_folder, genre):
    test_file_path = os.path.join(test_folder, test_file)
    test_contents = read_file_contents(test_file_path)

    for train_file in train_files:
        train_file_path = os.path.join(train_folder, train_file)
        train_contents = read_file_contents(train_file_path)
        
        # Making the content simlar format 
        pattern = r'[\n\s{}]+'.format(re.escape(string.punctuation))
        test_contents_no_space = re.sub(pattern, "", test_contents).lower()
        train_contents_no_space = re.sub(pattern, "", train_contents).lower()
        
        
        # TODO slow need to be faster
        match = SequenceMatcher(None, test_contents_no_space, train_contents_no_space).ratio() 
        match = int(match * 100)

        if match > 50:
            os.remove(train_file_path)
            print(f"Removed {train_file} from Training Songs {genre} {match}")

def title_compare (test_folder,train_folder,genre):
    test_files = [file.lower() for file in os.listdir(test_folder) if file.endswith(".txt")]

    # List all text file titles in the Training folder
    train_files = [file.lower() for file in os.listdir(train_folder) if file.endswith(".txt")]

    for test_file in test_files:
        if test_file in train_files:
            # Remove the matched file in Test Songs # match lyrics
            training_file_path = os.path.join(train_folder, test_file)
            os.remove(training_file_path)
            print(f"Removed {test_file} from Training Songs {genre}")

# Function to count the number of files in a directory
def count_files(path):
    return sum([len(files) for _, _, files in os.walk(path)])

# Removes extra txt files
def remove_extra(subfolder_list,min_count):
    for genre, path in subfolder_list.items():
        count = count_files(path)
        if count > min_count:
            files_to_delete = count - min_count
            files = [os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
            files.sort(key=lambda x: os.path.getctime(x), reverse=True)
            for i in range(files_to_delete):
                os.remove(files[i])
            print(f"Genre {genre} deleted {files_to_delete} file's")
        elif count < min_count:
            print(f"Error: Genre {genre} has fewer files than the minimum count.")

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
    min_genre = None
    subfolder_list = {}
    for genre, url in genres_dict.items():
        # Making genre folders
        subfolder_path = os.path.join(train_filepath, genre)
        os.makedirs(subfolder_path, exist_ok=True)
        subfolder_list[genre] = subfolder_path
        # Gets the number of songs form the genres that has the lowest
        count = find_min(url,headers)
        if count < min_count:
            min_count = count
            min_genre = genre
    
    # Running each file tasks independently 
        # File tasks = putting lyrcs that are in txt document into the correct genre folder
    with concurrent.futures.ThreadPoolExecutor() as executor:
         # Creating the tasks/threads
        futures = []
        for genre, url in genres_dict.items():
            futures.append(executor.submit(songs_in_genre_files, genre, url, train_filepath, headers, min_count, min_genre))     

    # Removing training songs that match with test songs 
    print("Start match check processes: removes songs that match Test Songs")
    for genre, _ in genres_dict.items():
        test_folder = os.path.join(find_directory, "Test Songs",genre)
        train_folder = os.path.join(find_directory, "Training Songs", genre)

        # List all text file titles in the Test folder
        test_files = [file for file in os.listdir(test_folder) if file.endswith(".txt")]
        # List all text file titles in the Training folder
        train_files = [file for file in os.listdir(train_folder) if file.endswith(".txt")]
        for test_file in test_files:
            compare_folders(test_file, train_files, test_folder, train_folder, genre) 
        #title_compare (test_folder,train_folder,genre)
    print("Match check complete")
    
    # All the genres have the same amount of songs
    file_counts = {genre: count_files(path) for genre, path in subfolder_list.items()}
    min_genre = min(file_counts, key=file_counts.get)
    min_count = file_counts[min_genre]
    with concurrent.futures.ThreadPoolExecutor() as executor:
        threads = []
        for genre, path in subfolder_list.items():
            threads.append(executor.submit(remove_extra, {genre: path}, min_count))
    
    print("Scraping compleated!")
        
if __name__ == "__main__":
    main()