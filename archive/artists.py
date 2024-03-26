import requests
import re
from bs4 import BeautifulSoup
from archive.genres import rock, pop, metal


def scrapr(regex, urls):
    names_regex = re.compile(regex)
    for url in urls:
        response = requests.get(url)
        url = url.split('/')[-1].split('?')[0]
        soup = BeautifulSoup(response.text, 'html.parser')
        for artist in soup.find_all('textarea'):
            artist = artist.text
            artists = names_regex.findall(artist)
            # print(artists)
            with open(f'{url}.txt', 'w') as file:
                for artist in artists:
                    try:
                        file.write(artist + '\n')
                    except:
                        pass


def genre_scrapr(regex,genre_list, genre):
    names_regex = re.compile(regex)
    musicians = []
    for category in genre_list:
        category = category.lower()
        url = f'https://en.wikipedia.org/wiki/List_of_{category}_bands'
        response = requests.get(url)
        if response.status_code != 200:
            url = f'https://en.wikipedia.org/wiki/List_of_{category}_artists'
            response = requests.get(url)
        if response.status_code != 200:
            url = f'https://en.wikipedia.org/wiki/List_of_{category}_musicians'
            response = requests.get(url)
        if response.status_code != 200:
            print(f'Error: {response.status_code}, no page for {category}')
            continue
        url = url+"?action=edit"
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        for artist in soup.find_all('textarea'):
            artist = artist.text
            artists = names_regex.findall(artist)
            musicians.extend(artists)
    with open(f'List_of_{genre}_musicians.txt', 'w') as file:
        for m in musicians:
            try:
                file.write(m + '\n')
            except:
                pass


def main():
    scrapr(r'\*\[\[(.*?)\]\]', urls=[
        'https://en.wikipedia.org/wiki/List_of_country_music_performers?action=edit',
        'https://en.wikipedia.org/wiki/List_of_hip_hop_musicians?action=edit',
    ])

    scrapr(r'\[\[(.*?)\]\]\}\}',
           urls=['https://en.wikipedia.org/wiki/List_of_blues_musicians?action=edit',])

    genre_scrapr(r'\|\[\[(.*?)\]\]', metal, metal)
    genre_scrapr(r'\*\[\[(.*?)\]\]', pop, 'pop')
    genre_scrapr(r'\*\[\[(.*?)\]\]', rock, 'rock')
