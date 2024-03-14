# https://en.wikipedia.org/wiki/List_of_country_music_performers?action=edit
# https://en.wikipedia.org/wiki/List_of_blues_musicians?action=edit
# https://en.wikipedia.org/wiki/List_of_hip_hop_musicians?action=edit
# https://en.wikipedia.org/wiki/List_of_heavy_metal_bands?action=edit
# https://en.wikipedia.org/wiki/List_of_alternative_rock_artists?action=edit
# https://en.wikipedia.org/wiki/List_of_dance-pop_artists?action=edit
import requests
import re
from bs4 import BeautifulSoup


def scrapr(regex, urls):
    names_regex = re.compile(regex)
    for url in urls:
        response = requests.get(url)
        url = url.split('/')[-1].split('?')[0]
        soup = BeautifulSoup(response.text, 'html.parser')
        for artist in soup.find_all('textarea'):
            artist = artist.text
            artists = regex.findall(artist)
            # print(artists)
            with open(f'{url}.txt', 'w') as file:
                for artist in artists:
                    try:
                        file.write(artist + '\n')
                    except:
                        pass
def genre_scrapr(regex, urls):
    names_regex = re.compile(regex)
    for url in urls:
        response = requests.get(url)
        url = url.split('/')[-1].split('?')[0]
        soup = BeautifulSoup(response.text, 'html.parser')
        for artist in soup.find_all('textarea'):
            artist = artist.text
            artists = regex.findall(artist)
            # print(artists)
            with open(f'{url}.txt', 'w') as file:
                for artist in artists:
                    try:
                        file.write(artist + '\n')
                    except:
                        pass

def main():
    scrapr(r'\*\[\[(.*?)\]\]', urls=[
        'https://en.wikipedia.org/wiki/List_of_country_music_performers?action=edit',
        'https://en.wikipedia.org/wiki/List_of_hip_hop_musicians?action=edit',
    ])

    scrapr(r'\[\[(.*?)\]\]\}\}', urls=['https://en.wikipedia.org/wiki/List_of_blues_musicians?action=edit',])

    scrapr(r'==(.*?)==', urls=['https://en.wikipedia.org/wiki/Heavy_metal_genres?action=edit',
                                        'https://en.wikipedia.org/wiki/List_of_rock_genres?action=edit',
                                        'https://en.wikipedia.org/wiki/Category:Pop_music_genres'
                                        ])
