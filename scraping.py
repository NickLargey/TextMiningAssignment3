import requests
from bs4 import BeautifulSoup
import re
from datetime import datetime
import pytz

utc_time = datetime.now(pytz.utc)
formatted_time = utc_time.strftime('%a, %d %b %Y %H:%M:%S GMT')

session = requests.Session()
res = session.get('http://www.mldb.org/')
cookie = session.cookies
print(cookie)
artist = "metallica"

headers = {
  'Host': 'www.mldb.org',
  'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:123.0) Gecko/20100101 Firefox/123.0',
  'Accept':'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
  }

page = 0
search_string = f'http://www.mldb.org/search?mq={artist}&mm=0&si=1&from={page}'

response = requests.get(search_string, headers=headers)
print(response.status_code)
soup = BeautifulSoup(response.text, 'html.parser')

pages = soup.find_all('a', href=True)
max_page = int(pages[-1].text) * 30

for i in range(0, max_page, 30):
  search_string = f'http://www.mldb.org/search?mq={artist}&mm=0&si=1&from={i}'
  response = requests.get(search_string, headers=headers)
  soup = BeautifulSoup(response.text, 'html.parser')

  for link in soup.find_all('td', class_="ft"):
    print(link)
