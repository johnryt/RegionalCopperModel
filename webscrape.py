# import urllib.request as ul
# from bs4 import BeautifulSoup as soup



# url = 'https://avalanche.state.co.us/?lat=39.36222560655586&lng=-106.26562499999999'
# req = ul.Request(url, headers={'User-Agent':'Mizilla/5.0'})
# client = ul.urlopen(req)
# htmldata = client.read()
# client.close()

# pagesoup_ph = soup(htmldata, 'html.parser')

# pagesoup_ph.find_all('div')

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By

s=Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=s)
driver.maximize_window()
driver.get('https://avalanche.state.co.us/?lat=39.36222560655586&lng=-106.26562499999999')
elements = driver.find_elements(By.XPATH, '//*[@id="userActivityGraph"]')
svg = [WebElement.get_attribute('innerHTML') for WebElement in elements]
