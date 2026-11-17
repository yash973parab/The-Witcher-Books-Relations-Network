import pandas as pd
import matplotlib.pyplot as plt
import time
from selenium import webdriver
from selenium.webdriver.common.by import By

# 1. CONFIGURE CHROME OPTIONS
options = webdriver.ChromeOptions()
# 'eager' tells Selenium to continue as soon as the HTML is loaded (ignoring slow ads/images)
options.page_load_strategy = 'eager'

driver = webdriver.Chrome(options=options)
# Give the browser a huge 5-minute buffer so it doesn't time out
driver.set_page_load_timeout(300)

# 2. MAIN NAVIGATION
page_url = "https://witcher.fandom.com/wiki/Category:Characters_in_the_stories"
driver.get(page_url)
time.sleep(5) # Give extra time for the first page

# Accept Cookies
try:
    driver.find_element(By.XPATH, '//button[text()="I Accept"]').click()
except:
    pass # If button doesn't appear, just move on

# FIND BOOKS
book_categories = driver.find_elements(By.CLASS_NAME, 'category-page__member-link')
# check if book names are scraped
books = []
for category in book_categories:
    book_url = category.get_attribute('href')
    book_name = category.text
    books.append({"Book Name": book_name, "URL": book_url})

character_list = []
for book in books:
    driver.get(book['URL'])
    # Crucial: Give the page 3 seconds to load so it doesn't time out
    time.sleep(5)
    character_elements = driver.find_elements(By.CLASS_NAME, 'category-page__member-link')

    for element in character_elements:
        character_list.append({"Book":book["Book Name"], "Character":element.text})

# RESULT DISPLAY
character_df = pd.DataFrame(character_list)
character_df["Book"].value_counts().plot(kind='bar')
plt.show()
