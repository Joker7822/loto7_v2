import csv
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import pandas as pd

# Configure Chrome Driver
chrome_service = Service("C:/Users/lsaka/Desktop/Numbers/chromedriver.exe")  # Update path if necessary
chrome_options = Options()
chrome_options.add_argument("--headless")  # Run in headless mode for faster execution
driver = webdriver.Chrome(service=chrome_service, options=chrome_options)

# Load the target page
url = "https://www.mizuhobank.co.jp/takarakuji/check/loto/loto7/index.html"
driver.get(url)

# Locate elements for required data
draw_date_elements = driver.find_elements(By.CLASS_NAME, "js-lottery-date-pc")
main_numbers_elements = driver.find_elements(By.CLASS_NAME, "js-lottery-number-pc")
bonus_number_elements = driver.find_elements(By.CLASS_NAME, "js-lottery-bonus-pc")
draw_number_elements = driver.find_elements(By.CLASS_NAME, "js-lottery-issue-pc")

# Process and store the data in a list of dictionaries
data = []
for i in range(len(draw_date_elements)):
    # Format draw date as YYYY-MM-DD
    draw_date = datetime.strptime(draw_date_elements[i].text, "%Y年%m月%d日").strftime("%Y-%m-%d")
    
    # Format main numbers as a space-separated string without quotes
    main_numbers = " ".join([main_numbers_elements[j].text for j in range(i * 7, (i + 1) * 7)])  # Collect six main numbers
    
    # Remove parentheses from bonus number
    bonus_number = bonus_number_elements[i].text.strip("()") if i < len(bonus_number_elements) else ""
    
    # Get draw number (回別)
    draw_number = draw_number_elements[i].text

    data.append({
        "回別": draw_number,
        "抽せん日": draw_date,
        "本数字": main_numbers,
        "ボーナス数字": bonus_number
    })

# Close the driver
driver.quit()

# Load existing CSV to check for duplicates and get the format
csv_file_path = r'C:\Users\lsaka\Desktop\Numbers\loto7\loto7.csv'
existing_data = pd.read_csv(csv_file_path)
existing_dates = existing_data["抽せん日"].tolist()
fieldnames = existing_data.columns.tolist()  # Preserve existing column order and names

# Filter out duplicates
new_data = [row for row in data if row["抽せん日"] not in existing_dates]

# Append new data to CSV if there are new entries
if new_data:
    with open(csv_file_path, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writerows(new_data)

# Display results in the console
for row in new_data:
    print(row)
