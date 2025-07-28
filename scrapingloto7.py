import csv
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd

# === Chrome設定 ===
options = Options()
options.add_argument("--headless")
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")

# Load the target page
driver = webdriver.Chrome(options=options)
url = "https://www.mizuhobank.co.jp/takarakuji/check/loto/loto7/index.html"
driver.get(url)

# Explicitly wait for the elements to load
wait = WebDriverWait(driver, 10)

# Locate elements for required data
data = []
try:
    # Find the number of draws listed on the page
    draw_count = len(wait.until(EC.presence_of_all_elements_located((By.CLASS_NAME, "js-lottery-date-pc"))))

    for i in range(draw_count):
        # Refetch elements to avoid stale references
        draw_date_elements = driver.find_elements(By.CLASS_NAME, "js-lottery-date-pc")
        main_numbers_elements = driver.find_elements(By.CLASS_NAME, "js-lottery-number-pc")
        bonus_number_elements = driver.find_elements(By.CLASS_NAME, "js-lottery-bonus-pc")
        draw_number_elements = driver.find_elements(By.CLASS_NAME, "js-lottery-issue-pc")
        
        # Skip if the draw date is empty
        if not draw_date_elements[i].text.strip():
            continue

        # Format draw date as YYYY-MM-DD
        draw_date = datetime.strptime(draw_date_elements[i].text, "%Y年%m月%d日").strftime("%Y-%m-%d")

        # Format main numbers as a space-separated string
        main_numbers = " ".join([main_numbers_elements[j].text for j in range(i * 7, (i + 1) * 7)])  # Collect seven main numbers

        # Collect two bonus numbers, stripping parentheses if present
        bonus_numbers = " ".join([bonus_number_elements[j].text.strip("()") for j in range(i * 2, (i + 1) * 2)])

        # Get draw number (回別)
        draw_number = draw_number_elements[i].text

        data.append({
            "回別": draw_number,
            "抽せん日": draw_date,
            "本数字": main_numbers,
            "ボーナス数字": bonus_numbers
        })

except Exception as e:
    print("Error occurred:", e)
finally:
    # Close the driver
    driver.quit()

# === 保存処理 ===
csv_path = "loto7.csv"
try:
    existing = pd.read_csv(csv_path)
    existing_dates = existing["抽せん日"].tolist()
    fieldnames = existing.columns.tolist()
except FileNotFoundError:
    existing = pd.DataFrame()
    existing_dates = []
    fieldnames = ["抽せん日", "本数字", "ボーナス数字", "回別"]

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
