import os
from glob import glob
from bs4 import BeautifulSoup
import pandas as pd
import re

# パーシング関数の修正
def parse(soup):
    parsed_data = []
    
    # 'section__table' クラスのテーブルを全て取得
    tables = soup.find_all('table', class_='section__table')
    
    for table in tables:
        data = {}
        rows = table.find_all('tr')
        
        for row in rows:
            th = row.find('th')
            td = row.find('td')
            
            if not th or not td:
                continue
            
            label = th.get_text(strip=True)
            if label == '回別':
                data['回別'] = td.get_text(strip=True)
            elif label == '抽せん日':
                data['抽せん日'] = td.get_text(strip=True)
            elif label == '本数字':
                # 本数字を抽出
                numbers = [num_tag.get_text(strip=True) for num_tag in td.find_all('b')]
                if not numbers:
                    numbers = td.get_text(strip=True).split()
                data['本数字'] = ' '.join(numbers)
            elif label == 'ボーナス数字':
                # ボーナス数字を抽出
                numbers = [num_tag.get_text(strip=True) for num_tag in td.find_all('b')]
                if not numbers:
                    numbers = td.get_text(strip=True).split()
                data['ボーナス数字'] = ' '.join(numbers)
        
        if data:
            parsed_data.append(data)
    
    # 他の重要な要素を含むセクションを取得
    sections = soup.select('div.section__table-wrap')
    
    for section in sections:
        # 回別 (draw number)
        draw_number_tag = section.select_one('span.js-lottery-issue-sp')
        draw_number = draw_number_tag.text.strip() if draw_number_tag else 'N/A'
        
        # 抽せん日 (draw date)
        draw_date_tag = section.select_one('span.js-lottery-date-sp')
        draw_date = draw_date_tag.text.strip() if draw_date_tag else 'N/A'
        
        # 本数字 (main numbers)
        main_number_tag = section.select_one('p.section__text b.js-lottery-number-sp')
        main_numbers = main_number_tag.text.strip() if main_number_tag else 'N/A'
        
        # ボーナス数字 (bonus numbers)
        bonus_number_tag = section.select_one('p.section__text b.js-lottery-bonus-sp')
        bonus_numbers = bonus_number_tag.text.strip() if bonus_number_tag else 'N/A'
        
        # データの追加
        parsed_data.append({
            '回別': draw_number,
            '抽せん日': draw_date,
            '本数字': main_numbers,
            'ボーナス数字': bonus_numbers
        })
    
    return parsed_data

# パス設定
dir_name = os.path.dirname(os.path.abspath(__file__))
html_path = os.path.join(dir_name, 'html', '*.html')

# 全てのパース結果を格納するリストを初期化
d_list = []

# 各 HTML ファイルをループしてパース
for path in glob(html_path):
    with open(path, 'r', encoding='utf-8') as f:
        html = f.read()
    soup = BeautifulSoup(html, 'lxml')
    parsed_dicts = parse(soup)
    d_list += parsed_dicts

# データを DataFrame に変換
df = pd.DataFrame(d_list)

# DataFrameの内容を確認
print("DataFrame content:")
print(df)

# データが空か確認
if df.empty:
    print("保存するデータがありません。")
else:
    # '回別' 列が空白の行を削除
    df = df[df['回別'].notna() & (df['回別'] != 'N/A') & (df['回別'] != '')]

    # '抽せん日' 列を日付型に変換
    df['抽せん日'] = pd.to_datetime(df['抽せん日'], format='%Y年%m月%d日', errors='coerce')

    # 日付が無効な行を削除
    df = df.dropna(subset=['抽せん日'])

    # 日付順にソート
    df = df.sort_values(by='抽せん日')

    # CSV ファイルにエクスポート
    output_path = os.path.join(dir_name, 'loto7.csv')
    df.to_csv(output_path, index=False, encoding='utf-8-sig')

    print(f"データが {output_path} に保存されました。")
