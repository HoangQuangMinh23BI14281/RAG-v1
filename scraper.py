import requests
from bs4 import BeautifulSoup
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import time

def fetch_article(link, title, description, headers, today):
    try:
        article_resp = requests.get(link, headers=headers, timeout=30)
        article_soup = BeautifulSoup(article_resp.content, "html.parser")
        content = article_soup.select_one("article.fck_detail")
        content_text = content.text.strip() if content else description
        return f"Title: {title}\nLink: {link}\nDate: {today}\nContent: {content_text}\n\n"
    except Exception as e:
        print(f"Lỗi khi đọc bài {link}: {e}")
        return None

def scrape_vnexpress(max_pages=10):
    base_url = "https://vnexpress.net/tin-tuc-24h"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    today = datetime.now().strftime("%Y-%m-%d")
    filename = f"vnexpress_news_{today}.txt"
    
    article_tasks = []

    try:
        for page in range(1, max_pages + 1):
            if page == 1:
                url = base_url
            else:
                url = f"{base_url}-p{page}"
            
            print(f"Đang xử lý trang {page}: {url}")
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, "html.parser")
            articles = soup.select("article.item-news")

            if not articles:
                print("Không tìm thấy thêm bài viết. Dừng lại.")
                break

            for article in articles:
                title_tag = article.select_one("h3.title-news > a")
                desc_tag = article.select_one("p.description > a")

                if title_tag and desc_tag:
                    title = title_tag.get("title", "").strip()
                    link = title_tag.get("href", "").strip()
                    description = desc_tag.text.strip()
                    article_tasks.append((link, title, description, headers, today))
            
            time.sleep(1)  # tránh bị chặn IP

        # Tải đồng thời các bài viết
        print(f"Đang tải nội dung từ {len(article_tasks)} bài viết...")
        with ThreadPoolExecutor(max_workers=30) as executor:
            results = executor.map(lambda args: fetch_article(*args), article_tasks)

        news_data = [res for res in results if res]

        # Ghi ra file
        with open(filename, "w", encoding="utf-8") as f:
            f.writelines(news_data)

        print(f"Đã lưu {len(news_data)} bài viết vào {filename}")
        return filename

    except Exception as e:
        print(f"ERROR: {e}")
        return None
