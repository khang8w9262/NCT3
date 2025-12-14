from seleniumbase import Driver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
import time
from retrying import retry
import os
import random

# Hàm retry nếu gặp lỗi
def retry_if_exception(exception):
    return isinstance(exception, Exception)

@retry(wait_exponential_multiplier=1000, wait_exponential_max=10000, stop_max_attempt_number=3)
def safe_find_element(driver, by, value):
    return driver.find_element(by, value)

@retry(wait_exponential_multiplier=1000, wait_exponential_max=10000, stop_max_attempt_number=3)
def safe_open_url(driver, url):
    driver.uc_open_with_reconnect(url, 4)

# Hàm kiểm tra xem bài viết đã tồn tại trong CSV hay chưa dựa trên URL
def is_article_already_scraped(url, filename="investing_articles_uc.csv"):
    if not os.path.exists(filename):
        return False
        
    try:
        df = pd.read_csv(filename, encoding="utf-8-sig")
        if 'URL' in df.columns and url in df['URL'].values:
            print(f"URL đã tồn tại: {url}")
            return True
        return False
    except Exception as e:
        print(f"Lỗi khi kiểm tra bài viết trùng lặp: {e}")
        return False

# Hàm lưu dữ liệu vào CSV
def save_to_csv(df, symbol=None):
    # Lưu file vào DATASET/SENTIMENT, tên file theo mã chứng khoán (in hoa)
    if symbol is None:
        symbol = "output"
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "DATASET", "SENTIMENT")
    os.makedirs(out_dir, exist_ok=True)
    filename = os.path.join(out_dir, f"{symbol.upper()}.csv")
    # Chỉ lưu 3 cột: Date, Header, Content
    df2 = df[["Date", "Header", "Content"]].copy()
    exists = os.path.exists(filename)
    if exists:
        try:
            existing_df = pd.read_csv(filename, encoding="utf-8-sig")
            combined_df = pd.concat([existing_df, df2])
            combined_df = combined_df.drop_duplicates(subset=["Date", "Header"], keep='first')
            combined_df.to_csv(filename, index=False, encoding="utf-8-sig")
        except Exception as e:
            print(f"Lỗi khi kết hợp dữ liệu: {e}")
            df2.to_csv(filename, index=False, encoding="utf-8-sig", mode='a', header=not exists)
    else:
        df2.to_csv(filename, index=False, encoding="utf-8-sig")

# Hàm lấy chi tiết bài báo từ tab hiện tại
def get_article_details_from_current_tab(driver):
    wait = WebDriverWait(driver, 30)
    current_url = driver.current_url
    
    try:
        driver.execute_script("return document.readyState === 'complete';")
        time.sleep(1)
        date_elem = wait.until(
            EC.presence_of_element_located(
                (By.XPATH, "//div[contains(@class, 'text-warren-gray-700')]//span[contains(text(), 'Ngày đăng')]")
            )
        )
        date_text = date_elem.text.strip()
        article_date = date_text.replace("Ngày đăng", "").strip()
    except Exception:
        article_date = "Unknown"
    
    try:
        header_elem = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "h1")))
        header = header_elem.text.strip()
    except Exception:
        header = "Unknown"
    
    content = ""
    try:
        content_elem = wait.until(
            EC.presence_of_element_located(
                (By.CSS_SELECTOR, "div#article div.article_WYSIWYG__O0uhw.article_articlePage__UMz3q")
            )
        )
        paragraphs = content_elem.find_elements(By.CSS_SELECTOR, "p")
        special_text = "Bài viết này được tạo và dịch với sự hỗ trợ của AI và đã được biên tập viên xem xét."
        for p in paragraphs:
            p_text = p.text.strip()
            if special_text in p_text:
                break
            content += p_text + "\n"
    except Exception:
        content = "Content not found"
    
    return article_date, header, content, current_url

# Tạo một tập hợp để theo dõi URL đã duyệt trong phiên hiện tại
processed_urls = set()

def get_equity_url(symbol_or_name):
    """
    Ánh xạ tên/mã chứng khoán sang URL investing.com
    Có thể mở rộng thêm mã khác nếu cần.
    """
    mapping = {
        # mã hoặc tên: phần cuối URL
        'apple': 'apple-computer-inc',
        'aapl': 'apple-computer-inc',
        'vinamilk': 'vietnam-dairy-products-jsc',
        'vnm': 'vietnam-dairy-products-jsc',
        'vietcombank': 'joint-stock-commercial-bank-for-foreign-trade-of-viet',
        'vcb': 'joint-stock-commercial-bank-for-foreign-trade-of-viet',
        'fpt': 'fpt-corp',
        'fpt corp': 'fpt-corp',
        'vietinbank': 'vietnam-joint-stock-commercial-bank-for-industry-and-trade',
        'ctg': 'vietnam-joint-stock-commercial-bank-for-industry-and-trade',
        'bidv': 'joint-stock-commercial-bank-for-investment-and-developmen',
        'bid': 'joint-stock-commercial-bank-for-investment-and-developmen',
        'google': 'google-inc',
        'googl': 'google-inc',
        'goog': 'google-inc-c',
        # Thêm các mã khác nếu cần
    }
    key = symbol_or_name.strip().lower()
    if key in mapping:
        return f"https://vn.investing.com/equities/{mapping[key]}-news"
    # Nếu nhập đúng slug investing thì dùng luôn
    if key.startswith('https://vn.investing.com/equities/'):
        return key
    # fallback: báo lỗi
    raise ValueError(f"Không tìm thấy mã/tên chứng khoán '{symbol_or_name}'. Hãy cập nhật mapping trong code.")

def crawl_investing():
    symbol_or_name = input("Nhập tên hoặc mã chứng khoán (ví dụ: vinamilk, VNM, vietcombank, VCB): ").strip()
    try:
        base_url = get_equity_url(symbol_or_name)
    except Exception as e:
        print(e)
        return
    driver = Driver(uc=True, undetectable=True)
    articles = []
    total_pages = 300
    try:
        safe_open_url(driver, base_url)
        wait = WebDriverWait(driver, 30)
        wait.until(lambda d: d.execute_script('return document.readyState') == 'complete')
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "article[data-test='article-item']")))
        for page in range(1, total_pages + 1):
            if page > 1:
                url = f"{base_url}/{page}"
                print(f"Opening page: {url}")
                safe_open_url(driver, url)
                wait.until(lambda d: d.execute_script('return document.readyState') == 'complete')
            try:
                wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "article[data-test='article-item']")))
            except Exception as e:
                print(f"Timeout waiting for articles on page {page}: {e}")
                continue
            article_items = driver.find_elements(By.CSS_SELECTOR, "article[data-test='article-item']")
            if not article_items:
                print(f"Không tìm thấy bài báo trên trang {page}")
                break
            for article in article_items:
                try:
                    pro_label_elements = article.find_elements(
                        By.XPATH, ".//div[contains(@class, 'mb-1') and contains(@class, 'mt-2.5') and contains(@class, 'flex')]"
                    )
                    if pro_label_elements:
                        print("Skipping pro article")
                        continue
                    link_elem = safe_find_element(article, By.CSS_SELECTOR, "a[data-test='article-title-link']")
                    article_url = link_elem.get_attribute("href")
                    if article_url in processed_urls:
                        print(f"Đã xử lý URL trong phiên này, bỏ qua: {article_url}")
                        continue
                    if is_article_already_scraped(article_url):
                        print(f"Bài viết đã tồn tại trong CSV, bỏ qua: {article_url}")
                        processed_urls.add(article_url)
                        continue
                    print(f"Found article: {article_url}")
                    driver.execute_script("window.open(arguments[0], '_blank');", article_url)
                    driver.switch_to.window(driver.window_handles[-1])
                    time.sleep(1)
                    article_date, header, content, current_url = get_article_details_from_current_tab(driver)
                    articles.append([article_date, header, content, current_url])
                    processed_urls.add(article_url)
                    driver.close()
                    driver.switch_to.window(driver.window_handles[0])
                except Exception as e:
                    print(f"Error processing article: {e}")
                    if len(driver.window_handles) > 1:
                        driver.close()
                    driver.switch_to.window(driver.window_handles[0])
                    continue
            if articles:
                df = pd.DataFrame(articles, columns=["Date", "Header", "Content", "URL"])
                df["Content"] = df["Content"].astype(str)
                save_to_csv(df, symbol=symbol_or_name)
                articles = []
            print(f"Đã cào xong trang {page}")
            time.sleep(random.uniform(1, 5))  # Độ trễ ngẫu nhiên
        if articles:
            df = pd.DataFrame(articles, columns=["Date", "Header", "Content", "URL"])
            df["Content"] = df["Content"].astype(str)
            save_to_csv(df, symbol=symbol_or_name)
        print(f"Crawling completed and saved to DATASET/SENTIMENT/{symbol_or_name.upper()}.csv")
    finally:
        driver.quit()

if __name__ == "__main__":
    crawl_investing()