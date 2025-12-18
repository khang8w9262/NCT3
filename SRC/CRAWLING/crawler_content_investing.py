from seleniumbase import Driver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
import pandas as pd
import time
import os
import random
import sys
import subprocess
import re
from datetime import datetime

# --- CẤU HÌNH ---
PAGE_LOAD_TIMEOUT = 60 
GLOBAL_PROCESSED_URLS = set()

# --- HÀM DIỆT PROCESS CŨ ---
def kill_browser_processes():
    try:
        if sys.platform == "win32":
            subprocess.call("taskkill /f /im chromedriver.exe >nul 2>&1", shell=True)
            subprocess.call("taskkill /f /im chrome.exe >nul 2>&1", shell=True)
    except:
        pass

# --- KHỞI TẠO DRIVER (BẢN FIX LỖI SẬP) ---
def init_driver():
    print("Đang khởi động trình duyệt (Chế độ UC Subprocess)...")
    try:
        # uc_subprocess=True: QUAN TRỌNG - Giúp driver chạy tách biệt, không bị sập kết nối
        driver = Driver(uc=True, headless=False, uc_subprocess=True) 
        driver.set_page_load_timeout(PAGE_LOAD_TIMEOUT)
        return driver
    except Exception as e:
        print(f"Lỗi khởi tạo driver: {e}")
        return None

# --- HÀM CHUẨN HÓA NGÀY (DD-MM-YYYY) ---
def format_date_standard(date_str):
    if not date_str or date_str == "Unknown": return "Unknown"
    date_str = date_str.strip()
    
    # 1. Dạng "trước/ago" -> Lấy hôm nay
    if any(x in date_str.lower() for x in ['trước', 'ago', 'min', 'hour', 'sec', 'vừa']):
        return datetime.now().strftime("%d-%m-%Y")

    # 2. Dạng "14/12/2025"
    match = re.search(r'(\d{1,2})[/-](\d{1,2})[/-](\d{4})', date_str)
    if match:
        day, month, year = match.groups()
        return f"{int(day):02d}/{int(month):02d}/{year}"

    # 3. Dạng tiếng Anh "Dec 14, 2025"
    try:
        clean_str = date_str.replace("(", "").replace(")", "").strip()
        for fmt in ["%b %d, %Y", "%B %d, %Y"]:
            try:
                dt = datetime.strptime(clean_str, fmt)
                return dt.strftime("%d/%m/%Y")
            except: continue
    except: pass
    
    return date_str

# --- XỬ LÝ CLOUDFLARE ---
def handle_cloudflare_check(driver):
    try:
        title = driver.title.lower()
        # Kiểm tra kỹ cả Title và Source để bắt Captcha
        if "just a moment" in title or "verify you are human" in driver.page_source.lower():
            print("\nDÍNH CAPTCHA! Vui lòng CLICK xác thực trên trình duyệt...")
            # Chờ tối đa 60s
            for i in range(60):
                time.sleep(1)
                if "just a moment" not in driver.title.lower():
                    print("Đã vượt qua! Tiếp tục...")
                    time.sleep(2) 
                    return True
            return False
    except:
        pass
    return True

# --- CÁC HÀM HỖ TRỢ ---
def visual_wait(seconds, message="Đang đợi"):
    for i in range(seconds, 0, -1):
        sys.stdout.write(f"\r{message}: {i}s...   ")
        sys.stdout.flush()
        time.sleep(1)
    sys.stdout.write("\r" + " " * (len(message) + 15) + "\r")
    sys.stdout.flush()

def get_equity_url(symbol_or_name):
    mapping = {
        'apple': 'apple-computer-inc', 'aapl': 'apple-computer-inc',
        'vinamilk': 'vietnam-dairy-products-jsc', 'vnm': 'vietnam-dairy-products-jsc',
        'vietcombank': 'joint-stock-commercial-bank-for-foreign-trade-of-viet', 'vcb': 'joint-stock-commercial-bank-for-foreign-trade-of-viet',
        'fpt': 'fpt-corp',
        'vietinbank': 'vietnam-joint-stock-commercial-bank-for-industry-and-trade', 'ctg': 'vietnam-joint-stock-commercial-bank-for-industry-and-trade',
        'bidv': 'joint-stock-commercial-bank-for-investment-and-developmen', 'bid': 'joint-stock-commercial-bank-for-investment-and-developmen',
        'google': 'google-inc', 'googl': 'google-inc', 'goog': 'google-inc-c',
        'alibaba': 'alibaba', 'baba': 'alibaba',
        
    }
    key = symbol_or_name.strip().lower()
    if key in mapping:
        return f"https://vn.investing.com/equities/{mapping[key]}-news"
    if key.startswith('https://vn.investing.com/equities/'):
        return key
    return f"https://vn.investing.com/equities/{key}-news"

def is_header_already_scraped(header_text, ticker_name):
    filename = f"{ticker_name.upper()}.csv"
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "DATASET", "SENTIMENT")
    out_path = os.path.join(out_dir, filename)
    if os.path.exists(out_path):
        try:
            df = pd.read_csv(out_path, encoding="utf-8-sig")
            if 'Header' in df.columns and header_text in df['Header'].values:
                return True
        except: pass
    return False

def save_to_csv(df, ticker_name):
    filename = f"{ticker_name.upper()}.csv"
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "DATASET", "SENTIMENT")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, filename)
    if 'URL' in df.columns: df = df.drop(columns=['URL'])
    # Normalize Date column to DD-MM-YYYY for both new and existing data
    try:
        if 'Date' in df.columns:
            df['Date'] = df['Date'].apply(lambda x: format_date_standard(x) if pd.notnull(x) else x)
    except Exception:
        pass

    if os.path.exists(out_path):
        try:
            existing_df = pd.read_csv(out_path, encoding="utf-8-sig")
            if 'Date' in existing_df.columns:
                try:
                    existing_df['Date'] = existing_df['Date'].apply(lambda x: format_date_standard(x) if pd.notnull(x) else x)
                except Exception:
                    pass
            combined_df = pd.concat([existing_df, df], ignore_index=True)
            combined_df = combined_df.drop_duplicates(subset=['Header'], keep='first')
            combined_df.to_csv(out_path, index=False, encoding="utf-8-sig")
        except Exception:
            df.to_csv(out_path, index=False, encoding="utf-8-sig", mode='a', header=False)
    else:
        df.to_csv(out_path, index=False, encoding="utf-8-sig")

def get_article_content(driver):
    try:
        handle_cloudflare_check(driver)
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "h1")))
        # 1. Ngày đăng (sử dụng hàm trích xuất mạnh mẽ)
        article_date = extract_article_date(driver)

        # 2. Header
        header = "Unknown"
        try:
            header = driver.find_element(By.TAG_NAME, "h1").text.strip()
        except: pass
            
        # 3. Content
        content = ""
        try:
            paragraphs = driver.find_elements(By.XPATH, "//div[contains(@class, 'articlePage')]//p")
            if not paragraphs:
                paragraphs = driver.find_elements(By.XPATH, "//div[contains(@class, 'WYSIWYG')]//p")
            special_text = "Bai viet nay duoc tao va dich voi su ho tro cua AI"
            for p in paragraphs:
                p_text = p.text.strip()
                if special_text in p_text: break
                if p_text: content += p_text + "\n"
        except: content = "Content not found"
            
        return article_date, header, content
    except: return "Unknown", "Unknown", "Error"


def extract_article_date(driver):
    """Try multiple strategies to extract the article publication date from the page.
    Returns a formatted date string (DD-MM-YYYY) or 'Unknown'.
    """
    # 1) meta tags like article:published_time
    try:
        metas = driver.find_elements(By.XPATH, "//meta[@property='article:published_time' or @property='og:article:published_time' or @name='pubdate' or @name='article:published_time']")
        for m in metas:
            content = m.get_attribute('content')
            if content:
                # content often like 2025-12-14T08:00:00Z or Dec 14, 2025
                # normalize by taking date part or parsing
                # Try ISO-like
                iso_match = re.search(r"(\d{4}-\d{2}-\d{2})", content)
                if iso_match:
                    return format_date_standard(iso_match.group(1))
                return format_date_standard(content)
    except: pass

    # 2) <time datetime="..."> tags
    try:
        times = driver.find_elements(By.TAG_NAME, 'time')
        for t in times:
            dt = t.get_attribute('datetime')
            text = t.text
            if dt:
                iso_match = re.search(r"(\d{4}-\d{2}-\d{2})", dt)
                if iso_match:
                    return format_date_standard(iso_match.group(1))
            if text:
                # if the time element has readable text
                parsed = format_date_standard(text)
                if parsed != text and parsed != 'Unknown':
                    return parsed
    except: pass

    # 3) Look for nodes containing 'Published' or 'Ngày đăng' or variants
    try:
        candidates = driver.find_elements(By.XPATH, "//*[contains(text(), 'Published') or contains(text(), 'Published:') or contains(text(), 'Ngày đăng') or contains(text(), 'Ngày đăng:') or contains(text(), 'Ngay dang')]")
        for c in candidates:
            txt = c.text.strip()
            # remove label words
            txt_clean = re.sub(r"(?i)Published[:]?")
            txt_clean = txt.lower().replace('published', '').replace('ngày đăng', '').replace('ngay dang', '').replace(':', '').strip()
            if txt_clean:
                parsed = format_date_standard(txt_clean)
                if parsed != 'Unknown':
                    return parsed
    except: pass

    # 4) Regex search in page source for ISO datetime
    try:
        src = driver.page_source
        m = re.search(r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})", src)
        if m:
            return format_date_standard(m.group(1))
        m2 = re.search(r"(\d{1,2}[/-]\d{1,2}[/-]\d{4})", src)
        if m2:
            return format_date_standard(m2.group(1))
    except: pass

    return 'Unknown'

def safe_get(driver, url):
    try:
        driver.get(url)
    except TimeoutException: pass 
    except Exception as e: raise e

def process_page(driver, base_url, page_num, ticker_filename):
    url = base_url if page_num == 1 else f"{base_url}/{page_num}"
    print(f"\n--- ĐANG MỞ TRANG: {page_num} ---")
    print(f"Link: {url}")
    
    safe_get(driver, url)
    handle_cloudflare_check(driver)
    
    try: driver.execute_script("window.scrollTo(0, 500);")
    except: pass

    try:
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR, "article[data-test='article-item']")))
    except:
        if "Just a moment" in driver.title:
            print("Dính Captcha. Đợi xử lý...")
            handle_cloudflare_check(driver)
            try: WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.CSS_SELECTOR, "article[data-test='article-item']")))
            except: return False
        else:
            print(f"Hết bài hoặc lỗi trang {page_num}.")
            return False

    article_items = driver.find_elements(By.CSS_SELECTOR, "article[data-test='article-item']")
    links_to_visit = []
    
    for article in article_items:
        try:
            if article.find_elements(By.XPATH, ".//*[contains(@class, 'fa-lock')]") or "PRO" in article.text: continue
            link_elem = article.find_element(By.CSS_SELECTOR, "a[data-test='article-title-link']")
            links_to_visit.append(link_elem.get_attribute("href"))
        except: continue

    if not links_to_visit: return False
    print(f"Tìm thấy {len(links_to_visit)} bài.")
    
    articles_data = []
    
    for idx, article_url in enumerate(links_to_visit, 1):
        if article_url in GLOBAL_PROCESSED_URLS: continue
        sys.stdout.write(f"   [Bài {idx}/{len(links_to_visit)}] Đang đọc... ")
        sys.stdout.flush()
        
        main_window = driver.current_window_handle
        try:
            driver.execute_script("window.open(arguments[0], '_blank');", article_url)
            time.sleep(1)
            driver.switch_to.window([w for w in driver.window_handles if w != main_window][0])
            
            a_date, a_header, a_content = get_article_content(driver)
            
            if is_header_already_scraped(a_header, ticker_filename):
                print(f"Đã có (Skip)")
            elif len(a_content) > 50:
                articles_data.append([a_date, a_header, a_content, article_url])
                print(f"OK ({a_date}): {a_header[:30]}...")
            elif "just a moment" in driver.title.lower():
                print("Dính Captcha...")
            else:
                print("Ngắn/Lỗi.")
                GLOBAL_PROCESSED_URLS.add(article_url)

            if "just a moment" not in driver.title.lower():
                GLOBAL_PROCESSED_URLS.add(article_url)
            
            driver.close()
            driver.switch_to.window(main_window)
            time.sleep(1)
        except Exception as e:
            if any(x in str(e) for x in ["WinError", "Refused", "died"]): raise e
            try: 
                if len(driver.window_handles) > 1: driver.close()
                driver.switch_to.window(main_window)
            except: pass

    if articles_data:
        df = pd.DataFrame(articles_data, columns=["Date", "Header", "Content", "URL"])
        df["Content"] = df["Content"].astype(str)
        save_to_csv(df, ticker_filename)
        print(f"Đã lưu {len(articles_data)} bài.")
    
    return True

def crawl_investing():
    kill_browser_processes()
    symbol = input("Nhập mã chứng khoán (VNM, ALIBABA...): ").strip()
    ticker = symbol.upper()
    base_url = get_equity_url(symbol)
    
    # --- START DRIVER ---
    driver = init_driver()
    if not driver: return
    
    # Đợi driver ổn định
    time.sleep(4) 
    
    page = 1
    while page <= 300:
        try:
            if driver is None: 
                driver = init_driver()
                time.sleep(4)
            
            if not process_page(driver, base_url, page, ticker):
                print(" Hết bài/Lỗi. Chuyển trang...")
            page += 1
            
        except Exception as e:
            print(f"\n LỖI SYSTEM: {str(e)[:50]}... -> Restarting")
            try: driver.quit()
            except: pass
            kill_browser_processes()
            driver = None
            time.sleep(3)

    print("\n HOÀN TẤT!")
    if driver: driver.quit()

if __name__ == "__main__":
    crawl_investing()