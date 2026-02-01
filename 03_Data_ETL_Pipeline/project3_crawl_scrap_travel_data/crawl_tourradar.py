import os
import requests
try:
    from bs4 import BeautifulSoup
    _BS4_AVAILABLE = True
except Exception:
    BeautifulSoup = None
    _BS4_AVAILABLE = False
from datetime import datetime, timezone
import logging
import re
import time
import json
try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    _SELENIUM_AVAILABLE = True
except Exception:
    webdriver = None
    By = None
    Options = None
    WebDriverWait = None
    EC = None
    _SELENIUM_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Headers to mimic browser
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

# Output file
output_file = "tours.json"

url = "https://www.tourradar.com/d/tours"

def check_robots_txt(base_url):
    """Check robots.txt to ensure scraping is allowed."""
    robots_url = base_url.rstrip('/') + '/robots.txt'
    try:
        response = requests.get(robots_url, headers=headers)
        if response.status_code == 200:
            robots_content = response.text
            if 'Disallow: /' in robots_content or 'Disallow: *' in robots_content:
                logging.warning("Robots.txt disallows scraping for all user agents.")
                return False
            else:
                logging.info("Robots.txt allows scraping.")
                return True
        else:
            logging.warning(f"Could not fetch robots.txt: {response.status_code}")
            return True  # Assume allowed if not found
    except Exception as e:
        logging.warning(f"Error checking robots.txt: {e}")
        return True  # Assume allowed

def parse_tour_text(text):
    """Parse the tour text to extract title, duration, rating, price."""
    parts = text.split('•')
    first_part = parts[0]
    
    # Duration
    duration_match = re.search(r'(\d+) days?', first_part)
    duration = int(duration_match.group(1)) if duration_match else 1
    
    # Title
    if 'off' in first_part.lower():
        # Discount tour
        title_match = re.search(r'off(.+?)\d+ days?', first_part, re.IGNORECASE)
        title = title_match.group(1).strip() if title_match else first_part.split('off')[1].split(str(duration))[0].strip()
    else:
        title = first_part[:duration_match.start()].strip() if duration_match else first_part.strip()
    
    # Rating and price
    rating = None
    reviews = None
    original_price = None
    discounted_price = None
    if len(parts) > 1:
        second_part = parts[1]
        rating_match = re.search(r'(\d+\.\d+)\(([\d,]+)\)', second_part)
        if rating_match:
            rating = float(rating_match.group(1))
            reviews = int(rating_match.group(2).replace(',', ''))
        
        price_match = re.search(r'From\$([\d,]+)\$([\d,]+)', second_part)
        if price_match:
            original_price = int(price_match.group(1).replace(',', ''))
            discounted_price = int(price_match.group(2).replace(',', ''))
    else:
        # Check in first_part
        price_match = re.search(r'from\$([\d,]+)\$([\d,]+)', first_part, re.IGNORECASE)
        if price_match:
            original_price = int(price_match.group(1).replace(',', ''))
            discounted_price = int(price_match.group(2).replace(',', ''))
    
    return {
        'title': title,
        'duration_days': duration,
        'rating': rating,
        'reviews': reviews,
        'original_price': original_price,
        'discounted_price': discounted_price
    }

def scrape_tour_details(url):
    """Scrape detailed information from a tour page by automatically discovering sections."""
    max_retries = 3
    delay = 1
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            if not _BS4_AVAILABLE:
                logging.error("BeautifulSoup (bs4) is not installed. Cannot parse HTML.")
                return {}

            # Helper to normalize section keys
            def normalize_key(s):
                key = re.sub(r'[^\w\s]', '', s or '')
                return re.sub(r'\s+', ' ', key).strip().lower()

            # Automatically discover sections based on h2 headers
            sections = {}
            h2_elements = soup.find_all('h2')
            for h2 in h2_elements:
                section_name = h2.get_text(strip=True)
                content = extract_section_content(h2)
                if content:
                    # store under original and normalized keys (if different)
                    sections[section_name] = content
                    nk = normalize_key(section_name)
                    if nk and nk != section_name:
                        sections[nk] = content
            
            # Special handling for operator (if exists)
            operator_link = soup.find('a', href=re.compile(r'/o/'))
            if operator_link:
                sections['operator'] = operator_link.get_text(strip=True)
            
            return sections
        except Exception as e:
            logging.warning(f"Attempt {attempt+1} failed for {url}: {e}")
            if attempt < max_retries - 1:
                time.sleep(delay)
                delay *= 2  # exponential backoff
            else:
                logging.error(f"Failed to scrape {url} after {max_retries} attempts")
                return {}

def extract_section_content(h2):
    """Extract content after an h2 element."""
    content = []
    max_text_len = 5000
    current = h2.find_next_sibling()
    while current and current.name != 'h2':
        if current.name in ['ul', 'ol']:
            # Extract list items
            items = [li.get_text(strip=True) for li in current.find_all('li')]
            # truncate long items
            items = [it if len(it) <= max_text_len else it[:max_text_len] + '...' for it in items]
            content.append({'type': 'list', 'items': items})
        elif current.name == 'div':
            # Check for nested structure like itinerary or inclusions
            if current.find('h3'):
                # Nested categories like inclusions
                sub_sections = {}
                for h3 in current.find_all('h3'):
                    sub_name = h3.get_text(strip=True)
                    sub_ul = h3.find_next('ul')
                    if sub_ul:
                        sub_items = [li.get_text(strip=True) for li in sub_ul.find_all('li')]
                        sub_sections[sub_name] = sub_items
                content.append({'type': 'nested', 'data': sub_sections})
            else:
                # Plain text or paragraphs
                text = current.get_text(separator='\n', strip=True)
                if text:
                    if len(text) > max_text_len:
                        text = text[:max_text_len] + '...'
                    content.append({'type': 'text', 'content': text})
        elif current.name == 'p':
            text = current.get_text(strip=True)
            if text:
                if len(text) > max_text_len:
                    text = text[:max_text_len] + '...'
                content.append({'type': 'text', 'content': text})
        current = current.find_next_sibling()
    
    # Simplify if only one item
    if len(content) == 1:
        return content[0]
    elif content:
        return content
    return None

def scrape_tourradar():
    logging.info("Starting to scrape TourRadar...")
    
    # Check robots.txt
    if not check_robots_txt(url):
        logging.error("Scraping not allowed according to robots.txt. Exiting.")
        return
    if not _SELENIUM_AVAILABLE:
        logging.error("Selenium is not installed in this environment. Install selenium to run the scraper.")
        return
    
    # Set up Selenium (safe init with helpful error logging)
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")

    def init_driver():
        try:
            return webdriver.Chrome(options=options)
        except Exception as e:
            logging.error(f"Failed to start Chrome WebDriver: {e}")
            # Try using CHROMEDRIVER_PATH env if present
            cd_path = os.environ.get('CHROMEDRIVER_PATH')
            if cd_path:
                try:
                    return webdriver.Chrome(executable_path=cd_path, options=options)
                except Exception as e2:
                    logging.error(f"Failed to start Chrome with CHROMEDRIVER_PATH: {e2}")
            logging.error("Ensure ChromeDriver is installed and on PATH, or set CHROMEDRIVER_PATH environment variable.")
            raise

    driver = init_driver()
    
    try:
        driver.get(url)
        # Wait for page to load
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
        
        # Scroll to load all tours
        last_height = driver.execute_script("return document.body.scrollHeight")
        scroll_attempts = 0
        max_scrolls = 50  # Prevent infinite loop
        while scroll_attempts < max_scrolls:
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)
            new_height = driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height
            scroll_attempts += 1
        
        logging.info("Finished scrolling, collecting tour links...")
        
        # Find all tour links
        tour_elements = driver.find_elements(By.CSS_SELECTOR, 'a[href*="/t/"]')
        tour_data = []
        for elem in tour_elements:
            href = elem.get_attribute('href')
            text = elem.text.strip()
            if href and text:
                tour_data.append((href, text))

        # Preserve order when deduplicating (first occurrence wins)
        seen = set()
        unique_tour_data = []
        for href, text in tour_data:
            key = (href, text)
            if key in seen:
                continue
            seen.add(key)
            unique_tour_data.append((href, text))

        logging.info(f"Found {len(unique_tour_data)} unique tours.")
        
        tours = []
        for href, text in unique_tour_data:
            parsed = parse_tour_text(text)
            if parsed:
                tour = {
                    "tour_id": f"tourradar_{href.split('/')[-1]}",
                    "title": parsed['title'],
                    "description": f"Tour from TourRadar: {parsed['title']}",
                    "destination": "Various",  # Could extract from title or link
                    "duration_days": parsed['duration_days'],
                    "rating": parsed['rating'],
                    "reviews": parsed['reviews'],
                    "original_price": parsed['original_price'],
                    "discounted_price": parsed['discounted_price'],
                    "link": href,
                    "source": "TourRadar",
                    "scraped_at": datetime.now(timezone.utc)
                }
                tours.append(tour)
        
        # Scrape details for each tour
        valid_tours = []
        for i, tour in enumerate(tours):
            logging.info(f"Scraping details for tour {i+1}/{len(tours)}: {tour['title']}")
            details = scrape_tour_details(tour['link'])
            tour.update(details)
            if tour.get('title') and tour.get('link'):
                valid_tours.append(tour)
            else:
                logging.warning(f"Skipping invalid tour: {tour.get('tour_id')}")
            time.sleep(1)  # Be polite to the server
        
        # Save to JSON file
        if valid_tours:
            try:
                # Convert datetime to ISO string for JSON serialization
                for tour in valid_tours:
                    if 'scraped_at' in tour and isinstance(tour['scraped_at'], datetime):
                        tour['scraped_at'] = tour['scraped_at'].isoformat()
                
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(valid_tours, f, indent=4, ensure_ascii=False)
                logging.info(f"Saved {len(valid_tours)} tours with details to {output_file}.")
            except Exception as e:
                logging.error(f"Failed to save tours to JSON: {e}")
        else:
            logging.info("No valid tours found.")
    
    except Exception as e:
        logging.error(f"Error scraping TourRadar: {e}")
    finally:
        driver.quit()

if __name__ == "__main__":
    scrape_tourradar()