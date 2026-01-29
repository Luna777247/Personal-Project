"""
Advanced scraper using Selenium for dynamic content
"""
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from bs4 import BeautifulSoup
import json
import time
from pathlib import Path
import logging
from datetime import datetime
from typing import List, Dict, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SeleniumMBScraper:
    """Selenium-based scraper for dynamic content"""
    
    def __init__(self, headless: bool = True, delay: int = 2):
        """
        Initialize Selenium scraper
        
        Args:
            headless: Run browser in headless mode
            delay: Delay between requests
        """
        self.delay = delay
        self.driver = self._setup_driver(headless)
        self.scraped_data = []
    
    def _setup_driver(self, headless: bool) -> webdriver.Chrome:
        """Setup Chrome driver"""
        options = webdriver.ChromeOptions()
        
        if headless:
            options.add_argument('--headless')
        
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-gpu')
        options.add_argument('--window-size=1920,1080')
        options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
        
        driver = webdriver.Chrome(options=options)
        driver.set_page_load_timeout(30)
        
        return driver
    
    def scrape_page(self, url: str, wait_for_selector: Optional[str] = None) -> Optional[Dict]:
        """
        Scrape page with Selenium
        
        Args:
            url: Page URL
            wait_for_selector: CSS selector to wait for
            
        Returns:
            Scraped data
        """
        try:
            logger.info(f"Scraping with Selenium: {url}")
            
            self.driver.get(url)
            
            # Wait for dynamic content
            if wait_for_selector:
                try:
                    WebDriverWait(self.driver, 10).until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, wait_for_selector))
                    )
                except TimeoutException:
                    logger.warning(f"Timeout waiting for {wait_for_selector}")
            else:
                time.sleep(3)  # Default wait
            
            # Scroll to load lazy content
            self._scroll_page()
            
            # Get page source
            page_source = self.driver.page_source
            soup = BeautifulSoup(page_source, 'html.parser')
            
            # Extract data
            data = {
                'url': url,
                'scraped_at': datetime.now().isoformat(),
                'title': self._extract_title(soup),
                'content': self._extract_content(soup),
                'products': self._extract_products(soup),
                'interest_rates': self._extract_interest_rates(soup),
                'forms': self._extract_forms(soup),
                'screenshots': self._take_screenshot(url)
            }
            
            time.sleep(self.delay)
            
            return data
            
        except Exception as e:
            logger.error(f"Failed to scrape {url}: {e}")
            return None
    
    def _scroll_page(self):
        """Scroll page to load lazy content"""
        try:
            # Get scroll height
            last_height = self.driver.execute_script("return document.body.scrollHeight")
            
            while True:
                # Scroll down
                self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(1)
                
                # Calculate new scroll height
                new_height = self.driver.execute_script("return document.body.scrollHeight")
                
                if new_height == last_height:
                    break
                
                last_height = new_height
        
        except Exception as e:
            logger.error(f"Failed to scroll: {e}")
    
    def _extract_title(self, soup: BeautifulSoup) -> str:
        """Extract page title"""
        for selector in ['h1', 'h2.product-title', '.page-title', 'title']:
            element = soup.select_one(selector)
            if element:
                return element.get_text(strip=True)
        return ""
    
    def _extract_content(self, soup: BeautifulSoup) -> List[str]:
        """Extract main content"""
        content = []
        
        # Remove noise
        for tag in soup(['script', 'style', 'nav', 'footer', 'header']):
            tag.decompose()
        
        # Extract paragraphs
        for p in soup.find_all(['p', 'li']):
            text = p.get_text(strip=True)
            if text and len(text) > 15:
                content.append(text)
        
        return content
    
    def _extract_products(self, soup: BeautifulSoup) -> List[Dict]:
        """Extract product cards"""
        products = []
        
        selectors = [
            '.product-item', '.product-card', '.service-item',
            '.product-box', '.item-product', '[class*="product"]'
        ]
        
        for selector in selectors:
            elements = soup.select(selector)
            if elements:
                for element in elements:
                    product = self._parse_product_element(element)
                    if product:
                        products.append(product)
                break
        
        return products
    
    def _parse_product_element(self, element) -> Optional[Dict]:
        """Parse individual product element"""
        product = {}
        
        # Name
        for selector in ['h3', 'h4', '.product-name', '.title', 'strong']:
            name_elem = element.select_one(selector)
            if name_elem:
                product['name'] = name_elem.get_text(strip=True)
                break
        
        # Description
        for selector in ['p', '.description', '.summary', '.detail']:
            desc_elem = element.select_one(selector)
            if desc_elem:
                product['description'] = desc_elem.get_text(strip=True)
                break
        
        # Benefits
        benefits = []
        for li in element.select('ul li, .benefit, .feature'):
            text = li.get_text(strip=True)
            if text:
                benefits.append(text)
        if benefits:
            product['benefits'] = benefits
        
        # Link
        link_elem = element.select_one('a[href]')
        if link_elem:
            product['link'] = link_elem.get('href')
        
        return product if product else None
    
    def _extract_interest_rates(self, soup: BeautifulSoup) -> List[Dict]:
        """Extract interest rate tables"""
        rates = []
        
        # Find tables
        tables = soup.find_all('table')
        
        for table in tables:
            # Check if it's an interest rate table
            table_text = table.get_text().lower()
            if any(keyword in table_text for keyword in ['lãi suất', 'kỳ hạn', 'interest', 'rate']):
                rows = table.select('tr')
                
                for row in rows[1:]:  # Skip header
                    cells = row.select('td, th')
                    
                    if len(cells) >= 2:
                        rate_info = {}
                        
                        for i, cell in enumerate(cells):
                            text = cell.get_text(strip=True)
                            if i == 0:
                                rate_info['term'] = text
                            elif i == 1:
                                rate_info['rate'] = text
                            elif i == 2:
                                rate_info['condition'] = text
                        
                        if rate_info:
                            rates.append(rate_info)
        
        return rates
    
    def _extract_forms(self, soup: BeautifulSoup) -> List[Dict]:
        """Extract forms information"""
        forms = []
        
        for form in soup.find_all('form'):
            form_data = {
                'action': form.get('action', ''),
                'method': form.get('method', ''),
                'fields': []
            }
            
            # Extract input fields
            for input_field in form.find_all(['input', 'select', 'textarea']):
                field = {
                    'type': input_field.get('type', input_field.name),
                    'name': input_field.get('name', ''),
                    'id': input_field.get('id', ''),
                    'placeholder': input_field.get('placeholder', '')
                }
                form_data['fields'].append(field)
            
            if form_data['fields']:
                forms.append(form_data)
        
        return forms
    
    def _take_screenshot(self, url: str) -> Optional[str]:
        """Take screenshot of page"""
        try:
            # Create screenshots directory
            screenshot_dir = Path('data/screenshots')
            screenshot_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate filename from URL
            filename = url.replace('https://', '').replace('http://', '').replace('/', '_')
            filename = f"{filename}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            
            filepath = screenshot_dir / filename
            self.driver.save_screenshot(str(filepath))
            
            return str(filepath)
        
        except Exception as e:
            logger.error(f"Failed to take screenshot: {e}")
            return None
    
    def click_and_scrape(self, url: str, button_selector: str) -> List[Dict]:
        """
        Click button and scrape resulting content
        
        Args:
            url: Page URL
            button_selector: CSS selector for button
            
        Returns:
            List of scraped data
        """
        results = []
        
        try:
            self.driver.get(url)
            time.sleep(2)
            
            # Find and click button
            button = WebDriverWait(self.driver, 10).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, button_selector))
            )
            button.click()
            
            time.sleep(2)
            
            # Scrape new content
            page_source = self.driver.page_source
            soup = BeautifulSoup(page_source, 'html.parser')
            
            data = {
                'url': url,
                'clicked_element': button_selector,
                'content': self._extract_content(soup)
            }
            
            results.append(data)
        
        except Exception as e:
            logger.error(f"Failed to click and scrape: {e}")
        
        return results
    
    def scrape_multiple_pages(self, urls: List[str]) -> List[Dict]:
        """Scrape multiple pages"""
        results = []
        
        for url in urls:
            data = self.scrape_page(url)
            if data:
                results.append(data)
        
        self.scraped_data.extend(results)
        return results
    
    def save_data(self, output_path: str):
        """Save scraped data"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.scraped_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved {len(self.scraped_data)} pages to {output_path}")
    
    def close(self):
        """Close browser"""
        if self.driver:
            self.driver.quit()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def scrape_with_selenium(urls: List[str], output_path: str = "data/raw/selenium_scraped.json"):
    """
    Scrape URLs using Selenium
    
    Args:
        urls: List of URLs
        output_path: Output file path
    """
    with SeleniumMBScraper(headless=True) as scraper:
        results = scraper.scrape_multiple_pages(urls)
        scraper.save_data(output_path)
        return results


if __name__ == "__main__":
    # Example usage
    urls = [
        "https://www.mbbank.com.vn/ca-nhan/san-pham/tiet-kiem",
        "https://www.mbbank.com.vn/ca-nhan/san-pham/the",
    ]
    
    scrape_with_selenium(urls)
