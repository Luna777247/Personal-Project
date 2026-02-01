"""
MB Bank Website Scraper
Scrape product information, interest rates, and loan terms
"""
import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Optional
import json
import time
from pathlib import Path
import logging
from datetime import datetime
from urllib.parse import urljoin, urlparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MBBankScraper:
    """Scraper for MB Bank website"""
    
    def __init__(self, base_url: str = "https://www.mbbank.com.vn", delay: int = 2):
        """
        Initialize scraper
        
        Args:
            base_url: Base URL of MB Bank website
            delay: Delay between requests (seconds)
        """
        self.base_url = base_url
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        self.scraped_data = []
    
    def scrape_page(self, url: str) -> Optional[Dict]:
        """
        Scrape single page
        
        Args:
            url: Page URL
            
        Returns:
            Scraped data dictionary
        """
        try:
            logger.info(f"Scraping: {url}")
            
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract data
            data = {
                'url': url,
                'scraped_at': datetime.now().isoformat(),
                'title': self._extract_title(soup),
                'content': self._extract_content(soup),
                'products': self._extract_products(soup),
                'interest_rates': self._extract_interest_rates(soup),
                'metadata': self._extract_metadata(soup)
            }
            
            # Delay to be polite
            time.sleep(self.delay)
            
            return data
            
        except Exception as e:
            logger.error(f"Failed to scrape {url}: {e}")
            return None
    
    def _extract_title(self, soup: BeautifulSoup) -> str:
        """Extract page title"""
        # Try different title selectors
        for selector in ['h1', 'h2.product-title', '.title', 'title']:
            element = soup.select_one(selector)
            if element:
                return element.get_text(strip=True)
        return ""
    
    def _extract_content(self, soup: BeautifulSoup) -> List[str]:
        """Extract main content paragraphs"""
        content = []
        
        # Remove script and style tags
        for tag in soup(['script', 'style', 'nav', 'footer']):
            tag.decompose()
        
        # Extract paragraphs
        for p in soup.find_all('p'):
            text = p.get_text(strip=True)
            if text and len(text) > 20:
                content.append(text)
        
        # Extract divs with content class
        for div in soup.select('.content, .description, .detail'):
            text = div.get_text(strip=True)
            if text and len(text) > 20:
                content.append(text)
        
        return content
    
    def _extract_products(self, soup: BeautifulSoup) -> List[Dict]:
        """Extract product information"""
        products = []
        
        # Look for product cards/sections
        product_elements = soup.select('.product-item, .product-card, .service-item')
        
        for element in product_elements:
            product = {}
            
            # Product name
            name_elem = element.select_one('h3, h4, .product-name, .title')
            if name_elem:
                product['name'] = name_elem.get_text(strip=True)
            
            # Description
            desc_elem = element.select_one('p, .description, .summary')
            if desc_elem:
                product['description'] = desc_elem.get_text(strip=True)
            
            # Benefits
            benefits = []
            benefit_elems = element.select('.benefit, .feature li')
            for benefit in benefit_elems:
                benefits.append(benefit.get_text(strip=True))
            if benefits:
                product['benefits'] = benefits
            
            if product:
                products.append(product)
        
        return products
    
    def _extract_interest_rates(self, soup: BeautifulSoup) -> List[Dict]:
        """Extract interest rate information"""
        rates = []
        
        # Look for rate tables
        tables = soup.select('table.interest-rate, table.rate-table, .rate-info table')
        
        for table in tables:
            rows = table.select('tr')
            
            for row in rows[1:]:  # Skip header
                cells = row.select('td, th')
                
                if len(cells) >= 2:
                    rate_info = {
                        'term': cells[0].get_text(strip=True),
                        'rate': cells[1].get_text(strip=True)
                    }
                    
                    if len(cells) >= 3:
                        rate_info['condition'] = cells[2].get_text(strip=True)
                    
                    rates.append(rate_info)
        
        # Also look for rate divs
        rate_divs = soup.select('.interest-rate, .rate-item')
        for div in rate_divs:
            rate_text = div.get_text(strip=True)
            if rate_text and '%' in rate_text:
                rates.append({'description': rate_text})
        
        return rates
    
    def _extract_metadata(self, soup: BeautifulSoup) -> Dict:
        """Extract metadata"""
        metadata = {}
        
        # Meta tags
        meta_desc = soup.select_one('meta[name="description"]')
        if meta_desc:
            metadata['description'] = meta_desc.get('content', '')
        
        meta_keywords = soup.select_one('meta[name="keywords"]')
        if meta_keywords:
            metadata['keywords'] = meta_keywords.get('content', '')
        
        return metadata
    
    def scrape_multiple_pages(self, urls: List[str]) -> List[Dict]:
        """
        Scrape multiple pages
        
        Args:
            urls: List of URLs to scrape
            
        Returns:
            List of scraped data
        """
        results = []
        
        for url in urls:
            data = self.scrape_page(url)
            if data:
                results.append(data)
        
        self.scraped_data.extend(results)
        return results
    
    def discover_pages(self, start_url: str, max_depth: int = 2) -> List[str]:
        """
        Discover pages by following links
        
        Args:
            start_url: Starting URL
            max_depth: Maximum depth to crawl
            
        Returns:
            List of discovered URLs
        """
        visited = set()
        to_visit = [(start_url, 0)]
        discovered = []
        
        base_domain = urlparse(self.base_url).netloc
        
        while to_visit:
            url, depth = to_visit.pop(0)
            
            if url in visited or depth > max_depth:
                continue
            
            visited.add(url)
            discovered.append(url)
            
            if depth < max_depth:
                try:
                    response = self.session.get(url, timeout=30)
                    soup = BeautifulSoup(response.content, 'html.parser')
                    
                    # Find links
                    for link in soup.find_all('a', href=True):
                        href = link['href']
                        full_url = urljoin(url, href)
                        
                        # Only follow links within same domain
                        if urlparse(full_url).netloc == base_domain:
                            if full_url not in visited:
                                to_visit.append((full_url, depth + 1))
                    
                    time.sleep(self.delay)
                    
                except Exception as e:
                    logger.error(f"Failed to discover from {url}: {e}")
        
        return discovered
    
    def save_data(self, output_path: str):
        """Save scraped data to JSON file"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.scraped_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved {len(self.scraped_data)} pages to {output_path}")


def scrape_mb_bank(output_path: str = "data/raw/scraped_data.json"):
    """
    Main function to scrape MB Bank website
    
    Args:
        output_path: Path to save scraped data
    """
    logger.info("Starting MB Bank scraper...")
    
    scraper = MBBankScraper()
    
    # Target URLs
    urls = [
        "https://www.mbbank.com.vn/ca-nhan",
        "https://www.mbbank.com.vn/ca-nhan/san-pham/tiet-kiem",
        "https://www.mbbank.com.vn/ca-nhan/san-pham/the",
        "https://www.mbbank.com.vn/ca-nhan/san-pham/vay",
        "https://www.mbbank.com.vn/ca-nhan/lai-suat",
    ]
    
    # Scrape pages
    results = scraper.scrape_multiple_pages(urls)
    
    logger.info(f"Scraped {len(results)} pages")
    
    # Save data
    scraper.save_data(output_path)
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Scrape MB Bank website")
    parser.add_argument('-o', '--output', default='data/raw/scraped_data.json',
                       help='Output file path')
    parser.add_argument('-u', '--urls', nargs='+',
                       help='Custom URLs to scrape')
    
    args = parser.parse_args()
    
    if args.urls:
        scraper = MBBankScraper()
        scraper.scrape_multiple_pages(args.urls)
        scraper.save_data(args.output)
    else:
        scrape_mb_bank(args.output)
