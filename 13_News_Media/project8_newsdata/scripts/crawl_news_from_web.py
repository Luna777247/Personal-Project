import requests
from bs4 import BeautifulSoup
import trafilatura
import time
import re
import json
import pandas as pd
from datetime import datetime
from urllib.parse import quote, urljoin
from collections import defaultdict
import urllib3
import ssl
from requests.adapters import HTTPAdapter
from urllib3.poolmanager import PoolManager
from urllib3.util.retry import Retry
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import glob
from transformers import pipeline
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set up data directory path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
os.makedirs(DATA_DIR, exist_ok=True)  # Ensure data directory exists

# Tắt warning SSL
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Danh sách các loại thiên tai
DISASTER_TYPES = [
    ("Thiên tai khí tượng – thủy văn", "Bão, áp thấp nhiệt đới"),
    ("Thiên tai khí tượng – thủy văn", "Lốc xoáy, vòi rồng"),
    ("Thiên tai khí tượng – thủy văn", "Mưa lớn kéo dài"),
    ("Thiên tai khí tượng – thủy văn", "Lũ, lũ quét"),
    ("Thiên tai khí tượng – thủy văn", "Ngập úng"),
    ("Thiên tai khí tượng – thủy văn", "Hạn hán"),
    ("Thiên tai khí tượng – thủy văn", "Xâm nhập mặn"),
    ("Thiên tai khí tượng – thủy văn", "Sương muối, rét đậm – rét hại"),
    ("Thiên tai khí tượng – thủy văn", "Nắng nóng, sóng nhiệt"),
    ("Thiên tai địa chất", "Động đất"),
    ("Thiên tai địa chất", "Sóng thần"),
    ("Thiên tai địa chất", "Núi lửa phun"),
    ("Thiên tai địa chất", "Sạt lở đất, trượt đất, sụt lún"),
    ("Thiên tai địa chất", "Hang động karst sụp đổ"),
    ("Thiên tai sinh học", "Dịch bệnh ở người"),
    ("Thiên tai sinh học", "Dịch bệnh ở động vật"),
    ("Thiên tai sinh học", "Dịch bệnh cây trồng"),
    ("Thiên tai sinh học", "Sinh vật ngoại lai xâm hại"),
    ("Thiên tai môi trường – con người gây ra", "Cháy rừng"),
    ("Thiên tai môi trường – con người gây ra", "Ô nhiễm môi trường nghiêm trọng"),
    ("Thiên tai môi trường – con người gây ra", "Tràn dầu"),
    ("Thiên tai môi trường – con người gây ra", "Sự cố hóa chất, phóng xạ")
]

# Cấu hình cho từng trang báo (đã được cập nhật)
NEWS_SOURCES = {
    'vnexpress': {
        'name': 'VnExpress',
        'search_url': 'https://timkiem.vnexpress.net/?q={query}&media_type=text&search_f=title,tag_list&page={page}',
        'article_selector': 'article.item-news a[data-medium="Item-1"], article.item-news h3.title-news a, h3.title-news a[title]',
        'pagination_type': 'url',
        'max_pages': 5,
        'needs_ssl_workaround': False,
        'base_domain': 'vnexpress.net'
    },
    'baotintuc': {
        'name': 'Báo Tin Tức',
        'search_url': 'https://baotintuc.vn/tim-kiem.htm?q={query}&p={page}',
        'article_selector': '.story h2 a, .story__heading a, .item-news h3 a, h3 a.title',
        'pagination_type': 'url',
        'max_pages': 5,
        'needs_ssl_workaround': True,
        'base_domain': 'baotintuc.vn'
    },
    'sggp': {
        'name': 'Sài Gòn Giải Phóng',
        'search_url': 'https://www.sggp.org.vn/tim-kiem/?q={query}&page={page}',
        'article_selector': 'h3.article-title a, h2.article-title a, .story-item h3 a',
        'pagination_type': 'url',
        'max_pages': 5,
        'needs_ssl_workaround': False
    },
    'vietnamnet': {
        'name': 'VietnamNet',
        'search_url': 'https://vietnamnet.vn/tim-kiem-p{page}?q={query}',
        'article_selector': 'h3.vnn-title a, .vnn-search-item h3 a, .horizontalPost__main-title a, .article-title a',
        'pagination_type': 'url',
        'max_pages': 5,
        'needs_ssl_workaround': False
    },
    'dantri': {
        'name': 'Dân Trí',
        'search_url': 'https://dantri.com.vn/tim-kiem/{query}.htm?pi={page}',
        'article_selector': 'h3.article-title a, h4.article-title a, article h2 a',
        'pagination_type': 'url',
        'max_pages': 5,
        'needs_ssl_workaround': False
    },
    'thanhnien': {
        'name': 'Thanh Niên',
        'search_url': 'https://thanhnien.vn/tim-kiem?keywords={query}',
        'article_selector': 'h2.story__heading a, h3.story__heading a, .story h3 a',
        'pagination_type': 'scroll',
        'max_pages': 3,
        'needs_ssl_workaround': False
    },
    'tuoitre': {
        'name': 'Tuổi Trẻ',
        'search_url': 'https://tuoitre.vn/tim-kiem.htm?keywords={query}',
        'article_selector': 'h3.title-news a, a.box-category-link-title, .story h3 a',
        'pagination_type': 'scroll',
        'max_pages': 3,
        'needs_ssl_workaround': False
    },
    'nld': {
        'name': 'Người Lao Động',
        'search_url': 'https://nld.com.vn/tim-kiem.htm?keywords={query}&trang={page}',
        'article_selector': 'h3.art-title a, article a.title-news, .story h2 a, .item-news h3 a',
        'pagination_type': 'url',
        'max_pages': 5,
        'needs_ssl_workaround': False
    },
    'qdnd': {
        'name': 'Quân Đội Nhân Dân',
        'search_url': 'https://www.qdnd.vn/tim-kiem/q/{query}/p/{page}',
        'article_selector': 'h3.cms-title a, h2.cms-title a, .article-item h3 a',
        'pagination_type': 'url',
        'max_pages': 5,
        'needs_ssl_workaround': False
    }
}

# Custom SSL Adapter để xử lý các trang báo dùng SSL cũ
class SSLAdapter(HTTPAdapter):
    def init_poolmanager(self, *args, **kwargs):
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS)
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        # Cho phép các cipher cũ hơn
        ctx.set_ciphers('DEFAULT@SECLEVEL=1')
        # Cho phép unsafe legacy renegotiation (0x00040000)
        ctx.options |= 0x00040000
        kwargs['ssl_context'] = ctx
        return super().init_poolmanager(*args, **kwargs)

class NewsScraperMultiSource:
    def __init__(self):
        self.session = requests.Session()

        # Cấu hình retry strategy
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"]
        )

        # Mount adapter với retry
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

        # Mount SSL adapter cho các trang cần workaround
        ssl_adapter = SSLAdapter()
        self.session.mount("https://", ssl_adapter)

        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'vi,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        })

        self.results = []

    def extract_urls_from_source(self, source_key, query, max_pages=5):
        """Thu thập URLs từ một nguồn báo cụ thể"""
        source = NEWS_SOURCES[source_key]
        all_urls = set()

        print(f"\n{'='*80}")
        print(f"Nguồn: {source['name']} | Truy vấn: {query}")
        print(f"{'='*80}")

        if source['pagination_type'] == 'url':
            for page in range(1, min(max_pages + 1, source['max_pages'] + 1)):
                urls = self._extract_from_page(source_key, query, page)
                if not urls:
                    print(f"  Không tìm thấy bài ở trang {page}. Dừng.")
                    break

                new_urls = urls - all_urls
                if new_urls:
                    all_urls.update(new_urls)
                    print(f"  Trang {page}: Thêm {len(new_urls)} URL mới")
                else:
                    print(f"  Trang {page}: Không có URL mới, dừng phân trang.")
                    break

                time.sleep(1)
        else:
            # Trang dùng infinite scroll - chỉ lấy trang đầu
            urls = self._extract_from_page(source_key, query, 1)
            all_urls.update(urls)

        print(f"  → Tổng URL từ {source['name']}: {len(all_urls)}")
        return list(all_urls)

    def _extract_from_page(self, source_key, query, page):
        """Trích xuất URLs từ một trang cụ thể"""
        source = NEWS_SOURCES[source_key]
        urls = set()

        try:
            # Format query cho từng trang
            if source_key == 'dantri':
                formatted_query = query.replace(' ', '+').replace(',', '%2c')
            else:
                formatted_query = quote(query)

            # Tạo URL tìm kiếm
            if '{page}' in source['search_url']:
                search_url = source['search_url'].format(query=formatted_query, page=page)
            else:
                search_url = source['search_url'].format(query=formatted_query)

            print(f"  Trang {page}: {search_url[:100]}...")

            # Xác định có cần sử dụng SSL workaround không
            verify_ssl = False  # Sử dụng SSL adapter tùy chỉnh cho tất cả

            response = self.session.get(
                search_url,
                timeout=15,
                verify=verify_ssl,
                allow_redirects=True
            )

            if response.status_code != 200:
                print(f"    HTTP {response.status_code}: {response.reason}")
                return []

            response.encoding = 'utf-8'
            soup = BeautifulSoup(response.text, 'html.parser')

            # Tìm các thẻ a theo selector
            articles = soup.select(source['article_selector'])

            # Nếu không tìm thấy, thử các selector chung
            if not articles:
                print(f"    Không tìm thấy với selector chính, thử selector dự phòng...")

                # Thử tìm theo class chứa keyword
                possible_selectors = [
                    'a[href*="/"]',  # Tất cả link
                    'h2 a', 'h3 a', 'h4 a',  # Tiêu đề
                    '.title a', '.news-title a', '.article-title a',  # Class title
                    'article a', '.news-item a', '.story a',  # Container
                ]

                for selector in possible_selectors:
                    articles = soup.select(selector)
                    if articles:
                        print(f"    Tìm thấy {len(articles)} elements với selector: {selector}")
                        break

            # Lọc và xử lý URLs
            base_domain = source.get('base_domain', source['search_url'].split('/')[2])

            for article in articles:
                href = article.get('href', '')
                if not href or href == '#':
                    continue

                # Xử lý URL tương đối
                if href.startswith('/'):
                    protocol = 'https' if 'https://' in source['search_url'] else 'http'
                    href = f"{protocol}://{base_domain}{href}"
                elif href.startswith('//'):
                    href = f"https:{href}"
                elif not href.startswith('http'):
                    continue

                # Skip patterns chung
                skip_patterns = [
                    'video', 'gallery', 'photo', 'javascript:', 'mailto:',
                    '/tag/', '/topic/', '/tags', '-tags', '.tag',
                    '/tin-tuc-24h/', '/category', '/rss', '.rss',
                    'facebook.com', 'twitter.com', 'zalo.me', 'youtube.com',
                    '/lien-he', '/dieu-khoan', '/chinh-sach', '/gioi-thieu',
                    '/static/', '/nguoi-lao-dong-news.htm',
                    '/moitruongdothi/', '/tin-nong-trong-ngay/',
                    '/tieu-dung-thong-minh', '/giai-tri.htm',
                    '/lien-he.htm', '/contact', '/about',
                    '/chuyen-muc/', '/category/', '/chu-de/',
                ]

                # Skip patterns đặc thù cho từng trang
                source_specific_skips = {
                    'nld': ['/suc-khoe/', '/giao-duc/', '/van-hoa/', '/the-thao/'],
                    'vietnamnet': ['/ban-doc/', '/giao-duc/', '/suc-khoe/', '/doi-song/', '/giai-tri/'],
                    'thanhnien': ['/lien-he', '/giai-tri.htm'],
                    'tuoitre': ['/lien-he', '/hoi-dap/'],
                }

                if source_key in source_specific_skips:
                    skip_patterns.extend(source_specific_skips[source_key])

                if any(pattern in href.lower() for pattern in skip_patterns):
                    continue

                # Kiểm tra URL có pattern bài viết
                has_article_pattern = any([
                    re.search(r'-\d{6,}', href),  # Có số dài (ID bài viết)
                    re.search(r'\d{4}-\d{2}-\d{2}', href),  # Có ngày tháng
                    re.search(r'/\d{4}/', href),  # Có năm trong đường dẫn
                    '/thoi-su/' in href, '/xa-hoi/' in href, '/kinh-te/' in href,
                    '.html' in href, '.htm' in href,
                    re.search(r'/\d+\.html$', href),  # Kết thúc bằng số.html
                ])

                # Chỉ lấy URL thuộc domain của nguồn
                if base_domain in href and len(href) > 30:
                    if has_article_pattern or self._looks_like_article_url(href, source_key):
                        urls.add(href)

            print(f"    Tìm thấy: {len(urls)} URLs tiềm năng")

        except Exception as e:
            print(f"    Lỗi khi trích xuất trang {page}: {e}")

        return urls

    def _looks_like_article_url(self, url, source_key):
        """Kiểm tra heuristic xem URL có giống bài viết không"""
        # Các pattern đặc thù cho từng trang
        patterns_by_source = {
            'vnexpress': [r'vnexpress\.net/[^/]+-\d+\.html'],
            'dantri': [r'dantri\.com\.vn/[^/]+/\d+\.htm'],
            'thanhnien': [r'thanhnien\.vn/[^/]+-\d+\.html'],
            'tuoitre': [r'tuoitre\.vn/[^/]+-\d+\.htm'],
            'nld': [r'nld\.com\.vn/[^/]+-\d+\.html'],
            'vietnamnet': [r'vietnamnet\.vn/[^/]+-\d+\.html'],
            'qdnd': [r'qdnd\.vn/[^/]+-\d+\.html'],
            'sggp': [r'sggp\.org\.vn/[^/]+-\d+\.html'],
        }

        if source_key in patterns_by_source:
            for pattern in patterns_by_source[source_key]:
                if re.search(pattern, url):
                    return True

        return False

    def extract_content_and_date(self, url, source_name, disaster_type=None):
        """Trích xuất nội dung và metadata từ URL"""
        max_retries = 2
        for attempt in range(max_retries):
            try:
                # Xác định có cần verify SSL không
                verify_ssl = True
                for source_key, config in NEWS_SOURCES.items():
                    if config['name'] == source_name and config.get('needs_ssl_workaround', False):
                        verify_ssl = False
                        break

                response = self.session.get(url, timeout=20, verify=verify_ssl)
                response.raise_for_status()

                # Dùng Trafilatura
                extracted_data = trafilatura.extract(
                    response.text,
                    output_format='json',
                    with_metadata=True,
                    include_comments=False,
                    include_tables=False,
                    no_fallback=False
                )

                content = "Không có nội dung"
                date = "N/A"
                title = "N/A"

                if extracted_data:
                    try:
                        data_dict = json.loads(extracted_data)
                        content = data_dict.get('text', "Không có nội dung")
                        title = data_dict.get('title', "N/A")
                        date_from_traf = data_dict.get('date')
                        if date_from_traf:
                            # Xử lý nhiều định dạng date
                            date_match = re.search(r'\d{4}-\d{2}-\d{2}', date_from_traf)
                            if date_match:
                                date = date_match.group(0)
                    except:
                        content = "Không có nội dung"

                # Fallback với BeautifulSoup
                soup = BeautifulSoup(response.text, 'html.parser')

                # Trích xuất tiêu đề
                if title == "N/A":
                    # Thử nhiều cách lấy title
                    title_selectors = [
                        'h1', 'h1.title', 'h1.article-title',
                        'h1.title-detail', 'h1.title-news',
                        'meta[property="og:title"]',
                        'meta[name="twitter:title"]',
                        'title'
                    ]

                    for selector in title_selectors:
                        element = soup.select_one(selector)
                        if element:
                            if selector.startswith('meta'):
                                title = element.get('content', 'N/A')
                            else:
                                title = element.get_text(strip=True)
                            if title and title != "N/A":
                                break

                # Validate title dựa trên từ khóa thiên tai
                if disaster_type and title != "N/A":
                    disaster_keywords = disaster_type.lower().split(', ')
                    title_lower = title.lower()
                    has_keyword = any(keyword in title_lower for keyword in disaster_keywords)
                    if not has_keyword:
                        # Kiểm tra trong content nếu có
                        content_lower = content.lower() if content != "Không có nội dung" else ""
                        has_keyword = any(keyword in content_lower for keyword in disaster_keywords)
                        if not has_keyword:
                            print(f"      Bỏ qua bài không liên quan: {title[:50]}...")
                            return None

                # Trích xuất ngày
                if date == "N/A":
                    # Thử các meta tag và time tag
                    date_selectors = [
                        ('meta', {'property': 'article:published_time'}),
                        ('meta', {'name': 'pubdate'}),
                        ('meta', {'property': 'datePublished'}),
                        ('meta', {'name': 'publish_date'}),
                        ('meta', {'itemprop': 'datePublished'}),
                        ('time', {}),
                        ('span', {'class': re.compile(r'date|time|datetime', re.I)}),
                        ('div', {'class': re.compile(r'date|time|datetime', re.I)}),
                    ]

                    for tag, attrs in date_selectors:
                        element = soup.find(tag, attrs)
                        if element:
                            if tag == 'meta' and 'content' in element.attrs:
                                date_str = element['content']
                            else:
                                date_str = element.get('datetime', element.get_text(strip=True))

                            # Tìm ngày trong chuỗi
                            date_match = re.search(r'\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4}|\d{2}-\d{2}-\d{4}', date_str)
                            if date_match:
                                date_found = date_match.group(0)
                                # Chuẩn hóa định dạng
                                if '/' in date_found:
                                    day, month, year = date_found.split('/')
                                    date = f"{year}-{month}-{day}"
                                elif '-' in date_found and len(date_found.split('-')[0]) == 2:
                                    day, month, year = date_found.split('-')
                                    date = f"{year}-{month}-{day}"
                                else:
                                    date = date_found
                                break

                return {
                    'url': url,
                    'title': title[:500],  # Giới hạn độ dài
                    'content': content[:10000],  # Giới hạn nội dung
                    'date': date,
                    'source': source_name,
                    'content_length': len(content),
                    'scrape_time': datetime.now().isoformat()
                }

            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"      Thử lại ({attempt + 1}/{max_retries})...")
                    time.sleep(2)
                else:
                    print(f"      Lỗi trích xuất {url[:80]}: {e}")
                    return None

    def scrape_all_sources(self, disaster_type, category, max_articles_per_source=20, date_from=None, date_to=None):
        """Thu thập từ tất cả các nguồn cho một loại thiên tai"""
        print(f"\n{'#'*80}")
        print(f"Thiên tai: {disaster_type}")
        print(f"Nhóm: {category}")
        if date_from or date_to:
            print(f"Khoảng thời gian: {date_from or 'Không giới hạn'} đến {date_to or 'Không giới hạn'}")
        print(f"{'#'*80}")

        source_results = []

        def process_source(source_key, source_config):
            try:
                print(f"\n{'='*80}")
                print(f"Nguồn: {source_config['name']} | Truy vấn: {disaster_type}")
                print(f"{'='*80}")

                # Thu thập URLs
                urls = self.extract_urls_from_source(
                    source_key,
                    disaster_type,
                    max_pages=source_config['max_pages']
                )

                # Giới hạn số bài viết
                urls = urls[:max_articles_per_source]

                if not urls:
                    print(f"  Không tìm thấy URL nào từ {source_config['name']}")
                    return []

                print(f"  Trích xuất nội dung từ {source_config['name']}...")
                successful_articles = 0
                articles = []

                for idx, url in enumerate(urls, 1):
                    print(f"    [{idx}/{len(urls)}] {url[:70]}...")

                    article_data = self.extract_content_and_date(url, source_config['name'], disaster_type)
                    if article_data:
                        # Kiểm tra ngày tháng
                        if date_from or date_to:
                            article_date = article_data.get('date')
                            if article_date:
                                try:
                                    article_date_obj = datetime.strptime(article_date, "%Y-%m-%d").date()
                                    if date_from and article_date_obj < date_from:
                                        continue
                                    if date_to and article_date_obj > date_to:
                                        continue
                                except ValueError:
                                    pass  # Bỏ qua nếu không parse được date

                        article_data['category'] = category
                        article_data['disaster_type'] = disaster_type
                        articles.append(article_data)
                        successful_articles += 1

                    time.sleep(0.5)  # Giãn cách để tránh bị block

                print(f"  ✓ {source_config['name']}: {successful_articles}/{len(urls)} bài thành công")
                return articles

            except Exception as e:
                print(f"  ✗ Lỗi với {source_config['name']}: {e}")
                return []

        # Sử dụng đa luồng để xử lý các nguồn song song
        with ThreadPoolExecutor(max_workers=3) as executor:  # Giới hạn 3 luồng để tránh quá tải
            futures = {executor.submit(process_source, source_key, source_config): source_key 
                      for source_key, source_config in NEWS_SOURCES.items()}
            
            for future in as_completed(futures):
                source_results.extend(future.result())
                time.sleep(1)  # Giãn cách giữa các nguồn

        self.results.extend(source_results)
        return source_results

    def scrape_all_disasters(self, max_articles_per_source=10):
        """Thu thập tất cả các loại thiên tai"""
        print("\n" + "="*80)
        print("BẮT ĐẦU THU THẬP DỮ LIỆU TỪ 10 NGUỒN BÁO")
        print("="*80)

        total_start = time.time()

        for category, disaster_type in DISASTER_TYPES:
            print(f"\n\n{'#'*80}")
            print(f"Đang thu thập: {disaster_type}")
            print(f"Thuộc nhóm: {category}")
            print(f"{'#'*80}")

            start_time = time.time()
            self.scrape_all_sources(disaster_type, category, max_articles_per_source)
            elapsed = time.time() - start_time

            print(f"\n⏱️  Hoàn thành '{disaster_type}' trong {elapsed:.1f} giây")
            print(f"📊 Tổng số bài hiện tại: {len(self.results)}")

            time.sleep(3)  # Giãn cách giữa các loại thiên tai

        total_elapsed = time.time() - total_start
        print(f"\n{'='*80}")
        print(f"🎯 HOÀN TẤT THU THẬP TẤT CẢ THIÊN TAI!")
        print(f"⏱️  Tổng thời gian: {total_elapsed:.1f} giây")
        print(f"📊 Tổng số bài viết: {len(self.results)}")
        print(f"{'='*80}")

        self.save_results()
        self.print_statistics()

        return self.results

    def debug_extraction(self, source_key, query, page=1):
        """Debug chi tiết quá trình trích xuất"""
        source = NEWS_SOURCES[source_key]
        formatted_query = quote(query)

        if '{page}' in source['search_url']:
            search_url = source['search_url'].format(query=formatted_query, page=page)
        else:
            search_url = source['search_url'].format(query=formatted_query)

        print(f"\n{'='*80}")
        print(f"DEBUG EXTRACTION: {source['name']}")
        print(f"URL: {search_url}")
        print(f"Query: {query}")
        print(f"Page: {page}")
        print(f"{'='*80}")

        try:
            verify_ssl = not source.get('needs_ssl_workaround', False)
            response = self.session.get(search_url, timeout=15, verify=verify_ssl)

            print(f"Status Code: {response.status_code}")
            print(f"Encoding: {response.encoding}")
            print(f"Content Length: {len(response.text)} chars")

            soup = BeautifulSoup(response.text, 'html.parser')

            # 1. Test selector chính
            print(f"\n1. Test selector chính: {source['article_selector']}")
            articles = soup.select(source['article_selector'])
            print(f"   Tìm thấy: {len(articles)} elements")

            if articles:
                for i, a in enumerate(articles[:5], 1):
                    href = a.get('href', 'No href')
                    text = a.get_text(strip=True)[:100]
                    print(f"   [{i}] Text: {text}")
                    print(f"       Href: {href[:100]}")

            # 2. Tìm tất cả thẻ H2, H3, H4
            print(f"\n2. Tìm tất cả thẻ tiêu đề:")
            for tag in ['h1', 'h2', 'h3', 'h4']:
                tags = soup.find_all(tag)
                print(f"   {tag.upper()}: {len(tags)} thẻ")
                for i, h in enumerate(tags[:2], 1):
                    link = h.find('a')
                    if link:
                        print(f"     [{i}] Text: {h.get_text(strip=True)[:80]}")
                        print(f"         Link: {link.get('href', '')[:80]}")

            # 3. Tìm tất cả links
            print(f"\n3. Tổng số links trên trang:")
            all_links = soup.find_all('a', href=True)
            print(f"   Tất cả links: {len(all_links)}")

            # 4. HTML snippet của container có thể chứa kết quả
            print(f"\n4. Tìm container kết quả tìm kiếm:")
            container_selectors = [
                '.search-results', '.results', '.list-news',
                '.news-list', '.article-list', '.story-list',
                '[class*="search"]', '[class*="result"]',
                'main', 'section', '.content'
            ]

            for selector in container_selectors:
                containers = soup.select(selector)
                if containers:
                    print(f"   Selector '{selector}': {len(containers)} containers")
                    for i, container in enumerate(containers[:1], 1):
                        print(f"   Container {i} HTML (500 chars):")
                        print(f"   {str(container)[:500]}...")
                        break

            # 5. In một phần HTML để kiểm tra
            print(f"\n5. Mẫu HTML trang (1000 ký tự đầu):")
            print(response.text[:1000])

        except Exception as e:
            print(f"Lỗi: {e}")
            import traceback
            traceback.print_exc()

    def debug_source(self, source_key, query):
        """Debug một nguồn cụ thể"""
        self.debug_extraction(source_key, query, 1)

    def save_results(self):
        """Lưu kết quả ra file"""
        if not self.results:
            print("Không có dữ liệu để lưu!")
            return None, None

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Lưu JSON
        json_filename = os.path.join(DATA_DIR, f"disaster_data_multisource_{timestamp}.json")
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        print(f"\n✓ Đã lưu JSON: {json_filename}")

        # Lưu CSV
        try:
            df = pd.DataFrame(self.results)
            csv_filename = os.path.join(DATA_DIR, f"disaster_data_multisource_{timestamp}.csv")
            df.to_csv(csv_filename, index=False, encoding='utf-8-sig')
            print(f"✓ Đã lưu CSV: {csv_filename}")
        except Exception as e:
            print(f"✗ Lỗi lưu CSV: {e}")
            csv_filename = None

        return json_filename, csv_filename

    def print_statistics(self):
        """In thống kê chi tiết"""
        if not self.results:
            print("Không có dữ liệu để thống kê!")
            return

        print("\n" + "="*80)
        print("📊 THỐNG KÊ THU THẬP CHI TIẾT")
        print("="*80)

        df = pd.DataFrame(self.results)

        print(f"\n📈 Tổng số bài viết: {len(df)}")

        print("\n--- Theo nguồn báo ---")
        source_stats = df.groupby('source').size().sort_values(ascending=False)
        for source, count in source_stats.items():
            percentage = (count / len(df)) * 100
            print(f"  📰 {source}: {count} bài ({percentage:.1f}%)")

        print("\n--- Theo loại thiên tai ---")
        disaster_stats = df.groupby('disaster_type').size().sort_values(ascending=False)
        for disaster, count in disaster_stats.items():
            print(f"  ⚠️  {disaster}: {count} bài")

        print("\n--- Theo nhóm thiên tai ---")
        category_stats = df.groupby('category').size().sort_values(ascending=False)
        for cat, count in category_stats.items():
            print(f"  📁 {cat}: {count} bài")

        print("\n--- Theo ngày tháng ---")
        if 'date' in df.columns:
            date_counts = df['date'].value_counts().head(10)
            for date, count in date_counts.items():
                print(f"  📅 {date}: {count} bài")

        print("\n--- Độ dài nội dung ---")
        if 'content_length' in df.columns:
            avg_length = df['content_length'].mean()
            max_length = df['content_length'].max()
            min_length = df['content_length'].min()
            print(f"  📝 Trung bình: {avg_length:.0f} ký tự")
            print(f"  📝 Ngắn nhất: {min_length} ký tự")
            print(f"  📝 Dài nhất: {max_length} ký tự")

        # Mẫu dữ liệu
        print("\n" + "="*80)
        print("📋 MẪU DỮ LIỆU (3 bài đầu tiên)")
        print("="*80)

        for i, item in enumerate(self.results[:3], 1):
            print(f"\n[{i}] Nguồn: {item['source']}")
            print(f"    Loại: {item['disaster_type']}")
            print(f"    Tiêu đề: {item['title'][:100]}...")
            print(f"    Ngày: {item['date']}")
            print(f"    URL: {item['url'][:80]}...")
            print(f"    Độ dài nội dung: {item.get('content_length', 'N/A')} ký tự")

    def quick_test(self):
        """Kiểm tra nhanh tất cả các nguồn"""
        print("\n" + "="*80)
        print("🔍 KIỂM TRA NHANH TẤT CẢ NGUỒN BÁO")
        print("="*80)

        test_query = "Bão"
        test_results = {}

        for source_key in NEWS_SOURCES.keys():
            print(f"\n--- Test {NEWS_SOURCES[source_key]['name']} ---")
            try:
                urls = self.extract_urls_from_source(source_key, test_query, max_pages=1)
                if urls:
                    test_results[source_key] = {
                        'status': 'OK',
                        'urls_found': len(urls),
                        'sample_url': list(urls)[0] if urls else None
                    }
                    print(f"✅ OK: Tìm thấy {len(urls)} URLs")
                    print(f"   Ví dụ: {list(urls)[0][:80]}...")
                else:
                    test_results[source_key] = {
                        'status': 'NO_URLS',
                        'urls_found': 0,
                        'sample_url': None
                    }
                    print(f"⚠️  CẢNH BÁO: Không tìm thấy URL nào")
            except Exception as e:
                test_results[source_key] = {
                    'status': 'ERROR',
                    'error': str(e)[:100],
                    'urls_found': 0
                }
                print(f"❌ LỖI: {str(e)[:100]}")

            time.sleep(1)

        # Tổng kết
        print("\n" + "="*80)
        print("📋 TỔNG KẾT KIỂM TRA")
        print("="*80)

        ok_count = sum(1 for r in test_results.values() if r['status'] == 'OK')
        warning_count = sum(1 for r in test_results.values() if r['status'] == 'NO_URLS')
        error_count = sum(1 for r in test_results.values() if r['status'] == 'ERROR')

        print(f"\n✅ Hoạt động tốt: {ok_count}/{len(NEWS_SOURCES)} nguồn")
        print(f"⚠️  Cảnh báo: {warning_count}/{len(NEWS_SOURCES)} nguồn")
        print(f"❌ Lỗi: {error_count}/{len(NEWS_SOURCES)} nguồn")

        if error_count > 0:
            print("\n🔧 Các nguồn cần debug:")
            for source_key, result in test_results.items():
                if result['status'] == 'ERROR':
                    print(f"   - {NEWS_SOURCES[source_key]['name']}: {result.get('error', 'Lỗi không xác định')}")

# Chạy chương trình
if __name__ == "__main__":
    scraper = NewsScraperMultiSource()

    # === MENU CHÍNH ===
    print("\n" + "="*80)
    print("🌪️  HỆ THỐNG THU THẬP DỮ LIỆU THIÊN TAI ĐA NGUỒN")
    print("="*80)

    while True:
        print("\n" + "-"*80)
        print("MENU CHÍNH:")
        print("1. Kiểm tra nhanh tất cả nguồn")
        print("2. Debug một nguồn cụ thể")
        print("3. Thu thập dữ liệu mẫu (1 loại thiên tai)")
        print("4. Thu thập toàn bộ dữ liệu thiên tai")
        print("5. Xem thống kê (nếu có dữ liệu)")
        print("6. Thoát")
        print("-"*80)

        choice = input("Chọn chức năng (1-6): ").strip()

        if choice == "1":
            # Kiểm tra nhanh
            scraper.quick_test()

        elif choice == "2":
            # Debug một nguồn
            print("\nCác nguồn có sẵn:")
            for i, (key, config) in enumerate(NEWS_SOURCES.items(), 1):
                print(f"{i}. {config['name']} ({key})")

            try:
                source_num = int(input("Chọn số nguồn (1-10): "))
                source_keys = list(NEWS_SOURCES.keys())
                if 1 <= source_num <= len(source_keys):
                    source_key = source_keys[source_num - 1]
                    query = input("Nhập từ khóa tìm kiếm (mặc định: Bão): ") or "Bão"
                    scraper.debug_source(source_key, query)
                else:
                    print("Số không hợp lệ!")
            except ValueError:
                print("Vui lòng nhập số!")

        elif choice == "3":
            # Thu thập dữ liệu mẫu
            print("\nChọn loại thiên tai mẫu:")
            print("1. Bão, áp thấp nhiệt đới")
            print("2. Lũ, lũ quét")
            print("3. Hạn hán")
            print("4. Sạt lở đất")
            print("5. Cháy rừng")

            disaster_choice = input("Chọn số (1-5, mặc định: 1): ").strip() or "1"

            disasters_map = {
                "1": "Bão, áp thấp nhiệt đới",
                "2": "Lũ, lũ quét",
                "3": "Hạn hán",
                "4": "Sạt lở đất, trượt đất, sụt lún",
                "5": "Cháy rừng"
            }

            if disaster_choice in disasters_map:
                disaster_type = disasters_map[disaster_choice]
                category = "Thiên tai khí tượng – thủy văn" if disaster_choice in ["1", "2", "3"] else \
                          "Thiên tai địa chất" if disaster_choice == "4" else \
                          "Thiên tai môi trường – con người gây ra"

                try:
                    max_articles = int(input("Số bài tối đa mỗi nguồn (mặc định: 10): ") or "10")
                except ValueError:
                    max_articles = 10

                date_from = input("Ngày bắt đầu (YYYY-MM-DD, mặc định: không giới hạn): ").strip() or None
                date_to = input("Ngày kết thúc (YYYY-MM-DD, mặc định: không giới hạn): ").strip() or None

                if date_from:
                    try:
                        date_from = datetime.strptime(date_from, "%Y-%m-%d").date()
                    except ValueError:
                        print("Định dạng ngày không hợp lệ, bỏ qua.")
                        date_from = None
                if date_to:
                    try:
                        date_to = datetime.strptime(date_to, "%Y-%m-%d").date()
                    except ValueError:
                        print("Định dạng ngày không hợp lệ, bỏ qua.")
                        date_to = None

                print(f"\nĐang thu thập dữ liệu cho: {disaster_type}...")
                results = scraper.scrape_all_sources(
                    disaster_type=disaster_type,
                    category=category,
                    max_articles_per_source=max_articles,
                    date_from=date_from,
                    date_to=date_to
                )
                scraper.save_results()
                scraper.print_statistics()
            else:
                print("Lựa chọn không hợp lệ!")

        elif choice == "4":
            # Thu thập toàn bộ
            confirm = input("CẢNH BÁO: Thao tác này có thể mất nhiều thời gian. Tiếp tục? (y/n): ")
            if confirm.lower() == 'y':
                try:
                    max_articles = int(input("Số bài tối đa mỗi nguồn cho mỗi thiên tai (mặc định: 5): ") or "5")
                except ValueError:
                    max_articles = 5

                scraper.scrape_all_disasters(max_articles_per_source=max_articles)

        elif choice == "5":
            # Xem thống kê
            if scraper.results:
                scraper.print_statistics()
            else:
                print("Chưa có dữ liệu. Vui lòng thu thập trước.")

        elif choice == "6":
            print("Thoát chương trình.")
            break

        else:
            print("Lựa chọn không hợp lệ!")

        # Hỏi tiếp tục hay không
        if choice in ["3", "4", "5"]:
            continue_option = input("\nTiếp tục với menu chính? (y/n): ")
            if continue_option.lower() != 'y':
                print("Thoát chương trình.")
                break