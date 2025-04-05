import requests
from bs4 import BeautifulSoup
import os
import time
from urllib.parse import urljoin, parse_qs, urlparse
import random
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('aao_scraper.log'),
        logging.StreamHandler()
    ]
)

class AAOScraper:
    def __init__(self):
        self.base_url = "https://www.uscis.gov/administrative-appeals/aao-decisions/aao-non-precedent-decisions"
        self.download_dir = "aao_decisions"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        })
        
        if not os.path.exists(self.download_dir):
            os.makedirs(self.download_dir)

    def get_page(self, page_num=0):
        params = {
            'uri_1': '22',
            'm': 'All',
            'y': 'All',
            'items_per_page': '10',
            'page': page_num
        }
        
        time.sleep(random.uniform(3, 7))
        
        try:
            response = self.session.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            
            if 'text/html' not in response.headers.get('Content-Type', ''):
                logging.error(f"Unexpected content type: {response.headers.get('Content-Type')}")
                return None
                
            return response.text
        except requests.RequestException as e:
            logging.error(f"Error fetching page {page_num}: {str(e)}")
            return None

    def get_pdf_links(self, html_content):
        if not html_content:
            return []
        
        soup = BeautifulSoup(html_content, 'html.parser')
        pdf_links = []
        
        # Look for links within the main content area
        content_area = soup.find('div', class_='usa-layout-docs__main')
        if not content_area:
            content_area = soup  # fallback to entire page
        
        for link in content_area.find_all('a', href=True):
            href = link['href']
            if href.lower().endswith('.pdf'):
                full_url = urljoin(self.base_url, href)
                pdf_links.append(full_url)
                logging.info(f"Found PDF link: {full_url}")
        
        return pdf_links

    def download_pdf(self, pdf_url):
        try:
            filename = os.path.basename(urlparse(pdf_url).path)
            if not filename.endswith('.pdf'):
                filename += '.pdf'
            
            filepath = os.path.join(self.download_dir, filename)
            
            if os.path.exists(filepath):
                logging.info(f"File already exists: {filename}")
                return True
            
            time.sleep(random.uniform(2, 5))
            
            response = self.session.get(pdf_url, stream=True, timeout=30)
            response.raise_for_status()
            
            if 'application/pdf' not in response.headers.get('Content-Type', '').lower():
                logging.error(f"Unexpected content type for PDF: {response.headers.get('Content-Type')}")
                return False
            
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            logging.info(f"Successfully downloaded: {filename}")
            return True
            
        except requests.RequestException as e:
            logging.error(f"Error downloading {pdf_url}: {str(e)}")
            return False

    def scrape(self, max_pages=100):  # Added max_pages parameter for safety
        logging.info("Starting AAO decisions scraper")
        page_num = 0
        no_pdfs_count = 0
        
        while page_num < max_pages:
            logging.info(f"Processing page {page_num}")
            
            html_content = self.get_page(page_num)
            if not html_content:
                logging.error(f"Failed to fetch page {page_num}, stopping")
                break
            
            pdf_links = self.get_pdf_links(html_content)
            
            if not pdf_links:
                no_pdfs_count += 1
                if no_pdfs_count >= 3:  # If we find no PDFs for 3 consecutive pages, assume we're done
                    logging.info("No PDFs found for 3 consecutive pages, assuming end of content")
                    break
            else:
                no_pdfs_count = 0  # Reset counter when we find PDFs
                
            for pdf_url in pdf_links:
                self.download_pdf(pdf_url)
            
            # Check if there might be a next page by looking for pagination
            soup = BeautifulSoup(html_content, 'html.parser')
            next_link = soup.find('a', {'rel': 'next'}) or soup.find('a', text=lambda t: t and 'next' in t.lower())
            if not next_link:
                logging.info("No next page link found, stopping")
                break
                
            page_num += 1

if __name__ == "__main__":
    scraper = AAOScraper()
    scraper.scrape()