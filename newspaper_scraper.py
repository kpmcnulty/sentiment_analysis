import sys
import time
import os
import json
import re
import logging
from datetime import datetime, timedelta
import requests
import urllib.parse
from newspaper import Article
from GoogleNews import GoogleNews
from bs4 import BeautifulSoup
import shutil

# Setup logging
def setup_logging(output_dir="logs"):
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, f"scraper_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    # Configure logging - keep it simple with just INFO level
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return log_file

def load_config(config_file="scraper_config.json"):
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        logging.info(f"Loaded configuration from {config_file}")
        return config
    except Exception as e:
        logging.error(f"Error loading configuration: {str(e)}")
        return {}

def clean_url(url):
    # Google redirect URL
    if "&ved=" in url:
        url = url.split("&ved=")[0]

    try:
        parsed = urllib.parse.urlparse(url)
        # Reconstruct URL without query parameters
        if "google" in parsed.netloc:
            params = urllib.parse.parse_qs(parsed.query)
            if 'url' in params:
                return params['url'][0] 
            if 'q' in params:
                return params['q'][0]  
        return url
    except:
        return url

def search_news(query, config):
    # Get config values
    max_results = config.get('max_results', 100)
    max_pages = config.get('google_news', {}).get('max_pages', 5)
    blocked_domains = config.get('blocked_domains', [])
    
    # Get progress callback if available
    progress_callback = getattr(search_news, 'progress_callback', None)
    
    if progress_callback:
        if progress_callback(f"Searching for articles about '{query}' from past {config.get('period', '1m')}...", 5):
            return []  # Stop requested
    else:
        print(f"Searching for articles about '{query}' from past {config.get('period', '1m')}...")
    
    # Initial delay before starting
    if progress_callback:
        if progress_callback(f"Waiting {initial_delay} seconds before starting search...", 8):
            return []  # Stop requested
    else:
        print(f"Waiting {initial_delay} seconds before starting search...")
    time.sleep(initial_delay)
    
    # Initialize GoogleNews
    googlenews = GoogleNews()
    googlenews.set_lang('en')
    googlenews.set_period(config.get('period', '1m'))
    
    # Retry mechanism for initial search with longer delays
    max_retries = 3  # Reduced retries but longer delays
    retry_delay = 60  # Start with 60 seconds
    search_success = False
    
    for retry in range(max_retries):
        try:
            googlenews.search(query)
            search_success = True
            break
        except Exception as e:
            if "429" in str(e) or "Too Many Requests" in str(e):
                if retry < max_retries - 1:
                    retry_delay = retry_delay * 2  # Exponential backoff: 60, 120, 240 seconds
                    if progress_callback:
                        if progress_callback(f"Rate limited! Waiting {retry_delay} seconds before retry {retry+1}/{max_retries}...", 10):
                            return []  # Stop requested
                    else:
                        print(f"Rate limited! Waiting {retry_delay} seconds before retry {retry+1}/{max_retries}...")
                    time.sleep(retry_delay)
                else:
                    if progress_callback:
                        progress_callback(f"Failed to search after {max_retries} retries due to rate limiting.", 15)
                    else:
                        print(f"Failed to search after {max_retries} retries due to rate limiting.")
            else:
                if progress_callback:
                    progress_callback(f"Search error: {str(e)}", 15)
                else:
                    print(f"Search error: {str(e)}")
                break
    
    if not search_success:
        if progress_callback:
            progress_callback("Could not complete search due to persistent rate limiting.", 20)
        else:
            print("Could not complete search due to persistent rate limiting.")
        return []
    
    # Get results
    results = []
    blocked_count = 0
    page = 1
    
    while len(results) < max_results and page <= max_pages:
        print(f"Fetching page {page} of results...")
        # Add delay before getting results from each page (10-30 seconds)
        page_delay = config.get("page_delay", 20)
        if page > 1:
            print(f"Waiting {page_delay} seconds before fetching page {page}...")
            time.sleep(page_delay)
        
        try:
            search_results = googlenews.result()
            if not search_results:
                print("No more results found.")
                break
            
            # Track how many new articles are added this page
            new_articles_this_page = 0
            
            for result in search_results:
                if 'link' in result and result['link']:
                    # Clean the URL
                    clean_link = clean_url(result['link'])
                    
                    # Skip if URL is from a blocked domain
                    if clean_link:
                        parsed_url = urllib.parse.urlparse(clean_link)
                        domain = parsed_url.netloc
                        if any(blocked_domain in domain for blocked_domain in blocked_domains):
                            print(f"Skipping blocked domain: {domain}")
                            blocked_count += 1
                            continue
                        
                        # Check if this URL already exists in results
                        if not any(r['link'] == clean_link for r in results):
                            results.append({
                                'title': result.get('title', ''),
                                'media': result.get('media', ''),
                                'date': result.get('date', ''),
                                'desc': result.get('desc', ''),
                                'link': clean_link
                            })
                            print(f"Found article: {result.get('title', '')[:50]}...")
                            new_articles_this_page += 1
                            if len(results) >= max_results:
                                break
            
            # Quit once we start getting only blocked domains or duplicates
            if new_articles_this_page == 0:
                print("No new articles found on this page. Stopping further page fetches.")
                break
            
            if len(results) < max_results:
                page += 1
                # Add retry mechanism for page fetching
                page_fetch_success = False
                for retry in range(max_retries):
                    try:
                        googlenews.getpage(page)
                        page_fetch_success = True
                        break
                    except Exception as e:
                        if "429" in str(e) or "Too Many Requests" in str(e):
                            if retry < max_retries - 1:
                                retry_delay = retry_delay * 2  # Exponential backoff
                                print(f"Rate limited! Waiting {retry_delay} seconds before retry {retry+1}/{max_retries}...")
                                time.sleep(retry_delay)
                            else:
                                print(f"Failed to fetch page {page} after {max_retries} retries.")
                        else:
                            print(f"Error fetching page {page}: {str(e)}")
                            break
                
                if not page_fetch_success:
                    print(f"Stopping at page {page-1} due to persistent rate limiting.")
                    break
                
        except Exception as e:
            print(f"Error getting results from page {page}: {str(e)}")
            break
    
    print(f"Found {len(results)} articles for '{query}' (blocked {blocked_count} domains)")
    return results[:max_results]

def extract_with_bs4(url, user_agent, config):
    headers = {
        'User-Agent': user_agent,
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Connection': 'keep-alive'
    }
    
    max_retries = 5  
    base_delay = 10
    max_delay = 120  
    
    for attempt in range(max_retries):
        try:
            timeout = config.get("google_news", {}).get("request_timeout", 10)
            response = requests.get(url, headers=headers, timeout=timeout)
            
            # If we get a 429, wait and retry with exponential backoff
            if response.status_code == 429:
                wait_time = min(base_delay * (2 ** attempt), max_delay)  # Cap the maximum wait time
                logging.warning(f"Rate limited (429) for {url}. Waiting {wait_time} seconds before retry {attempt + 1}/{max_retries}")
                time.sleep(wait_time)
                continue
                
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Add a small delay after successful request to be nice to servers
            time.sleep(2)
            
            # Extract title
            title = ""
            if soup.title:
                title = soup.title.text.strip()
            
            # Extract paragraphs
            paragraphs = soup.find_all('p')
            
            # Remove very short paragraphs (likely navigation, etc.)
            min_paragraph_length = config.get("extraction", {}).get("min_paragraph_length", 40)
            content_paragraphs = [p.text.strip() for p in paragraphs if len(p.text.strip()) > min_paragraph_length]
            
            # Combine paragraphs
            text = '\n\n'.join(content_paragraphs)
            
            # Get meta description
            meta_desc = ""
            meta_desc_tag = soup.find('meta', attrs={'name': 'description'}) or soup.find('meta', attrs={'property': 'og:description'})
            if meta_desc_tag and 'content' in meta_desc_tag.attrs:
                meta_desc = meta_desc_tag['content']
            
            # Get keywords
            keywords = []
            keywords_tag = soup.find('meta', attrs={'name': 'keywords'})
            if keywords_tag and 'content' in keywords_tag.attrs:
                keywords_content = keywords_tag['content']
                if keywords_content:
                    keywords = [k.strip() for k in keywords_content.split(',')]
            
            # Create basic summary from first paragraphs
            summary = '\n\n'.join(content_paragraphs[:3]) if content_paragraphs else ""
            
            logging.info(f"BS4 extracted: Title: '{title[:40]}...', Text: {len(text)} chars")
            return {
                'title': title,
                'text': text,
                'summary': summary,
                'meta_description': meta_desc,
                'keywords': keywords
            }
            
        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:  # Last attempt
                logging.error(f"Failed to fetch {url} after {max_retries} attempts: {str(e)}")
                return None
            wait_time = base_delay * (2 ** attempt)
            logging.warning(f"Request failed for {url}. Waiting {wait_time} seconds before retry {attempt + 1}/{max_retries}")
            time.sleep(wait_time)
    
    return None

def extract_article_content(url, config):
    try:
        # Get user agent from config
        user_agent = config.get("user_agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36")
        
        # Add delay before making request (configurable)
        delay_before_download = config.get("delay_between_downloads", 1)
        time.sleep(delay_before_download)
        
        # Try with newspaper3k first
        article = Article(url, browser_user_agent=user_agent)
        article.download()
        
        article.parse()
        
        # Log what newspaper3k found
        text_length = len(article.text or '')
        title_excerpt = article.title[:40] + "..." if article.title and len(article.title) > 40 else article.title
        logging.info(f"Newspaper3k extracted: Title: '{title_excerpt}', Text: {text_length} chars")
        
        # Check if we have enough text from newspaper
        min_text_length = config.get("extraction", {}).get("min_text_length", 150)
        if article.text and len(article.text.strip()) > min_text_length:
            try:
                article.nlp()
            except Exception as nlp_error:
                logging.warning(f"NLP processing failed for {url}")
                # Set defaults if NLP won't run
                article.summary = article.meta_description or ""
                article.keywords = []
            
            return {
                'title': article.title,
                'text': article.text,
                'summary': article.summary or "",
                'keywords': article.keywords or [],
                'authors': article.authors,
                'published_date': article.publish_date.strftime('%Y-%m-%d %H:%M:%S') if article.publish_date else None,
                'source': url,
                'canonical_link': article.canonical_link,
                'meta_lang': article.meta_lang,
                'meta_keywords': article.meta_keywords,
                'meta_description': article.meta_description,
                'meta_data': article.meta_data,
                'meta_favicon': article.meta_favicon
            }
        
        # If newspaper3k failed, try direct BS4 extraction
        logging.info(f"Falling back to BS4 extraction for {url}")
        bs4_article = extract_with_bs4(url, user_agent, config)
        
        min_bs4_text_length = config.get("extraction", {}).get("min_bs4_text_length", 100)
        if bs4_article and bs4_article['text'] and len(bs4_article['text'].strip()) > min_bs4_text_length:
            # Combine with any metadata we got from newspaper
            result = {
                'title': bs4_article['title'] or article.title or "",
                'text': bs4_article['text'],
                'summary': bs4_article['summary'] or article.summary or "",
                'keywords': bs4_article['keywords'] or article.keywords or [],
                'authors': article.authors,
                'published_date': article.publish_date.strftime('%Y-%m-%d %H:%M:%S') if article.publish_date else None,
                'source': url,
                'canonical_link': article.canonical_link,
                'meta_lang': article.meta_lang,
                'meta_keywords': article.meta_keywords or bs4_article['keywords'],
                'meta_description': article.meta_description or bs4_article['meta_description'],
                'meta_data': article.meta_data,
                'meta_favicon': article.meta_favicon,
                'extraction_method': 'bs4_fallback'
            }
            return result
        
        # Still couldn't extract enough text, check if we at least have metadata
        if article.title or article.meta_description:
            logging.info(f"Using minimal content (metadata only) from {url}")
            minimal_content = {
                'title': article.title or bs4_article['title'] if bs4_article else "",
                'text': article.meta_description or "",  # Use meta description as text
                'summary': article.meta_description or "",
                'keywords': article.keywords or [],
                'authors': article.authors,
                'published_date': article.publish_date.strftime('%Y-%m-%d %H:%M:%S') if article.publish_date else None,
                'source': url,
                'canonical_link': article.canonical_link,
                'meta_lang': article.meta_lang,
                'meta_keywords': article.meta_keywords,
                'meta_description': article.meta_description,
                'meta_data': article.meta_data,
                'meta_favicon': article.meta_favicon,
                'extraction_status': 'minimal'
            }
            return minimal_content
        
        logging.warning(f"Failed to extract any useful content from {url}")
        return None
        
    except Exception as e:
        logging.error(f"Error extracting article from {url}: {str(e)}")
        return None

def is_recent(published_date_str, max_age_days=30):
    if not published_date_str:
        return False
    try:
        published_date = datetime.strptime(published_date_str, "%Y-%m-%d %H:%M:%S")
        return published_date >= datetime.now() - timedelta(days=max_age_days)
    except Exception:
        return False

def parse_period_to_days(period):
    """Convert period string to number of days and GoogleNews format"""
    if not period:
        return 30, "1m"  # default
    
    match = re.match(r"(\d+)([dwmy])", period.strip().lower())
    if not match:
        return 30, "1m"
    
    num, unit = int(match.group(1)), match.group(2)
    
    # Convert to GoogleNews format (e.g., "1d", "7d", "1m", "1y")
    gnews_format = period.strip().lower()
    
    # Calculate days for article filtering
    if unit == 'd':
        return num, gnews_format
    elif unit == 'w':
        return num * 7, f"{num}d"  # Convert weeks to days for GoogleNews
    elif unit == 'm':
        return num * 30, gnews_format
    elif unit == 'y':
        return num * 365, gnews_format
    return 30, "1m"

def get_news_content(query, config):
    # Get config values with proper defaults
    max_results = config.get("max_results", 100)
    max_pages = config.get("google_news", {}).get("max_pages", 10)
    period = config.get("period", "1m")
    max_age_days, gnews_period = parse_period_to_days(period)
    blocked_domains = config.get("blocked_domains", [])
    
    # Get delay settings with proper defaults
    initial_delay = config.get("initial_delay", 15)
    page_delay = config.get("page_delay", 20)
    article_delay = config.get("delay_between_downloads", 3)
    
    # Get max_consecutive_no_relevant from config, default to 3
    max_consecutive_no_relevant = config.get("max_consecutive_no_relevant", 3)
    
    # Get progress callback if available
    progress_callback = getattr(get_news_content, 'progress_callback', None)
    
    if progress_callback:
        if progress_callback(f"\nSearching for articles about '{query}' from past {period}...", 25):
            return [], [], 0  # Stop requested
    else:
        print(f"\nSearching for articles about '{query}' from past {period}...")
    
    # Initialize GoogleNews with proper period format
    lang = config.get("language", "en")
    googlenews = GoogleNews(lang=lang, period=gnews_period)
    
    # Add initial delay before first search
    if progress_callback:
        if progress_callback(f"Waiting {initial_delay} seconds before starting search...", 30):
            return [], [], 0  # Stop requested
    else:
        print(f"Waiting {initial_delay} seconds before starting search...")
    time.sleep(initial_delay)
    
    # Add retries for the initial search
    max_retries = 3
    retry_delay = initial_delay  # Start with initial delay as base
    search_success = False
    
    for retry in range(max_retries):
        try:
            googlenews.search(query)
            search_success = True
            break
        except Exception as e:
            if "429" in str(e) or "Too Many Requests" in str(e):
                if retry < max_retries - 1:
                    retry_delay = retry_delay * 2  # Exponential backoff
                    if progress_callback:
                        if progress_callback(f"Rate limited! Waiting {retry_delay} seconds before retry {retry+1}/{max_retries}...", 35):
                            return [], [], 0  # Stop requested
                    else:
                        print(f"Rate limited! Waiting {retry_delay} seconds before retry {retry+1}/{max_retries}...")
                    time.sleep(retry_delay)
                else:
                    if progress_callback:
                        progress_callback(f"Failed to search after {max_retries} retries due to rate limiting.", 40)
                    else:
                        print(f"Failed to search after {max_retries} retries due to rate limiting.")
            else:
                if progress_callback:
                    progress_callback(f"Search error: {str(e)}", 40)
                else:
                    print(f"Search error: {str(e)}")
                break
    
    if not search_success:
        if progress_callback:
            progress_callback("Could not complete search due to persistent rate limiting.", 50)
        else:
            print("Could not complete search due to persistent rate limiting.")
        return [], [], 0
    
    relevant_articles = []
    failed_urls = []
    blocked_count = 0
    page = 1
    seen_urls = set()
    total_attempted = 0  # Track total attempts for progress
    consecutive_no_relevant = 0
    
    while len(relevant_articles) < max_results:
        if progress_callback:
            if progress_callback(f"Fetching page {page} of results... (Found {len(relevant_articles)} articles)", 50 + (page * 5)):
                break  # Stop requested
        else:
            print(f"Fetching page {page} of results... (Found {len(relevant_articles)} articles)")
        
        if page > 1:
            # Add delay between page fetches
            if progress_callback:
                if progress_callback(f"Waiting {page_delay} seconds before fetching page {page}...", 50 + (page * 5)):
                    break  # Stop requested
            else:
                print(f"Waiting {page_delay} seconds before fetching page {page}...")
            time.sleep(page_delay)
            
            # Add retry mechanism for page fetching
            page_fetch_success = False
            retry_delay = page_delay
            for retry in range(max_retries):
                try:
                    googlenews.getpage(page)
                    page_fetch_success = True
                    break
                except Exception as e:
                    if "429" in str(e) or "Too Many Requests" in str(e):
                        if retry < max_retries - 1:
                            retry_delay = retry_delay * 2  # Exponential backoff
                            if progress_callback:
                                if progress_callback(f"Rate limited! Waiting {retry_delay} seconds before retry {retry+1}/{max_retries}...", 50 + (page * 5)):
                                    return relevant_articles, failed_urls, blocked_count  # Stop requested
                            else:
                                print(f"Rate limited! Waiting {retry_delay} seconds before retry {retry+1}/{max_retries}...")
                            time.sleep(retry_delay)
                        else:
                            if progress_callback:
                                progress_callback(f"Failed to fetch page {page} after {max_retries} retries.", 50 + (page * 5))
                            else:
                                print(f"Failed to fetch page {page} after {max_retries} retries.")
                    else:
                        if progress_callback:
                            progress_callback(f"Error fetching page {page}: {str(e)}", 50 + (page * 5))
                        else:
                            print(f"Error fetching page {page}: {str(e)}")
                        break
            
            if not page_fetch_success:
                if progress_callback:
                    progress_callback(f"Stopping at page {page-1} due to persistent rate limiting.", 50 + (page * 5))
                else:
                    print(f"Stopping at page {page-1} due to persistent rate limiting.")
                break
        
        # Track relevant articles before processing this page
        articles_before = len(relevant_articles)
        
        # Process search results...
        try:
            search_results = googlenews.result()
            if not search_results:
                if progress_callback:
                    progress_callback("No more results found.", 50 + (page * 5))
                else:
                    print("No more results found.")
                break
            
            for i, result in enumerate(search_results, 1):
                raw_url = result.get('link')
                if not raw_url:
                    continue
                
                url = clean_url(raw_url)
                if url in seen_urls:
                    continue
                seen_urls.add(url)
                total_attempted += 1
                
                parsed_url = urllib.parse.urlparse(url)
                domain = parsed_url.netloc
                
                if any(blocked_domain in domain for blocked_domain in blocked_domains):
                    if progress_callback:
                        progress_callback(f"Skipping blocked domain: {domain} (Found {len(relevant_articles)}/{max_results} articles)", 50 + (page * 5))
                    else:
                        print(f"Skipping blocked domain: {domain} (Found {len(relevant_articles)}/{max_results} articles)")
                    blocked_count += 1
                    failed_urls.append({
                        'url': url,
                        'title': result.get('title', ''),
                        'reason': 'Domain is blocked',
                        'search_term': query
                    })
                    continue
                
                # Add delay between article extractions
                if progress_callback:
                    if progress_callback(f"Waiting {article_delay} seconds before extracting article... (Found {len(relevant_articles)}/{max_results} articles)", 50 + (page * 5)):
                        return relevant_articles, failed_urls, blocked_count  # Stop requested
                else:
                    print(f"Waiting {article_delay} seconds before extracting article... (Found {len(relevant_articles)}/{max_results} articles)")
                time.sleep(article_delay)
                
                article_data = extract_article_content(url, config)
                if article_data:
                    article_data['title'] = article_data['title'] or result.get('title', '')
                    article_data['media'] = result.get('media', '')
                    article_data['date'] = result.get('date', '')
                    article_data['search_term'] = query
                    published_date = article_data.get('published_date')
                    # Allow if missing date, or if date is recent
                    if (not published_date) or is_recent(published_date, max_age_days=max_age_days):
                        relevant_articles.append(article_data)
                        if progress_callback:
                            progress_callback(f"Relevant article: {article_data['title'][:50]}... ({published_date}) (Found {len(relevant_articles)}/{max_results} articles)", 50 + (page * 5))
                        else:
                            print(f"Relevant article: {article_data['title'][:50]}... ({published_date}) (Found {len(relevant_articles)}/{max_results} articles)")
                        if len(relevant_articles) >= max_results:
                            break
                    else:
                        if progress_callback:
                            progress_callback(f"Article too old: {article_data['title'][:50]}... ({published_date}) (Found {len(relevant_articles)}/{max_results} articles)", 50 + (page * 5))
                        else:
                            print(f"Article too old: {article_data['title'][:50]}... ({published_date}) (Found {len(relevant_articles)}/{max_results} articles)")
                else:
                    if progress_callback:
                        progress_callback(f"Failed to extract content (Found {len(relevant_articles)}/{max_results} articles)", 50 + (page * 5))
                    else:
                        print(f"Failed to extract content (Found {len(relevant_articles)}/{max_results} articles)")
                    failed_urls.append({
                        'url': url,
                        'title': result.get('title', ''),
                        'reason': 'Failed to extract content',
                        'search_term': query
                    })
            
            page += 1
            
            # Check if any new relevant articles were added this page
            articles_after = len(relevant_articles)
            if articles_after == articles_before:
                consecutive_no_relevant += 1
            else:
                consecutive_no_relevant = 0
            if consecutive_no_relevant >= max_consecutive_no_relevant:
                print(f"Stopping: {consecutive_no_relevant} pages in a row with no relevant articles extracted.")
                break
        
        except Exception as e:
            if progress_callback:
                progress_callback(f"Error processing page {page}: {str(e)}", 50 + (page * 5))
            else:
                print(f"Error processing page {page}: {str(e)}")
            break
    
    if progress_callback:
        progress_callback(f"\nExtraction complete: {len(relevant_articles)} relevant, {len(failed_urls)} failed (after {total_attempted} attempts)", 90)
    else:
        print(f"\nExtraction complete: {len(relevant_articles)} relevant, {len(failed_urls)} failed (after {total_attempted} attempts)")
    
    return relevant_articles, failed_urls, blocked_count

def sanitize_filename(title):
    # Replace invalid filename characters with underscore
    sanitized = re.sub(r'[\\/*?:"<>|]', "_", title)
    # Limit length
    return sanitized[:100]

def save_articles(articles, output_dir="articles"):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    saved_files = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"\nSaving {len(articles)} articles to {output_dir}/...")
    for i, article in enumerate(articles, 1):
        # Create base filename from title
        if article['title']:
            base_name = sanitize_filename(article['title'])
        else:
            base_name = f"article_{i}"
        
        # Create filenames
        text_filename = f"{base_name}_{timestamp}.txt"
        json_filename = f"{base_name}_{timestamp}.json"
        
        text_path = os.path.join(output_dir, text_filename)
        json_path = os.path.join(output_dir, json_filename)
        
        print(f"Saving article {i}/{len(articles)}: {text_filename}")
        
        # Save text content
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write(f"Title: {article['title']}\n\n")
            f.write(f"Source: {article['source']}\n")
            if article.get('canonical_link'):
                f.write(f"Canonical URL: {article['canonical_link']}\n")
            f.write(f"Media: {article.get('media', '')}\n")
            f.write(f"Date: {article.get('date', '')}\n")
            f.write(f"Search Term: {article.get('search_term', 'Unknown')}\n\n")
            
            # Author information
            f.write(f"Authors: {', '.join(article.get('authors', []))}\n")
            f.write(f"Published: {article.get('published_date', 'Unknown')}\n\n")
            
            # Language and metadata
            f.write(f"Language: {article.get('meta_lang', 'Unknown')}\n")
            if article.get('meta_description'):
                f.write(f"Description: {article['meta_description']}\n")
            
            # Keywords
            f.write("KEYWORDS (NLP):\n")
            f.write(f"{', '.join(article.get('keywords', []))}\n\n")
            
            if article.get('meta_keywords'):
                f.write("KEYWORDS (Meta):\n")
                meta_keywords = article['meta_keywords']
                if isinstance(meta_keywords, list):
                    f.write(f"{', '.join(meta_keywords)}\n\n")
                else:
                    f.write(f"{meta_keywords}\n\n")
            
            # Summary and content
            f.write("SUMMARY:\n")
            f.write(f"{article['summary']}\n\n")
            
            f.write("FULL TEXT:\n")
            f.write(article['text'])
        
        # Save metadata as JSON
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(article, f, indent=2, ensure_ascii=False)
        
        saved_files.append(text_path)
    
    print(f"Successfully saved {len(saved_files)} articles")
    return saved_files

def main():
    # Load configuration
    config_path = "scraper_config.json"
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    config = load_config(config_path)
    
    # Get progress callback if available
    progress_callback = getattr(sys.modules[__name__], 'progress_callback', None)
    
    # Update default delays if not present
    if "initial_delay" not in config:
        config["initial_delay"] = 15  # Seconds to wait before first search
    if "page_delay" not in config:
        config["page_delay"] = 20     # Seconds to wait between page fetches
    if "delay_between_downloads" not in config:
        config["delay_between_downloads"] = 3  # Seconds to wait between article downloads
    
    # Setup logging
    log_file = setup_logging()
    
    # Get search terms and blocked domains
    search_terms = config.get("search_terms")
    blocked_domains = config.get("blocked_domains", [])
    output_dir = config.get("output_directory", "articles")
    results_dir = os.path.join(output_dir, "results")
    archive_dir = os.path.join(output_dir, "archive")
    
    # Archive old results if present
    if os.path.exists(results_dir) and os.listdir(results_dir):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_subdir = os.path.join(archive_dir, ts)
        os.makedirs(archive_subdir, exist_ok=True)
        for fname in os.listdir(results_dir):
            shutil.move(os.path.join(results_dir, fname), os.path.join(archive_subdir, fname))
        if progress_callback:
            progress_callback(f"Archived previous results to {archive_subdir}", 30)
        else:
            print(f"Archived previous results to {archive_subdir}")
    
    os.makedirs(results_dir, exist_ok=True)
    
    # Log blocked domains
    if blocked_domains:
        if progress_callback:
            progress_callback(f"Blocking content from domains: {', '.join(blocked_domains)}", 35)
        else:
            print(f"Blocking content from domains: {', '.join(blocked_domains)}")
    
    total_articles = 0
    total_failed = 0
    total_blocked = 0
    
    # Share the progress callback with other functions
    search_news.progress_callback = progress_callback
    get_news_content.progress_callback = progress_callback
    
    for i, query in enumerate(search_terms):
        # Calculate progress percentage for this query
        base_progress = 35 + (i * 60 / len(search_terms))
        if progress_callback:
            if progress_callback(f"\nProcessing search term: '{query}'", base_progress):
                break  # Stop requested
        else:
            print(f"\nProcessing search term: '{query}'")
        
        # Get news content
        articles, failed_urls, blocked_count = get_news_content(query, config)
        if not articles:
            if progress_callback:
                progress_callback(f"No articles found for query: {query}", base_progress + 10)
            else:
                print(f"No articles found for query: {query}")
            continue
        
        # Save articles in results_dir
        saved_files = save_articles(articles, results_dir)
        
        # Save failed URLs to JSON in results_dir/failed
        if failed_urls:
            failed_dir = os.path.join(results_dir, "failed")
            os.makedirs(failed_dir, exist_ok=True)
            failed_file = os.path.join(failed_dir, f"failed_urls_{query.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            with open(failed_file, 'w', encoding='utf-8') as f:
                json.dump(failed_urls, f, indent=2)
            if progress_callback:
                progress_callback(f"Saved {len(failed_urls)} failed URLs to {failed_file}", base_progress + 20)
            else:
                print(f"Saved {len(failed_urls)} failed URLs to {failed_file}")
        
        total_articles += len(saved_files)
        total_failed += len(failed_urls) - blocked_count
        total_blocked += blocked_count
    
    # Summary
    summary = f"\nFinal Summary:\n"
    summary += f"Total articles saved: {total_articles}\n"
    summary += f"Total failed extractions: {total_failed}\n"
    summary += f"Total blocked domains: {total_blocked}\n"
    summary += f"Articles saved to: {os.path.abspath(results_dir)}"
    
    if progress_callback:
        progress_callback(summary, 95)
    else:
        print(summary)
    
    return {
        'total_articles': total_articles,
        'total_failed': total_failed,
        'total_blocked': total_blocked,
        'results_dir': results_dir
    }

if __name__ == "__main__":
    main() 