import os
import json
import glob
import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from bertopic import BERTopic
from transformers import pipeline
import torch
from umap import UMAP
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from collections import defaultdict
import numpy as np
from urllib.parse import urlparse
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import html
import sys
from datetime import datetime

# Download required NLTK data (only if not already present)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)  # New tokenizer format
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)


def is_boilerplate_sentence(sentence):
    s = sentence.lower().strip()
    boilerplate_keywords = [
        "please log in", "please sign in", "sign up", "subscribe", "subscription",
        "continue reading", "support local journalism", "create an account",
        "purchase subscription", "unlimited access", "privacy policy", "terms of service",
        "manage your subscription", "newsletter", "comments are closed",
        "this article is for subscribers", "register to continue", "activate your account",
        "click here to subscribe", "get unlimited access", "your browser is out of date",
        "your current subscription", "article updated", "consider subscribing", "your current subscription",
        "yahoo is using ai to generate takeaways from this article", "this means the info may not always match what's in the article",
        "this article may be updated with new information"
    ]
    # Remove very short or non-informative sentences
    if len(s.split()) < 4:
        return True
    for phrase in boilerplate_keywords:
        if phrase in s:
            return True
    return False


def get_root_domain(url):
    try:
        return urlparse(url).netloc
    except Exception:
        return "Unknown"


def load_config(config_file="sentiment_config.json"):
    with open(config_file, 'r') as f:
        return json.load(f)


def load_articles(input_dir):
    articles = []
    # Look for JSON files in the input directory and its subdirectories
    for root, dirs, files in os.walk(input_dir):
        # Skip the failed URLs directory
        if 'failed' in root:
            continue
            
        for file_name in files:
            if file_name.endswith('.json'):
                file_path = os.path.join(root, file_name)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = json.load(f)
                        # Skip if content is a list (failed URLs) or doesn't have required fields
                        if isinstance(content, list) or not isinstance(content, dict):
                            continue
                        text = content.get('text', '').strip()
                        url = content.get('source')
                        if text and url:
                            articles.append({
                                'file_path': file_path,
                                'content': text,
                                'title': content.get('title', 'Untitled'),
                                'source_url': url,
                                'search_term': content.get('search_term'),
                                'published_date': content.get('published_date')
                            })
                except Exception as e:
                    print(f"Error loading article from {file_path}: {str(e)}")
                    continue
    return articles


def preprocess_articles(articles, min_text_length=50):
    processed_articles = []
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    for article in articles:
        content = article['content']
        if len(content.split()) < min_text_length:
            continue
        content = re.sub(r'\s+', ' ', content)
        sentences = sent_tokenize(content)
        processed_sentences = []
        for sentence in sentences:
            original_sentence = sentence.strip()
            if is_boilerplate_sentence(original_sentence):
                continue
            cleaned = re.sub(r'[^\w\s]', '', original_sentence)
            words = word_tokenize(cleaned.lower())
            words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
            if words:
                processed_sentences.append({
                    'original': original_sentence,
                    'processed': ' '.join(words)
                })
        if processed_sentences:
            processed_articles.append({
                'file_path': article['file_path'],
                'title': article['title'],
                'content': content,
                'processed_sentences': processed_sentences,
                'source_url': article['source_url'],
                'search_term': article.get('search_term'),
                'published_date': article.get('published_date')
            })
    return processed_articles


def create_article_topic_model(processed_articles, n_topics=5):
    # Use the processed (stopword-removed, lemmatized) article text for topic modeling
    article_texts = []
    article_titles = []
    for a in processed_articles:
        # Join all processed sentences for the article
        processed_text = ' '.join([s['processed'] for s in a['processed_sentences']])
        article_texts.append(processed_text)
        article_titles.append(a['title'])
    bertopic_model = BERTopic(
        embedding_model=SentenceTransformer("all-MiniLM-L6-v2"),
        umap_model=UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=42),
        hdbscan_model=HDBSCAN(min_cluster_size=2, min_samples=1, metric='euclidean', cluster_selection_method='eom', prediction_data=True),
        nr_topics=n_topics,
        top_n_words=10,
        verbose=False
    )
    topics, probs = bertopic_model.fit_transform(article_texts)
    topic_info = bertopic_model.get_topic_info()
    custom_topics = []
    for idx, row in topic_info.iterrows():
        topic_id = row['Topic']
        if topic_id != -1:
            words_scores = bertopic_model.get_topic(topic_id)
            top_words = [word for word, _ in words_scores]
            topic_articles = []
            for i, t in enumerate(topics):
                if t == topic_id:
                    topic_articles.append({
                        'title': article_titles[i],
                        'content': processed_articles[i]['content'],
                        'index': i,
                        'proba': probs[i]
                    })
            custom_topics.append({
                'id': topic_id,
                'name': f"Topic {topic_id}",
                'top_words': top_words,
                'articles': topic_articles,
                'count': len(topic_articles)
            })
    custom_topics.sort(key=lambda x: x['count'], reverse=True)
    return custom_topics


def analyze_sentiment(text, sentiment_pipeline):
    max_length = 512
    tokenizer = sentiment_pipeline.tokenizer
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_length = 0
    for sentence in sentences:
        sentence_tokens = tokenizer(sentence, truncation=True, max_length=max_length, return_tensors="pt")
        sentence_length = sentence_tokens.input_ids.shape[1]
        if current_length + sentence_length > max_length and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_length = sentence_length
        else:
            current_chunk.append(sentence)
            current_length += sentence_length
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    if not chunks:
        chunks = [text]
    star_scores = []
    for chunk in chunks:
        if not chunk.strip():
            continue
        try:
            result = sentiment_pipeline(chunk, truncation=True, max_length=max_length)[0]
            label = result['label']
            stars = None
            if 'star' in label.lower():
                stars = int(label.split()[0])
            star_score = (stars - 3) / 2 if stars is not None else 0.0
            star_scores.append(star_score)
        except Exception:
            continue
    total = len(star_scores)
    if total == 0:
        return {
            'score': 0.0,
            'chunks_analyzed': 0,
            'star_score': 0.0
        }
    avg_star_score = sum(star_scores) / total
    return {
        'score': avg_star_score,
        'chunks_analyzed': len(chunks),
        'star_score': avg_star_score
    }


def analyze_articles_with_topics(processed_articles, topics, sentiment_pipeline):
    results = []
    # Build a mapping from article title to topic ids
    article_to_topic_ids = defaultdict(list)
    for topic in topics:
        for art in topic['articles']:
            article_to_topic_ids[art['title']].append(topic['id'])
    for article in processed_articles:
        # Find all topics this article belongs to
        topic_ids = article_to_topic_ids.get(article['title'], [])
        article_topics = []
        for topic in topics:
            if topic['id'] in topic_ids:
                article_topics.append({
                    'topic_id': topic['id'],
                    'top_words': topic['top_words'],
                    # For reporting, use the article's summary or fallback
                    'summary': article.get('summary') or ' '.join(article['content'].split()[:20]) + '...'
                })
        article_sentiment = analyze_sentiment(article['content'], sentiment_pipeline)
        topic_sentiments = []
        for topic in article_topics:
            # Use the summary for topic-level sentiment
            topic_text = topic['summary']
            topic_sentiment = analyze_sentiment(topic_text, sentiment_pipeline)
            topic_sentiments.append({
                'topic_id': topic['topic_id'],
                'top_words': topic['top_words'],
                'sentiment': topic_sentiment
            })
        results.append({
            'title': article['title'],
            'file_path': article['file_path'],
            'source_url': article['source_url'],
            'article_sentiment': article_sentiment,
            'topics': topic_sentiments,
            'search_term': article.get('search_term'),
            'published_date': article.get('published_date')
        })
    return results


def classify_sentiment(score):
    # Make thresholds more sensitive to sentiment variations
    if score >= 0.3:  # Was 0.7
        return "Very Positive"
    elif score >= 0.15:  # Was 0.35
        return "Positive"
    elif score > -0.15:  # Was -0.35
        return "Neutral"
    elif score > -0.3:  # Was -0.7
        return "Negative"
    else:
        return "Very Negative"


def sentiment_color(label):
    if label == "Very Positive":
        return '#2ca02c'
    elif label == "Positive":
        return '#98df8a'
    elif label == "Neutral":
        return '#d3d3d3'
    elif label == "Negative":
        return '#ff9896'
    elif label == "Very Negative":
        return '#d62728'
    else:
        return '#cccccc'


def generate_visualizations(topic_infos, source_scores, source_article_count, all_scores, processed_articles, output_dir, per_search_term_scores=None):
    # Sentiment by Topic (Bar Chart)
    topic_names = [', '.join(t['top_words'][:6]) for t in topic_infos]
    topic_scores = [t['avg_score'] for t in topic_infos]
    topic_colors = [sentiment_color(t['sentiment_label']) for t in topic_infos]
    plt.figure(figsize=(max(10, len(topic_names) * 1.2), 5))
    plt.bar(topic_names, topic_scores, color=topic_colors)
    plt.ylabel('Average Sentiment Score')
    plt.title('Sentiment by Topic')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    topic_chart_path = os.path.join(output_dir, 'sentiment_by_topic.png')
    plt.savefig(topic_chart_path)
    plt.close()

    # Sentiment by Source (Bar Chart, sorted)
    sources = list(source_scores.keys())
    src_scores = [np.mean(source_scores[s]) for s in sources]
    src_labels = [classify_sentiment(np.mean(source_scores[s])) for s in sources]
    src_counts = [source_article_count[s] for s in sources]
    sorted_src = sorted(zip(sources, src_scores, src_labels, src_counts), key=lambda x: x[1], reverse=True)
    sorted_sources, sorted_scores, sorted_labels, sorted_counts = zip(*sorted_src) if sorted_src else ([], [], [], [])
    src_colors = [sentiment_color(label) for label in sorted_labels]
    plt.figure(figsize=(max(12, len(sorted_sources) * 0.6), 5))
    plt.bar(sorted_sources, sorted_scores, color=src_colors)
    plt.ylabel('Average Sentiment Score')
    plt.title('Sentiment by Source (Sorted)')
    plt.xticks(rotation=90)
    plt.tight_layout()
    src_chart_path = os.path.join(output_dir, 'sentiment_by_source.png')
    plt.savefig(src_chart_path)
    plt.close()

    # Overall Sentiment (Pie Chart) for all articles
    all_labels = [classify_sentiment(score) for score in all_scores]
    from collections import Counter
    sentiment_counts = Counter(all_labels)
    labels = list(sentiment_counts.keys())
    sizes = list(sentiment_counts.values())
    colors = [sentiment_color(l) for l in labels]
    plt.figure(figsize=(7, 7))
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
    plt.title('Overall Sentiment Distribution (All Articles)')
    plt.tight_layout()
    pie_chart_path = os.path.join(output_dir, 'overall_sentiment_pie.png')
    plt.savefig(pie_chart_path)
    plt.close()

    # Per-search-term pie charts
    per_search_term_pie_paths = {}
    if per_search_term_scores:
        for term, scores in per_search_term_scores.items():
            if term is None:
                continue
            term_labels = [classify_sentiment(s) for s in scores]
            term_counts = Counter(term_labels)
            t_labels = list(term_counts.keys())
            t_sizes = list(term_counts.values())
            t_colors = [sentiment_color(l) for l in t_labels]
            plt.figure(figsize=(6, 6))
            plt.pie(t_sizes, labels=t_labels, colors=t_colors, autopct='%1.1f%%', startangle=140)
            plt.title(f'Sentiment Distribution: {term}')
            plt.tight_layout()
            safe_term = re.sub(r'[^a-zA-Z0-9_]', '_', term)[:40]
            pie_path = os.path.join(output_dir, f'sentiment_pie_{safe_term}.png')
            plt.savefig(pie_path)
            plt.close()
            per_search_term_pie_paths[term] = pie_path

    # Word Cloud for all processed words from all articles
    all_processed_words = []
    for a in processed_articles:
        for s in a['processed_sentences']:
            all_processed_words.extend(s['processed'].split())
    wordcloud = WordCloud(width=1200, height=600, background_color='white').generate(' '.join(all_processed_words))
    plt.figure(figsize=(12, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud of All Article Words')
    plt.tight_layout()
    wordcloud_path = os.path.join(output_dir, 'all_wordcloud.png')
    plt.savefig(wordcloud_path)
    plt.close()

    return topic_chart_path, src_chart_path, pie_chart_path, wordcloud_path, sorted_src, topic_names, per_search_term_pie_paths


def generate_html_report(results, topics, output_dir, processed_articles, sentiment_pipeline=None):
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, 'sentiment_report.html')
    source_scores = defaultdict(list)
    source_article_count = defaultdict(int)
    all_scores = []
    source_root_map = {}
    article_title_to_score = {r['title']: r['article_sentiment']['star_score'] for r in results}
    # Map source to article info for linking
    source_to_articles = defaultdict(list)
    for a in processed_articles:
        root = get_root_domain(a['source_url'])
        source_root_map[a['title']] = root
        source_to_articles[root].append({'title': a['title'], 'url': a['source_url']})
    for r in results:
        root = get_root_domain(r['source_url'])
        score = r['article_sentiment'].get('star_score', r['article_sentiment'].get('score', 0.0))
        source_scores[root].append(score)
        source_article_count[root] += 1
        all_scores.append(score)
    topic_infos = []
    for topic in topics:
        quotes = []
        for art in topic['articles'][:5]:
            sentences = sent_tokenize(art['content'])
            good_sent = None
            for s in sentences:
                if not is_boilerplate_sentence(s) and len(s.split()) > 6:
                    good_sent = s.strip()
                    break
            if good_sent:
                quotes.append({'text': good_sent, 'source': processed_articles[art['index']]['source_url']})
            else:
                summary = processed_articles[art['index']].get('summary')
                if not summary and sentiment_pipeline:
                    try:
                        summary = sentiment_pipeline(art['content'][:512])[0]['summary_text']
                    except Exception:
                        summary = None
                if not summary:
                    summary = ' '.join(art['content'].split()[:20]) + '...'
                quotes.append({'text': summary, 'source': processed_articles[art['index']]['source_url']})
        topic_scores = [article_title_to_score.get(art['title'], 0.0) for art in topic['articles']]
        avg_topic_score = sum(topic_scores) / len(topic_scores) if topic_scores else 0.0
        sentiment_label = classify_sentiment(avg_topic_score)
        topic_infos.append({
            'topic_id': topic['id'],
            'top_words': topic['top_words'],
            'avg_score': avg_topic_score,
            'sentiment_label': sentiment_label,
            'quotes': quotes
        })
    # Collect per-search-term sentiment scores
    per_search_term_scores = defaultdict(list)
    per_search_term_articles = defaultdict(list)
    for r in results:
        search_term = r['search_term']
        if search_term is not None:
            per_search_term_scores[search_term].append(r['article_sentiment'].get('star_score', 0.0))
            per_search_term_articles[search_term].append(r)
    # Get all unique search terms from the articles
    search_terms = sorted([term for term in per_search_term_scores if term is not None])
    # Generate and save visualizations (now returns per-search-term pie chart paths)
    topic_chart_path, src_chart_path, pie_chart_path, wordcloud_path, sorted_src, topic_names, per_search_term_pie_paths = generate_visualizations(
        topic_infos, source_scores, source_article_count, all_scores, processed_articles, output_dir, per_search_term_scores=per_search_term_scores)
    # Build a mapping from article title to sentiment info
    article_sentiment_map = {r['title']: r['article_sentiment'] for r in results}
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("""
<!DOCTYPE html>
<html lang=\"en\">
<head>
<meta charset=\"UTF-8\">
<title>Sentiment Analysis Report</title>
<style>
    body { font-family: Arial, sans-serif; margin: 40px; background: #fafbfc; color: #222; }
    h1, h2, h3 { color: #2c3e50; }
    .search-terms { font-size: 1.2em; background: #e8f0fe; color: #174ea6; padding: 10px 18px; border-radius: 8px; margin-bottom: 18px; display: inline-block; }
    table { border-collapse: collapse; width: 100%; margin-bottom: 2em; }
    th, td { border: 1px solid #ccc; padding: 8px 12px; text-align: left; }
    th { background: #f4f4f4; }
    tr:nth-child(even) { background: #f9f9f9; }
    .sentiment-label { font-weight: bold; padding: 2px 8px; border-radius: 6px; }
    .Very-Positive { background: #2ca02c; color: #fff; }
    .Positive { background: #98df8a; color: #222; }
    .Neutral { background: #d3d3d3; color: #222; }
    .Negative { background: #ff9896; color: #222; }
    .Very-Negative { background: #d62728; color: #fff; }
    img { max-width: 100%; height: auto; margin-bottom: 2em; border: 1px solid #ccc; }
    .quote { margin: 0.5em 0 0.5em 1.5em; padding-left: 1em; border-left: 3px solid #eee; color: #444; }
    ul.article-list { margin: 0.5em 0 0.5em 2em; padding-left: 1em; }
    ul.article-list li { font-size: 0.98em; margin-bottom: 0.2em; }
</style>
</head>
<body>
""")
        # Add search terms at the very top
        if search_terms:
            f.write(f'<div class="search-terms"><b>Search Terms:</b> {html.escape(", ".join(search_terms))}</div><br>\n')
        # Add date range if available
        all_dates = [a.get('published_date') for a in processed_articles if a.get('published_date')]
        if all_dates:
            try:
                date_objs = [datetime.strptime(d, "%Y-%m-%d %H:%M:%S") for d in all_dates]
                min_date = min(date_objs).strftime("%Y-%m-%d")
                max_date = max(date_objs).strftime("%Y-%m-%d")
                f.write(f'<div class="search-terms"><b>Date Range:</b> {min_date} to {max_date}</div><br>\n')
            except Exception:
                pass
        f.write("<h1>Sentiment Analysis Report</h1>\n")
        # 1. Overall Sentiment (number, avg, label, search terms)
        if all_scores and (not search_terms or len(search_terms) > 1):
            avg = np.mean(all_scores)
            overall_label = classify_sentiment(avg)
            f.write(f'<h2>Overall Sentiment</h2>\n')
            f.write(f'<b>Number of Articles:</b> {len(all_scores)}<br>\n')
            if search_terms:
                f.write(f'<b>Search Terms:</b> {html.escape(", ".join(search_terms))}<br>\n')
            f.write(f'<b>Average Sentiment Score:</b> {avg:.2f} <span class="sentiment-label {overall_label.replace(" ", "-")}">{overall_label}</span><br><br>\n')
            f.write(f'<h2>Overall Sentiment Distribution (All Articles)</h2>\n')
            f.write(f'<img src="{os.path.basename(pie_chart_path)}" alt="Overall Sentiment Pie Chart"><br>\n')
        # Per-search-term sentiment
        if search_terms:
            for term in search_terms:
                scores = per_search_term_scores.get(term, [])
                if not scores:
                    continue
                avg = np.mean(scores)
                label = classify_sentiment(avg)
                f.write(f'<h2>Sentiment for Search Term: {html.escape(term)}</h2>\n')
                f.write(f'<b>Number of Articles:</b> {len(scores)}<br>\n')
                f.write(f'<b>Average Sentiment Score:</b> {avg:.2f} <span class="sentiment-label {label.replace(" ", "-")}">{label}</span><br>\n')
                pie_path = per_search_term_pie_paths.get(term)
                if pie_path:
                    f.write(f'<img src="{os.path.basename(pie_path)}" alt="Sentiment Pie for {html.escape(term)}"><br>\n')
        # 2. Topic sections (text, quotes, etc.), then sentiment by topic chart
        f.write(f'<h2>Topics</h2>\n')
        for idx, t in enumerate(topic_infos):
            topic_name = ', '.join(t['top_words'][:6])
            f.write(f'<h3>Topic: {topic_name}</h3>\n')
            f.write(f'<b>Average Sentiment Score:</b> {t["avg_score"]:.2f} <span class="sentiment-label {t["sentiment_label"].replace(" ", "-")}">{t["sentiment_label"]}</span><br>\n')
            f.write(f'<b>Quotes:</b><br>\n')
            for q in t['quotes']:
                # Find sentiment for this article
                art_title = None
                for a in processed_articles:
                    if a['source_url'] == q['source']:
                        art_title = a['title']
                        break
                sentiment_info = article_sentiment_map.get(art_title, {'star_score': 0.0})
                score = sentiment_info.get('star_score', 0.0)
                label = classify_sentiment(score)
                f.write(f'<div class="quote">"{q["text"]}" <span style="font-size:0.9em; color:#888;">(Source: <a href="{q["source"]}" target="_blank">{q["source"]}</a> | Sentiment: {score:.2f} <span class="sentiment-label {label.replace(" ", "-")}">{label}</span>)</span></div>\n')
            f.write('<br>\n')
        f.write(f'<h2>Sentiment by Topic</h2>\n')
        f.write(f'<img src="{os.path.basename(topic_chart_path)}" alt="Sentiment by Topic"><br>\n')
        # 4. Sources sorted by sentiment, table with article links, then chart
        f.write(f'<h2>Sources Sorted by Sentiment</h2>\n')
        f.write('<table>\n<tr><th>Source</th><th>Avg Sentiment</th><th>Label</th><th>Article Count</th><th>Articles</th></tr>\n')
        for src, score, label, count in sorted_src:
            f.write(f'<tr><td>{src}</td><td>{score:.2f}</td><td><span class="sentiment-label {label.replace(" ", "-")}">{label}</span></td><td>{count}</td><td>')
            f.write('<ul class="article-list">')
            for art in source_to_articles[src]:
                sentiment_info = article_sentiment_map.get(art['title'], {'star_score': 0.0})
                art_score = sentiment_info.get('star_score', 0.0)
                art_label = classify_sentiment(art_score)
                f.write(f'<li><a href="{art["url"]}" target="_blank">{html.escape(art["title"])}</a> | Sentiment: {art_score:.2f} <span class="sentiment-label {art_label.replace(" ", "-")}">{art_label}</span></li>')
            f.write('</ul></td></tr>\n')
        f.write('</table>\n')
        f.write(f'<img src="{os.path.basename(src_chart_path)}" alt="Sentiment by Source"><br>\n')
        # 5. Word Cloud for all processed words from all articles
        f.write(f'<h2>Word Cloud of All Article Words</h2>\n')
        f.write(f'<img src="{os.path.basename(wordcloud_path)}" alt="Word Cloud of All Article Words"><br>\n')
        
        # 6. All Articles Appendix
        f.write('<h2>Appendix: All Articles</h2>\n')
        f.write('<table>\n<tr><th>Title</th><th>Sentiment</th><th>Source</th><th>Summary</th></tr>\n')
        # Sort results by sentiment score (most negative to most positive)
        sorted_results = sorted(results, key=lambda r: r['article_sentiment'].get('star_score', 0.0))
        for r in sorted_results:
            title = html.escape(r['title'])
            url = r['source_url']
            score = r['article_sentiment'].get('star_score', 0.0)
            label = classify_sentiment(score)
            label_class = label.replace(' ', '-')
            source = get_root_domain(url)
            
            # Get article summary if available
            article_summary = ""
            for a in processed_articles:
                if a['title'] == r['title']:
                    # Try to get summary from article
                    summary = a.get('summary')
                    # If no summary exists, try to generate one
                    if not summary and sentiment_pipeline:
                        try:
                            summary = sentiment_pipeline(a['content'][:512])[0]['summary_text']
                        except Exception:
                            pass
                    # If still no summary, use first sentence or part of content
                    if not summary:
                        sentences = sent_tokenize(a['content'])
                        if sentences:
                            summary = sentences[0]
                        else:
                            summary = a['content'][:150] + "..."
                    
                    article_summary = html.escape(summary[:200] + "..." if len(summary) > 200 else summary)
                    break
            
            f.write(f'<tr><td><a href="{url}" target="_blank">{title}</a></td>'
                    f'<td>{score:.2f} <span class="sentiment-label {label_class}">{label}</span></td>'
                    f'<td>{html.escape(source)}</td>'
                    f'<td style="max-width:400px; font-size:0.9em;">{article_summary}</td></tr>\n')
        f.write('</table>\n')
        f.write("</body>\n</html>")
    return report_path


def generate_training_chunks(processed_articles, results, output_dir):
    """Generate training chunks from the analyzed articles for later training data creation"""
    training_chunks = []
    chunk_id = 0
    
    # Create a mapping from article title to sentiment results
    article_sentiment_map = {r['title']: r['article_sentiment'] for r in results}
    
    for article in processed_articles:
        article_title = article['title']
        article_content = article['content']
        
        # Get the article's overall sentiment
        article_sentiment = article_sentiment_map.get(article_title, {'star_score': 0.0})
        overall_score = article_sentiment.get('star_score', 0.0)
        overall_rating = score_to_rating(overall_score)
        
        # Split article into chunks (similar to how sentiment analysis does it)
        sentences = sent_tokenize(article_content)
        chunks = []
        current_chunk = []
        current_length = 0
        max_length = 512  # Similar to sentiment analysis
        
        for sentence in sentences:
            # Skip boilerplate sentences
            if is_boilerplate_sentence(sentence):
                continue
                
            # Estimate token length (rough approximation)
            sentence_length = len(sentence.split()) * 1.3  # Rough token estimate
            
            if current_length + sentence_length > max_length and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_length = sentence_length
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        # Create training chunks for each chunk
        for chunk_text in chunks:
            if len(chunk_text.strip()) < 50:  # Skip very short chunks
                continue
                
            training_chunks.append({
                'chunk_id': chunk_id,
                'article_title': article_title,
                'chunk_text': chunk_text,
                'source_url': article['source_url'],
                'search_term': article.get('search_term'),
                'published_date': article.get('published_date'),
                'rating': overall_rating,  # Default rating based on overall sentiment
                'rating_text': rating_to_text(overall_rating),
                'original_sentiment_score': overall_score
            })
            chunk_id += 1
    
    # Save training chunks to file
    chunks_file = os.path.join(output_dir, 'training_chunks.json')
    try:
        with open(chunks_file, 'w', encoding='utf-8') as f:
            json.dump(training_chunks, f, indent=2)
        print(f"Generated {len(training_chunks)} training chunks saved to: {chunks_file}")
    except Exception as e:
        print(f"Error saving training chunks: {str(e)}")
    
    return training_chunks


def score_to_rating(score):
    """Convert sentiment score to 1-5 rating scale"""
    if score >= 0.3:
        return 5  # very positive
    elif score >= 0.15:
        return 4  # positive
    elif score > -0.15:
        return 3  # neutral
    elif score > -0.3:
        return 2  # negative
    else:
        return 1  # very negative


def rating_to_text(rating):
    """Convert rating number to text"""
    rating_map = {
        1: "very negative",
        2: "negative", 
        3: "neutral",
        4: "positive",
        5: "very positive"
    }
    return rating_map.get(rating, "neutral")


def main():
    # Get progress callback if available
    progress_callback = getattr(sys.modules[__name__], 'progress_callback', None)
    
    try:
        if progress_callback:
            progress_callback("Loading configuration...", 0)
        
        config = load_config()
        input_dir = config['files'].get('input_directory', 'articles_newspaper/results')
        output_dir = config['files'].get('output_directory', 'articles_newspaper/reports')
        min_text_length = config['analysis'].get('min_text_length', 50)
        n_topics = config['topic_modeling'].get('n_topics', 5)
        model_name = config['topic_modeling'].get('sentiment_model', 'distilbert-base-uncased-finetuned-sst-2-english')
        
        if progress_callback:
            progress_callback("Setting up sentiment analysis model...", 10)
        
        device = 0 if torch.cuda.is_available() else -1
        sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model=model_name,
            device=device,
            truncation=True,
            max_length=512
        )
        
        if progress_callback:
            progress_callback("Loading articles...", 20)
        
        articles = load_articles(input_dir)
        if not articles:
            if progress_callback:
                progress_callback("No articles found in input directory", 100)
            return
        
        if progress_callback:
            progress_callback("Processing articles...", 30)
        
        processed_articles = preprocess_articles(
            articles,
            min_text_length=min_text_length
        )
        
        if progress_callback:
            progress_callback("Creating topic model...", 50)
        
        topics = create_article_topic_model(
            processed_articles,
            n_topics=n_topics
        )
        
        if progress_callback:
            progress_callback("Analyzing sentiment...", 70)
        
        results = analyze_articles_with_topics(processed_articles, topics, sentiment_pipeline)
        
        if progress_callback:
            progress_callback("Generating report...", 90)
        
        os.makedirs(output_dir, exist_ok=True)
        report_path = generate_html_report(results, topics, output_dir, processed_articles, sentiment_pipeline=sentiment_pipeline)
        
        if progress_callback:
            progress_callback(f"Analysis complete! Report saved to {report_path}", 100)
        
        training_chunks = generate_training_chunks(processed_articles, results, output_dir)
        
        return {
            'success': True,
            'report_path': report_path,
            'articles_analyzed': len(processed_articles),
            'topics_found': len(topics),
            'training_chunks': training_chunks
        }
        
    except Exception as e:
        error_msg = f"Error during sentiment analysis: {str(e)}"
        if progress_callback:
            progress_callback(error_msg, 100)
        else:
            print(error_msg)
        return {
            'success': False,
            'error': error_msg
        }


if __name__ == "__main__":
    main()