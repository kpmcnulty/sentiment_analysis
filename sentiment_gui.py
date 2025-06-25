import os
import sys
import json
import threading
import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox
import subprocess
import importlib.util
import newspaper_scraper
import simple_topic_sentiment
from datetime import datetime
import time
import nltk
from nltk.tokenize import sent_tokenize
import glob
import webbrowser
import multiprocessing

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Check if required modules are installed
required_modules = [
    'newspaper', 'GoogleNews', 'nltk', 'bertopic', 
    'transformers', 'torch', 'umap', 'hdbscan', 
    'sentence_transformers', 'matplotlib', 'wordcloud'
]

def get_resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller"""
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        # Not running as PyInstaller bundle, check if we're in an app bundle
        if sys.platform == 'darwin' and '.app/Contents/MacOS' in sys.executable:
            # We're running from inside a .app bundle
            # The executable is at: SentimentAnalysisTool.app/Contents/MacOS/SentimentAnalysisTool
            # Resources are at: SentimentAnalysisTool.app/Contents/Resources/
            app_contents = os.path.dirname(os.path.dirname(sys.executable))
            base_path = os.path.join(app_contents, 'Resources')
        else:
            # Development mode or standalone executable
            base_path = os.path.abspath(".")
    
    return os.path.join(base_path, relative_path)

def check_imports():
    missing = []
    for module in required_modules:
        if importlib.util.find_spec(module) is None:
            missing.append(module)
    return missing

def find_latest_model(models_dir="models"):
    """Find the most recently trained model directory"""
    if not os.path.exists(models_dir):
        return None
        
    model_dirs = [d for d in os.listdir(models_dir) if d.startswith("sentiment_model_")]
    if not model_dirs:
        return None
        
    # Sort by timestamp in directory name
    latest_model = sorted(model_dirs)[-1]
    return os.path.join(models_dir, latest_model)

class SentimentAnalysisGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("News Sentiment Analysis Tool")
        self.root.geometry("1200x800")
        self.root.minsize(1200, 800)
        
        # Initialize configurations
        self.scraper_config = self.load_config("scraper_config.json")
        self.sentiment_config = self.load_config("sentiment_config.json")
        
        # Find and set the latest model
        latest_model = find_latest_model()
        if latest_model:
            print(f"Using latest trained model: {latest_model}")
            self.sentiment_config['topic_modeling']['sentiment_model'] = latest_model
            self.save_config(self.sentiment_config, "sentiment_config.json")
        
        # Initialize run history
        self.run_history = self.load_run_history()
        
        self.create_gui()
        
    def load_config(self, config_file):
        # First try to load from bundled resources
        bundled_config_path = get_resource_path(config_file)
        if os.path.exists(bundled_config_path):
            try:
                with open(bundled_config_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading bundled {config_file}: {str(e)}")
        
        # Fall back to current directory (for development or user configs)
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading {config_file}: {str(e)}")
        
        # Return default config if neither exists
        if config_file == "scraper_config.json":
            return {
                "max_articles": 100,
                "timeout": 30,
                "user_agent": "Mozilla/5.0 (compatible; SentimentAnalyzer/1.0)",
                "search_terms": ["Vandenberg Space Force Base"],
                "period": "1m",
                "max_results": 100,
                "max_consecutive_no_relevant": 3,
                "blocked_domains": ["spaceforce.mil", "vandenberg.spaceforce.mil", "defense.gov", "youtube.com", "nasa.gov"]
            }
        elif config_file == "sentiment_config.json":
            return {
                "model_name": "cardiffnlp/twitter-roberta-base-sentiment-latest",
                "batch_size": 32,
                "max_length": 512
            }
        return {}
    
    def save_config(self, config, config_file):
        try:
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving {config_file}: {str(e)}")
            return False
    
    def load_run_history(self):
        """Load run history from file or create empty history"""
        history_file = "run_history.json"
        if os.path.exists(history_file):
            try:
                with open(history_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading run history: {str(e)}")
        
        # Default empty history with default output locations
        return {
            "runs": [],
            "current_output_dir": "articles/results",
            "current_report_dir": "articles/reports"
        }
    
    def save_run_history(self):
        """Save run history to file"""
        history_file = "run_history.json"
        try:
            with open(history_file, 'w', encoding='utf-8') as f:
                json.dump(self.run_history, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving run history: {str(e)}")
            return False
    
    def add_to_run_history(self, search_terms, time_period, max_results, output_dir, report_dir, timestamp):
        """Add a new run to the history"""
        # Get blocked domains
        blocked_domains = [domain.strip() for domain in self.blocked_domains_text.get('1.0', tk.END).splitlines() if domain.strip()]
        
        # Create a new run entry
        run_entry = {
            "timestamp": timestamp,
            "search_terms": search_terms,
            "time_period": time_period,
            "max_results": max_results,
            "output_dir": output_dir,
            "report_dir": report_dir,
            "blocked_domains": blocked_domains
        }
        
        # Add to history
        self.run_history["runs"].insert(0, run_entry)  # Add to beginning of list (most recent first)
        
        # Limit to last 10 runs
        if len(self.run_history["runs"]) > 10:
            self.run_history["runs"] = self.run_history["runs"][:10]
        
        # Update current dirs
        self.run_history["current_output_dir"] = output_dir
        self.run_history["current_report_dir"] = report_dir
        
        # Save history
        self.save_run_history()
        
        # Update runs tree
        self.update_runs_tree()

    def create_gui(self):
        # Create notebook for tabs
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Create main tab
        main_tab = ttk.Frame(notebook)
        notebook.add(main_tab, text="News Sentiment Analysis")
        self.create_main_tab(main_tab)
        
        # Create previous runs tab
        previous_runs_tab = ttk.Frame(notebook)
        notebook.add(previous_runs_tab, text="Previous Runs")
        self.create_previous_runs_tab(previous_runs_tab)
        
        # Create feedback management tab
        feedback_tab = ttk.Frame(notebook)
        notebook.add(feedback_tab, text="Training Data Management")
        self.create_feedback_tab(feedback_tab)
        
        # Save notebook reference for tab switching
        self.notebook = notebook
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def create_main_tab(self, parent):
        # Create frame with padding
        frame = ttk.Frame(parent, padding="10")
        frame.pack(fill='both', expand=True)
        
        # Search terms
        ttk.Label(frame, text="Search Terms (one per line):").grid(column=0, row=0, sticky=tk.W, pady=5)
        self.search_terms_text = scrolledtext.ScrolledText(frame, width=40, height=5)
        self.search_terms_text.grid(column=0, row=1, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        # Populate search terms
        if 'search_terms' in self.scraper_config:
            self.search_terms_text.insert(tk.END, '\n'.join(self.scraper_config['search_terms']))
        
        # Simple options frame
        options_frame = ttk.Frame(frame)
        options_frame.grid(column=0, row=2, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        # Time period
        ttk.Label(options_frame, text="Time Period:").grid(column=0, row=0, sticky=tk.W, padx=5)
        self.period_var = tk.StringVar(value=self.scraper_config.get('period', '1m'))
        period_combo = ttk.Combobox(options_frame, textvariable=self.period_var, values=['1d', '3d', '1w', '2w', '1m', '3m', '6m', '1y'])
        period_combo.grid(column=1, row=0, sticky=tk.W, padx=5)

        # Max results
        ttk.Label(options_frame, text="Max Results:").grid(column=0, row=1, sticky=tk.W, padx=5)
        self.max_results_var = tk.StringVar(value=str(self.scraper_config.get('max_results', 100)))
        ttk.Entry(options_frame, textvariable=self.max_results_var, width=10).grid(column=1, row=1, sticky=tk.W, padx=5)

        # Max consecutive empty pages
        ttk.Label(options_frame, text="Max Consecutive Empty Pages:").grid(column=0, row=2, sticky=tk.W, padx=5)
        self.max_empty_pages_var = tk.StringVar(value=str(self.scraper_config.get('max_consecutive_no_relevant', 3)))
        ttk.Entry(options_frame, textvariable=self.max_empty_pages_var, width=10).grid(column=1, row=2, sticky=tk.W, padx=5)
        ttk.Label(options_frame, text="(How many pages in a row with no new articles before stopping. Increase for more results, decrease for faster runs.)", foreground="#888", font=("TkDefaultFont", 8)).grid(column=0, row=3, columnspan=3, sticky=tk.W, padx=5)
        
        # Blocked domains
        ttk.Label(frame, text="Blocked Domains (one per line):").grid(column=0, row=3, sticky=tk.W, pady=5)
        self.blocked_domains_text = scrolledtext.ScrolledText(frame, width=40, height=5)
        self.blocked_domains_text.grid(column=0, row=4, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        # Populate blocked domains
        if 'blocked_domains' in self.scraper_config:
            self.blocked_domains_text.insert(tk.END, '\n'.join(self.scraper_config['blocked_domains']))
        
        # Metrics frame
        metrics_frame = ttk.LabelFrame(frame, text="Metrics", padding="5")
        metrics_frame.grid(column=0, row=5, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        # Success count
        ttk.Label(metrics_frame, text="Successful:").grid(column=0, row=0, sticky=tk.W, padx=5)
        self.success_count_var = tk.StringVar(value="0")
        ttk.Label(metrics_frame, textvariable=self.success_count_var).grid(column=1, row=0, sticky=tk.W, padx=5)
        
        # Failed count
        ttk.Label(metrics_frame, text="Failed:").grid(column=2, row=0, sticky=tk.W, padx=5)
        self.failed_count_var = tk.StringVar(value="0")
        ttk.Label(metrics_frame, textvariable=self.failed_count_var).grid(column=3, row=0, sticky=tk.W, padx=5)
        
        # Blocked count
        ttk.Label(metrics_frame, text="Blocked:").grid(column=4, row=0, sticky=tk.W, padx=5)
        self.blocked_count_var = tk.StringVar(value="0")
        ttk.Label(metrics_frame, textvariable=self.blocked_count_var).grid(column=5, row=0, sticky=tk.W, padx=5)
        
        # Progress frame
        progress_frame = ttk.Frame(frame)
        progress_frame.grid(column=0, row=6, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_frame, length=300, mode='determinate', variable=self.progress_var)
        self.progress_bar.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # Progress label
        self.progress_label = ttk.Label(progress_frame, text="")
        self.progress_label.pack(side=tk.LEFT, padx=5)
        
        # Buttons
        button_frame = ttk.Frame(frame)
        button_frame.grid(column=0, row=7, columnspan=2, pady=10)
        
        self.run_button = ttk.Button(button_frame, text="Run Analysis", command=self.run_scraper_and_analysis)
        self.run_button.pack(side=tk.LEFT, padx=5)
        
        self.stop_button = ttk.Button(button_frame, text="Stop", command=self.stop_scraping, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(button_frame, text="View Results", command=self.open_latest_report).pack(side=tk.LEFT, padx=5)
        
        # Configure grid to expand
        frame.columnconfigure(0, weight=1)
    
    def create_previous_runs_tab(self, parent):
        frame = ttk.Frame(parent, padding="10")
        frame.pack(fill='both', expand=True)
        
        # Previous Runs label
        ttk.Label(frame, text="Previous Runs:").grid(column=0, row=0, sticky=tk.W, pady=5)
        # Treeview for previous runs
        self.runs_tree = ttk.Treeview(frame, columns=("Timestamp", "Search Terms", "Period", "Max Results"), show="headings", height=10)
        self.runs_tree.heading("Timestamp", text="Timestamp")
        self.runs_tree.heading("Search Terms", text="Search Terms")
        self.runs_tree.heading("Period", text="Period")
        self.runs_tree.heading("Max Results", text="Max Results")
        self.runs_tree.column("Timestamp", width=120)
        self.runs_tree.column("Search Terms", width=200)
        self.runs_tree.column("Period", width=80)
        self.runs_tree.column("Max Results", width=80)
        self.runs_tree.grid(column=0, row=1, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        self.runs_tree.bind("<<TreeviewSelect>>", self.on_run_selected)
        self.update_runs_tree()
        
        # Action buttons
        self.load_settings_btn = ttk.Button(frame, text="Load Settings", command=self.load_selected_run_settings, state=tk.DISABLED)
        self.open_report_btn = ttk.Button(frame, text="Open Report", command=self.open_selected_run_report, state=tk.DISABLED)
        self.open_folder_btn = ttk.Button(frame, text="Open Results Folder", command=self.open_selected_run_folder, state=tk.DISABLED)
        self.load_settings_btn.grid(column=0, row=2, pady=10, sticky=tk.W)
        self.open_report_btn.grid(column=1, row=2, pady=10, sticky=tk.W)
        self.open_folder_btn.grid(column=2, row=2, pady=10, sticky=tk.W)
        frame.columnconfigure(0, weight=1)
        frame.columnconfigure(1, weight=1)
        frame.columnconfigure(2, weight=1)
    
    def on_run_selected(self, event):
        selected = self.runs_tree.selection()
        if not selected:
            self.load_settings_btn.config(state=tk.DISABLED)
            self.open_report_btn.config(state=tk.DISABLED)
            self.open_folder_btn.config(state=tk.DISABLED)
            return
        self.load_settings_btn.config(state=tk.NORMAL)
        self.open_report_btn.config(state=tk.NORMAL)
        self.open_folder_btn.config(state=tk.NORMAL)
    
    def load_selected_run_settings(self):
        selected = self.runs_tree.selection()
        if not selected:
            return
        idx = self.runs_tree.index(selected[0])
        run = self.run_history["runs"][idx]
        # Switch to main tab and populate fields
        self.root.nametowidget(self.root.winfo_children()[0]).select(0)  # Switch to main tab
        self.search_terms_text.delete('1.0', tk.END)
        self.blocked_domains_text.delete('1.0', tk.END)
        if "search_terms" in run:
            self.search_terms_text.insert(tk.END, '\n'.join(run["search_terms"]))
        if "time_period" in run:
            self.period_var.set(run["time_period"])
        if "max_results" in run:
            self.max_results_var.set(str(run["max_results"]))
        if "blocked_domains" in run and run["blocked_domains"]:
            self.blocked_domains_text.insert(tk.END, '\n'.join(run["blocked_domains"]))
    
    def open_selected_run_report(self):
        selected = self.runs_tree.selection()
        if not selected:
            return
        idx = self.runs_tree.index(selected[0])
        run = self.run_history["runs"][idx]
        report_dir = run.get("report_dir")
        if not os.path.exists(report_dir):
            messagebox.showerror("Error", f"Report directory not found: {report_dir}")
            return
        html_files = [f for f in os.listdir(report_dir) if f.endswith(".html")]
        if not html_files:
            messagebox.showinfo("Info", "No HTML reports found in the selected run.")
            return
        report_path = os.path.join(report_dir, html_files[0])
        self.open_report_in_browser(report_path)
    
    def open_selected_run_folder(self):
        selected = self.runs_tree.selection()
        if not selected:
            return
        idx = self.runs_tree.index(selected[0])
        run = self.run_history["runs"][idx]
        report_dir = run.get("report_dir")
        if not os.path.exists(report_dir):
            messagebox.showerror("Error", f"Report directory not found: {report_dir}")
            return
        
        # Open folder with more specific handling
        try:
            if sys.platform == 'win32':
                os.startfile(os.path.abspath(report_dir))
            elif sys.platform == 'darwin':
                subprocess.call(['open', '-R', os.path.abspath(report_dir)])
            else:
                subprocess.call(['xdg-open', os.path.abspath(report_dir)])
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open folder: {str(e)}")
    
    def run_scraper_and_analysis(self):
        """Run the scraper and then run the sentiment analysis"""
        # Get values from GUI
        search_terms = [term.strip() for term in self.search_terms_text.get('1.0', tk.END).splitlines() if term.strip()]
        blocked_domains = [domain.strip() for domain in self.blocked_domains_text.get('1.0', tk.END).splitlines() if domain.strip()]
        max_results = int(self.max_results_var.get())
        period = self.period_var.get()
        max_consecutive_no_relevant = int(self.max_empty_pages_var.get())
        
        # Set default delays
        initial_delay = 5
        page_delay = 5
        article_delay = 1
        
        # Create timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create output directories
        output_dir = os.path.join("articles", "results", timestamp)
        report_dir = os.path.join("articles", "reports", timestamp)
        
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(report_dir, exist_ok=True)
        
        # Update scraper config
        self.scraper_config.update({
            'search_terms': search_terms,
            'blocked_domains': blocked_domains,
            'max_results': max_results,
            'period': period,
            'output_directory': output_dir,
            'initial_delay': initial_delay,
            'page_delay': page_delay,
            'delay_between_downloads': article_delay,
            'max_consecutive_no_relevant': max_consecutive_no_relevant,
            'google_news': {
                'max_pages': 10,
                'request_timeout': 30
            }
        })
        
        # Save scraper config to file
        self.save_config(self.scraper_config, "scraper_config.json")
        
        # Update sentiment config
        self.sentiment_config['files']['input_directory'] = output_dir
        self.sentiment_config['files']['output_directory'] = report_dir
        
        # Save sentiment config to file
        self.save_config(self.sentiment_config, "sentiment_config.json")
        
        # Add to history
        self.add_to_run_history(search_terms, period, max_results, output_dir, report_dir, timestamp)
        
        # Run scraper
        self.status_var.set("Running scraper... (this may take a while)")
        self.root.update()
        
        threading.Thread(target=self._run_scraper_and_analysis_thread, args=(timestamp,), daemon=True).start()
    
    def _run_scraper_and_analysis_thread(self, timestamp):
        try:
            # Reset metrics
            self.root.after(0, lambda: self.success_count_var.set("0"))
            self.root.after(0, lambda: self.failed_count_var.set("0"))
            self.root.after(0, lambda: self.blocked_count_var.set("0"))
            
            # Set stop flag
            self.stop_requested = False
            
            # Enable stop button, disable run button
            self.root.after(0, lambda: self.stop_button.configure(state=tk.NORMAL))
            self.root.after(0, lambda: self.run_button.configure(state=tk.DISABLED))
            
            # Run the scraper with progress updates
            def progress_callback(status, progress=None):
                if self.stop_requested:
                    return True  # Signal to stop
                
                # Print to command line
                print(status)
                
                # Update GUI
                self.root.after(0, lambda: self.progress_label.configure(text=status))
                if progress is not None:
                    self.root.after(0, lambda: self.progress_var.set(progress))
                
                # Update metrics if status contains relevant info
                if "Relevant article:" in status:
                    current = int(self.success_count_var.get())
                    self.root.after(0, lambda: self.success_count_var.set(str(current + 1)))
                elif "Failed to extract content" in status:
                    current = int(self.failed_count_var.get())
                    self.root.after(0, lambda: self.failed_count_var.set(str(current + 1)))
                elif "Skipping blocked domain:" in status:
                    current = int(self.blocked_count_var.get())
                    self.root.after(0, lambda: self.blocked_count_var.set(str(current + 1)))
                
                return False  # Continue processing
            
            # Create output directories
            output_dir = os.path.join("articles", "results", timestamp)
            report_dir = os.path.join("articles", "reports", timestamp)
            
            os.makedirs(output_dir, exist_ok=True)
            os.makedirs(report_dir, exist_ok=True)
            
            # Print initial status
            print("\nStarting news scraping and sentiment analysis...")
            print(f"Output directory: {output_dir}")
            print(f"Report directory: {report_dir}\n")
            
            # Modify newspaper_scraper to accept progress callback
            newspaper_scraper.progress_callback = progress_callback
            result = newspaper_scraper.main()
            
            if result:
                # Update final metrics
                self.root.after(0, lambda: self.success_count_var.set(str(result.get('total_articles', 0))))
                self.root.after(0, lambda: self.failed_count_var.set(str(result.get('total_failed', 0))))
                self.root.after(0, lambda: self.blocked_count_var.set(str(result.get('total_blocked', 0))))
                
                # Print summary to command line
                print("\nScraping Summary:")
                print(f"Total articles saved: {result.get('total_articles', 0)}")
                print(f"Total failed extractions: {result.get('total_failed', 0)}")
                print(f"Total blocked domains: {result.get('total_blocked', 0)}")
                print(f"Articles saved to: {result.get('results_dir', output_dir)}\n")
            
            if self.stop_requested:
                print("\nOperation stopped by user")
                self.root.after(0, lambda: self.status_var.set("Operation stopped by user"))
                return
            
            # Update status for sentiment analysis
            print("\nStarting sentiment analysis...")
            self.root.after(0, lambda: self.status_var.set("Scraping completed, starting sentiment analysis..."))
            self.root.after(0, lambda: self.progress_var.set(0))
            
            # Update sentiment config with correct paths
            self.sentiment_config['files'] = {
                'input_directory': output_dir,
                'output_directory': report_dir
            }
            self.save_config(self.sentiment_config, "sentiment_config.json")
            
            # Share progress callback with sentiment analyzer
            simple_topic_sentiment.progress_callback = lambda msg, progress: (
                print(msg),  # Print to command line
                self.root.after(0, lambda: (
                    self.progress_label.configure(text=msg),
                    self.progress_var.set(progress)
                ))
            )
            
            # Run sentiment analysis
            result = simple_topic_sentiment.main()
            
            if not result:
                print("\nNo articles found to analyze")
                self.root.after(0, lambda: self.status_var.set("No articles found to analyze"))
                self.root.after(0, lambda: messagebox.showinfo("Info", "No articles found to analyze"))
                return
            
            if result.get('success', False):
                # Print success message to command line
                print(f"\nAnalysis completed successfully!")
                print(f"Articles analyzed: {result['articles_analyzed']}")
                print(f"Topics found: {result['topics_found']}")
                print(f"Report saved to: {result['report_path']}")
                
                # Update status
                self.root.after(0, lambda: self.status_var.set(
                    f"Analysis completed successfully! Analyzed {result['articles_analyzed']} articles in {result['topics_found']} topics"
                ))
                self.root.after(0, lambda: messagebox.showinfo(
                    "Success", 
                    f"Analysis completed successfully!\n\n"
                    f"Articles analyzed: {result['articles_analyzed']}\n"
                    f"Topics found: {result['topics_found']}\n"
                    f"Report saved to: {result['report_path']}"
                ))
                
                # Open the report in the GUI
                if os.path.exists(result['report_path']):
                    self.root.after(0, lambda: self.open_report_in_browser(result['report_path']))
            else:
                error_msg = result.get('error', 'Unknown error occurred')
                print(f"\nAnalysis failed: {error_msg}")
                self.root.after(0, lambda: self.status_var.set(f"Analysis failed: {error_msg}"))
                self.root.after(0, lambda: messagebox.showerror("Error", f"Analysis failed: {error_msg}"))
            
            # Refresh results list
            self.root.after(0, self.update_runs_tree)
            
        except Exception as e:
            # Print error to command line
            print(f"\nOperation failed: {str(e)}")
            # Update status
            self.root.after(0, lambda: self.status_var.set(f"Operation failed: {str(e)}"))
            self.root.after(0, lambda: messagebox.showerror("Error", f"Operation failed: {str(e)}"))
        finally:
            # Reset buttons and progress
            self.root.after(0, lambda: self.stop_button.configure(state=tk.DISABLED))
            self.root.after(0, lambda: self.run_button.configure(state=tk.NORMAL))
            self.root.after(0, lambda: self.progress_var.set(0))
            self.root.after(0, lambda: self.progress_label.configure(text=""))
    
    def open_report_in_browser(self, report_path):
        """Open the report in the default web browser"""
        if not os.path.exists(report_path):
            messagebox.showerror("Error", f"Report not found: {report_path}")
            return
        
        # Open the report with more specific browser handling
        try:
            webbrowser.open(f'file://{os.path.abspath(report_path)}')
        except Exception as e:
            # Fallback to system open
            try:
                if sys.platform == 'win32':
                    os.startfile(os.path.abspath(report_path))
                elif sys.platform == 'darwin':
                    subprocess.call(['open', '-a', 'Safari', os.path.abspath(report_path)])
                else:
                    subprocess.call(['xdg-open', os.path.abspath(report_path)])
            except Exception as e2:
                messagebox.showerror("Error", f"Failed to open report: {str(e2)}")

    def open_latest_report(self):
        """Open the latest report"""
        if not self.run_history.get("runs"):
            messagebox.showinfo("Info", "No previous runs found.")
            return
        
        latest_run = self.run_history["runs"][0]
        report_dir = latest_run.get("report_dir")
        
        if not os.path.exists(report_dir):
            messagebox.showerror("Error", f"Report directory not found: {report_dir}")
            return
        
        # Find the HTML report
        html_files = [f for f in os.listdir(report_dir) if f.endswith(".html")]
        
        if not html_files:
            messagebox.showinfo("Info", "No HTML reports found in the latest run.")
            return
        
        # Open the first HTML report in the browser
        report_path = os.path.join(report_dir, html_files[0])
        self.open_report_in_browser(report_path)
    
    def update_runs_tree(self):
        # Clear existing
        for row in self.runs_tree.get_children():
            self.runs_tree.delete(row)
        # Add runs
        for run in self.run_history.get("runs", []):
            self.runs_tree.insert("", "end", values=(
                run.get("timestamp", ""),
                ", ".join(run.get("search_terms", [])[:2]),
                run.get("time_period", ""),
                run.get("max_results", "")
            ))
    
    def stop_scraping(self):
        """Stop the current scraping operation"""
        self.stop_requested = True
        self.status_var.set("Stopping... Please wait...")
        self.root.update()

    def get_available_models(self):
        """Get list of available models including the default model"""
        models = [("Default Model", "nlptown/bert-base-multilingual-uncased-sentiment")]
        
        if os.path.exists("models"):
            model_dirs = [d for d in os.listdir("models") if d.startswith("sentiment_model_")]
            for model_dir in sorted(model_dirs, reverse=True):
                # Get timestamp from directory name
                timestamp = model_dir.replace("sentiment_model_", "")
                try:
                    # Convert timestamp to readable date
                    date = datetime.strptime(timestamp, "%Y%m%d_%H%M%S")
                    readable_date = date.strftime("%Y-%m-%d %H:%M")
                    models.append((f"Trained Model ({readable_date})", os.path.join("models", model_dir)))
                except:
                    models.append((f"Trained Model ({timestamp})", os.path.join("models", model_dir)))
        
        return models

    def create_feedback_tab(self, parent):
        """Create the feedback management and model training tab"""
        frame = ttk.Frame(parent, padding="10")
        frame.pack(fill='both', expand=True)
        
        # Header
        ttk.Label(frame, text="Training Data Management", font=("TkDefaultFont", 12, "bold")).grid(column=0, row=0, columnspan=2, sticky=tk.W, pady=10)
        
        # Model selection frame
        model_frame = ttk.LabelFrame(frame, text="Select Sentiment Model", padding=10)
        model_frame.grid(column=0, row=1, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        
        # Model selector
        self.available_models = self.get_available_models()
        self.model_var = tk.StringVar()
        self.model_combo = ttk.Combobox(model_frame, textvariable=self.model_var, width=50)
        self.model_combo['values'] = [m[0] for m in self.available_models]
        self.model_combo.grid(column=0, row=0, sticky=(tk.W, tk.E), padx=5, pady=5)
        self.model_combo.bind('<<ComboboxSelected>>', self.on_model_selected)
        
        # Set initial model
        current_model = self.sentiment_config.get('topic_modeling', {}).get('sentiment_model', 'nlptown/bert-base-multilingual-uncased-sentiment')
        for i, (_, model_path) in enumerate(self.available_models):
            if model_path == current_model:
                self.model_combo.current(i)
                break
        
        # Train model button
        ttk.Button(model_frame, text="Train Model (Continue from Last)", command=self.train_model).grid(column=1, row=0, padx=5, pady=5)
        
        # Select run frame
        run_frame = ttk.LabelFrame(frame, text="Select Analysis Run", padding=10)
        run_frame.grid(column=0, row=2, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        
        # Combobox for selecting runs
        self.runs_combo = ttk.Combobox(run_frame, width=50)
        self.runs_combo.grid(column=0, row=0, sticky=(tk.W, tk.E), padx=5, pady=5)
        self.runs_combo.bind('<<ComboboxSelected>>', self.on_run_selected_for_training)
        
        # Refresh button
        ttk.Button(run_frame, text="Refresh Runs", command=self.refresh_runs_for_training).grid(column=1, row=0, padx=5, pady=5)
        
        # Chunks frame
        chunks_frame = ttk.LabelFrame(frame, text="Content Chunks", padding=10)
        chunks_frame.grid(column=0, row=3, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10)
        
        # Create a frame for the current chunk
        self.chunks_container = ttk.Frame(chunks_frame)
        self.chunks_container.pack(fill=tk.BOTH, expand=True)
        
        # Navigation frame
        nav_frame = ttk.Frame(chunks_frame)
        nav_frame.pack(fill=tk.X, pady=10)
        
        # Previous button
        self.prev_button = ttk.Button(nav_frame, text="Previous", command=self.show_previous_chunk, state=tk.DISABLED)
        self.prev_button.pack(side=tk.LEFT, padx=5)
        
        # Chunk counter
        self.chunk_counter_var = tk.StringVar(value="0/0")
        ttk.Label(nav_frame, textvariable=self.chunk_counter_var).pack(side=tk.LEFT, padx=20)
        
        # Next button
        self.next_button = ttk.Button(nav_frame, text="Next", command=self.show_next_chunk, state=tk.DISABLED)
        self.next_button.pack(side=tk.LEFT, padx=5)
        
        # Status label
        self.training_status_var = tk.StringVar(value="Select an analysis run to view chunks")
        ttk.Label(frame, textvariable=self.training_status_var).grid(column=0, row=4, sticky=tk.W, pady=5)
        
        # Save button
        self.save_button = ttk.Button(frame, text="Save Training Data", command=self.save_training_data, state=tk.DISABLED)
        self.save_button.grid(column=0, row=5, pady=10)
        
        # Configure grid expansion
        frame.columnconfigure(0, weight=1)
        chunks_frame.columnconfigure(0, weight=1)
        
        # Initialize
        self.refresh_runs_for_training()
        
        # Store chunks data
        self.current_chunks = []
        self.current_chunk_index = 0

    def show_current_chunk(self):
        """Display the current chunk in the UI"""
        if not self.current_chunks:
            return
            
        # Clear existing chunk display
        for widget in self.chunks_container.winfo_children():
            widget.destroy()
            
        chunk = self.current_chunks[self.current_chunk_index]
        
        # Article title
        ttk.Label(self.chunks_container, text=chunk['article_title'], font=("TkDefaultFont", 9, "bold")).pack(anchor=tk.W)
        
        # Chunk text (full text)
        text = chunk['chunk_text']
        ttk.Label(self.chunks_container, text=text, wraplength=800).pack(anchor=tk.W, pady=5)
        
        # Rating controls
        rating_frame = ttk.Frame(self.chunks_container)
        rating_frame.pack(anchor=tk.W)
        
        ttk.Label(rating_frame, text="Rating:").pack(side=tk.LEFT, padx=5)
        
        rating_var = tk.IntVar(value=chunk['rating'])
        
        # Function to update rating when changed
        def on_rating_change(*args):
            rating = rating_var.get()
            rating_text = ["very negative", "negative", "neutral", "positive", "very positive"][rating-1]
            # Update the chunk in current_chunks
            self.current_chunks[self.current_chunk_index]['rating'] = rating
            self.current_chunks[self.current_chunk_index]['rating_text'] = rating_text
        
        # Bind the rating change event
        rating_var.trace_add('write', on_rating_change)
        
        for i, label in enumerate(["Very Negative", "Negative", "Neutral", "Positive", "Very Positive"], 1):
            ttk.Radiobutton(rating_frame, text=label, variable=rating_var, value=i).pack(side=tk.LEFT, padx=5)
        
        # Store chunk data and rating var
        self.chunks_container.chunk_data = chunk
        self.chunks_container.rating_var = rating_var
        
        # Update navigation buttons
        self.prev_button.configure(state=tk.NORMAL if self.current_chunk_index > 0 else tk.DISABLED)
        self.next_button.configure(state=tk.NORMAL if self.current_chunk_index < len(self.current_chunks) - 1 else tk.DISABLED)
        
        # Update counter
        self.chunk_counter_var.set(f"{self.current_chunk_index + 1}/{len(self.current_chunks)}")

    def show_next_chunk(self):
        """Show the next chunk"""
        if self.current_chunk_index < len(self.current_chunks) - 1:
            self.current_chunk_index += 1
            self.show_current_chunk()

    def show_previous_chunk(self):
        """Show the previous chunk"""
        if self.current_chunk_index > 0:
            self.current_chunk_index -= 1
            self.show_current_chunk()

    def on_run_selected_for_training(self, event):
        """Handle selection of a run for training data"""
        if not hasattr(self, 'available_runs') or not self.available_runs:
            return
            
        selected_index = self.runs_combo.current()
        if selected_index < 0 or selected_index >= len(self.available_runs):
            return
            
        selected_run = self.available_runs[selected_index][1]
        report_dir = selected_run.get("report_dir")
        
        if not os.path.exists(report_dir):
            messagebox.showerror("Error", f"Report directory not found: {report_dir}")
            return
            
        # Look for training chunks file
        chunks_file = os.path.join(report_dir, "training_chunks.json")
        if not os.path.exists(chunks_file):
            messagebox.showinfo("Info", "No training chunks found in this run.")
            return
            
        # Load chunks
        try:
            with open(chunks_file, 'r', encoding='utf-8') as f:
                self.current_chunks = json.load(f)
                
            # Reset chunk index and show first chunk
            self.current_chunk_index = 0
            self.show_current_chunk()
            
            # Update status
            self.training_status_var.set(f"Loaded {len(self.current_chunks)} chunks from {selected_run['timestamp']}")
            
            # Enable save button
            self.save_button.configure(state=tk.NORMAL)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load chunks: {str(e)}")

    def save_training_data(self):
        """Save the rated chunks as training data"""
        try:
            # Collect all chunks with their ratings
            training_data = []
            for chunk in self.current_chunks:
                chunk_data = chunk.copy()
                # Get the current rating from the UI if this is the currently displayed chunk
                if (hasattr(self.chunks_container, 'chunk_data') and 
                    self.chunks_container.chunk_data.get('chunk_id') == chunk['chunk_id']):
                    rating = self.chunks_container.rating_var.get()
                    # Update rating and rating text
                    rating_text = ["very negative", "negative", "neutral", "positive", "very positive"][rating-1]
                    chunk_data['rating'] = rating
                    chunk_data['rating_text'] = rating_text
                training_data.append(chunk_data)
                
            if not training_data:
                messagebox.showwarning("No Data", "No chunks to save.")
                return
                
            # Create training data directory
            training_dir = "training_data"
            os.makedirs(training_dir, exist_ok=True)
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"training_data_{timestamp}.json"
            filepath = os.path.join(training_dir, filename)
            
            # Save the data
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(training_data, f, indent=2)
                
            messagebox.showinfo(
                "Success", 
                f"Saved {len(training_data)} training examples to:\n{filepath}"
            )
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save training data: {str(e)}")

    def refresh_runs_for_training(self):
        """Refresh the list of available runs for training data"""
        if not hasattr(self, 'run_history') or not self.run_history.get("runs"):
            self.runs_combo['values'] = ["No analysis runs found"]
            self.runs_combo.current(0)
            return
            
        # Get all runs with their timestamps and search terms
        runs = []
        for run in self.run_history["runs"]:
            timestamp = run.get("timestamp", "")
            search_terms = ", ".join(run.get("search_terms", [])[:2])
            display = f"{timestamp} - {search_terms}"
            runs.append((display, run))
            
        if runs:
            self.runs_combo['values'] = [r[0] for r in runs]
            self.runs_combo.current(0)
            self.available_runs = runs
        else:
            self.runs_combo['values'] = ["No analysis runs found"]
            self.runs_combo.current(0)
            self.available_runs = []

    def on_model_selected(self, event):
        """Handle model selection change"""
        selected_index = self.model_combo.current()
        if selected_index >= 0:
            _, model_path = self.available_models[selected_index]
            self.sentiment_config['topic_modeling']['sentiment_model'] = model_path
            self.save_config(self.sentiment_config, "sentiment_config.json")
            print(f"Selected model: {model_path}")

    def train_model(self):
        """Run the training script with continue option"""
        try:
            # Check if we have training data
            if not os.path.exists("training_data"):
                messagebox.showerror("Error", "No training data found. Please rate some chunks first.")
                return
                
            training_files = [f for f in os.listdir("training_data") if f.startswith("training_data_") and f.endswith(".json")]
            if not training_files:
                messagebox.showerror("Error", "No training data files found. Please rate some chunks first.")
                return
            
            # Run the training script
            self.status_var.set("Training model... (this may take a while)")
            self.root.update()
            
            # Run in a separate thread to keep GUI responsive
            threading.Thread(target=self._train_model_thread, daemon=True).start()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start training: {str(e)}")
    
    def _train_model_thread(self):
        """Thread function for training the model"""
        try:
            # Import the training module
            import train_sentiment_model
            
            # Run training with continue option
            model_dir = train_sentiment_model.train_model(continue_training=True)
            
            # Update the model list and select the new model
            self.root.after(0, self._update_after_training, model_dir)
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Training failed: {str(e)}"))
            self.root.after(0, lambda: self.status_var.set("Training failed"))
    
    def _update_after_training(self, model_dir):
        """Update the GUI after training completes"""
        try:
            # Refresh the model list
            self.available_models = self.get_available_models()
            self.model_combo['values'] = [m[0] for m in self.available_models]
            
            # Select the new model
            for i, (_, model_path) in enumerate(self.available_models):
                if model_path == model_dir:
                    self.model_combo.current(i)
                    self.on_model_selected(None)
                    break
            
            # Update status
            self.status_var.set("Training completed successfully!")
            messagebox.showinfo("Success", f"Model trained successfully!\nSaved to: {model_dir}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to update after training: {str(e)}")

def check_environment():
    """Check for missing dependencies and provide download instructions"""
    missing = check_imports()
    if missing:
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror(
            "Missing Dependencies", 
            f"The following Python modules are required but not installed:\n\n"
            f"{', '.join(missing)}\n\n"
            f"Please install them using:\n"
            f"pip install {' '.join(missing)}"
        )
        root.destroy()
        return False
    return True

def main():
    # Fix for PyInstaller multiprocessing issues
    multiprocessing.freeze_support()
    
    # Add debugging output
    print("Starting Sentiment Analysis GUI...")
    print(f"Python executable: {sys.executable}")
    print(f"Current working directory: {os.getcwd()}")
    
    if not check_environment():
        return
    
    # Create root window with explicit focus
    root = tk.Tk()
    
    # Ensure window appears on macOS
    if sys.platform == 'darwin':
        root.lift()
        root.attributes('-topmost', True)
        root.after_idle(root.attributes, '-topmost', False)
    
    print("Creating GUI application...")
    app = SentimentAnalysisGUI(root)
    
    print("Starting main loop...")
    root.mainloop()
    print("Application closed.")

if __name__ == "__main__":
    main()
