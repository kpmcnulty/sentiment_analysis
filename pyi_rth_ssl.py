import os
import sys
import ssl

# Runtime hook for SSL certificates in PyInstaller bundle
def setup_ssl_certificates():
    """Set up SSL certificates for PyInstaller bundle"""
    if getattr(sys, 'frozen', False):
        # Running in PyInstaller bundle
        import certifi
        
        # Set SSL certificate environment variables
        os.environ['SSL_CERT_FILE'] = certifi.where()
        os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()
        
        # Also set for urllib
        ssl._create_default_https_context = ssl._create_unverified_context

# Apply the setup when this hook is loaded
setup_ssl_certificates()