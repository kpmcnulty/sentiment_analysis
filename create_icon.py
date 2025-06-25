#!/usr/bin/env python3
"""
Create an app icon for the Sentiment Analysis Tool
"""

import os
import subprocess
from PIL import Image, ImageDraw, ImageFont
import tempfile

def create_icon_image(size=1024):
    """Create a professional-looking icon for the sentiment analysis app"""
    
    # Create a new image with a gradient background
    img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # Create a circular background with gradient effect
    center = size // 2
    radius = int(size * 0.45)
    
    # Draw multiple circles for gradient effect
    for i in range(radius, 0, -2):
        alpha = int(255 * (1 - i / radius) * 0.8)
        color = (52, 152, 219, alpha)  # Nice blue color
        draw.ellipse([center - i, center - i, center + i, center + i], 
                    fill=color, outline=None)
    
    # Add a subtle border
    border_radius = int(radius * 0.95)
    draw.ellipse([center - border_radius, center - border_radius, 
                 center + border_radius, center + border_radius], 
                outline=(41, 128, 185, 255), width=4)
    
    # Add sentiment symbols (happy, neutral, sad faces arranged in a triangle)
    symbol_size = size // 8
    symbol_offset = size // 6
    
    # Try to use a system font, fall back to default if not available
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", symbol_size)
    except:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", symbol_size)
        except:
            font = ImageFont.load_default()
    
    # Happy face (top)
    draw.text((center - symbol_size//2, center - symbol_offset), "😊", 
             font=font, fill=(46, 204, 113, 255), anchor="mm")
    
    # Neutral face (bottom left)
    draw.text((center - symbol_offset, center + symbol_offset//2), "😐", 
             font=font, fill=(241, 196, 15, 255), anchor="mm")
    
    # Sad face (bottom right)
    draw.text((center + symbol_offset, center + symbol_offset//2), "😞", 
             font=font, fill=(231, 76, 60, 255), anchor="mm")
    
    # Add a small chart/graph symbol in the center
    chart_size = size // 12
    chart_x = center - chart_size
    chart_y = center + chart_size//2
    
    # Draw simple bar chart
    bar_width = chart_size // 4
    for i, height in enumerate([0.3, 0.7, 0.5, 0.9]):
        x = chart_x + i * (bar_width + 2)
        bar_height = int(chart_size * height)
        draw.rectangle([x, chart_y - bar_height, x + bar_width, chart_y], 
                      fill=(255, 255, 255, 200))
    
    return img

def create_icns_file():
    """Create an .icns file for macOS"""
    
    # Create the base icon
    base_icon = create_icon_image(1024)
    
    # Create a temporary directory for icon files
    with tempfile.TemporaryDirectory() as temp_dir:
        iconset_dir = os.path.join(temp_dir, "AppIcon.iconset")
        os.makedirs(iconset_dir)
        
        # Define the required icon sizes for macOS
        sizes = [
            (16, "icon_16x16.png"),
            (32, "icon_16x16@2x.png"),
            (32, "icon_32x32.png"),
            (64, "icon_32x32@2x.png"),
            (128, "icon_128x128.png"),
            (256, "icon_128x128@2x.png"),
            (256, "icon_256x256.png"),
            (512, "icon_256x256@2x.png"),
            (512, "icon_512x512.png"),
            (1024, "icon_512x512@2x.png")
        ]
        
        # Generate all required sizes
        for size, filename in sizes:
            resized_icon = base_icon.resize((size, size), Image.Resampling.LANCZOS)
            resized_icon.save(os.path.join(iconset_dir, filename))
        
        # Convert to .icns using iconutil (macOS only)
        try:
            subprocess.run([
                "iconutil", "-c", "icns", iconset_dir, "-o", "app_icon.icns"
            ], check=True)
            print("✅ Created app_icon.icns successfully!")
            return True
        except subprocess.CalledProcessError:
            print("❌ Failed to create .icns file. iconutil not available.")
            return False
        except FileNotFoundError:
            print("❌ iconutil not found. This script requires macOS.")
            return False

def create_png_fallback():
    """Create a PNG fallback icon"""
    icon = create_icon_image(512)
    icon.save("app_icon.png")
    print("✅ Created app_icon.png as fallback")

if __name__ == "__main__":
    print("Creating app icon for Sentiment Analysis Tool...")
    
    # Try to create .icns file first
    if not create_icns_file():
        # Fall back to PNG
        create_png_fallback()
        print("💡 You can convert the PNG to .icns using online tools or:")
        print("   https://cloudconvert.com/png-to-icns") 