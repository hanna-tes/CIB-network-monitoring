import re

def extract_image_urls(df):
    image_urls = []
    if 'text' in df.columns:
        for text in df['text'].dropna():
            urls = re.findall(r'(https?://\\S+)', text)
            for url in urls:
                if any(ext in url.lower() for ext in ['.jpg', '.jpeg', '.png', '.webp']):
                    image_urls.append(url)
    if 'media_url' in df.columns:
        image_urls.extend(df['media_url'].dropna().tolist())
    return list(set(image_urls))
