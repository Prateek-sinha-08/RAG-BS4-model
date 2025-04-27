import requests
from bs4 import BeautifulSoup, Comment

def scrape_all_text(url):
    # 1. Fetch
    resp = requests.get(url)
    resp.raise_for_status()

    # 2. Parse
    soup = BeautifulSoup(resp.text, "html.parser")

    # 3. Remove scripts, styles, comments
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
        comment.extract()

    # 4. Get visible text
    texts = soup.stripped_strings
    return "\n".join(texts)

def save_to_file(text, filename):
    # 5. Save the text to a .txt file
    with open(filename, 'w', encoding='utf-8') as file:
        file.write(text)

if __name__ == "__main__":
    url = "https://www.geeksforgeeks.org/my-career-journey-from-a-beginner-to-a-master-in-machine-learning-engineering/"  # Replace with the URL you want to scrape
    full_text = scrape_all_text(url)
    
    # Save the scraped text to a file
    save_to_file(full_text, 'documents/scraped_text.txt')
    print("Text has been saved to 'scraped_text.txt'")
