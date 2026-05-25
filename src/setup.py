import nltk

def setup():
    corpora = ['gutenberg', 'punkt', 'wordnet', 'words']
    for corpus in corpora:
        print(f"Downloading {corpus}...")
        nltk.download(corpus)

if __name__ == "__main__":
    setup()
