from pdfminer.high_level import extract_text
import nltk
import re

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('stopwords')

PHONE_REG = re.compile(r'[\+\(]?[1-9][0-9 .\-\(\)]{8,}[0-9]')
EMAIL_REG = re.compile(r'[a-z0-9\.\-+_]+@[a-z0-9\.\-+_]+\.[a-z]+')
SKILLS_DB = [
    'python',
    'c++',
    'react',
    'reactjs',
    'java',
    'javascript',
    'word',
    'microsoft office',
    'english',
    'data science',
    'machine learning'
]


def extract_text_from_pdf(pdf_path):
    return extract_text(pdf_path)


def extract_name(resume_text):
    person_names = []
    for sent in nltk.sent_tokenize(resume_text):
        for chunk in nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(sent))):
            if hasattr(chunk, 'label') and chunk.label() == 'PERSON':
                person_names.append(
                    ' '.join(chunk_leave[0] for chunk_leave in chunk.leaves())
                )
    return person_names


def extract_phone_number(resume_text):
    phone = re.findall(PHONE_REG, resume_text)
    if phone:
        number = ''.join(phone[0])
        if resume_text.find(number) >= 0 and len(number) < 16:
            return number
    return None


def extract_emails(resume_text):
    return re.findall(EMAIL_REG, resume_text)


def extract_skills(resume_text):
    stop_words = set(nltk.corpus.stopwords.words('english'))
    word_tokens = nltk.tokenize.word_tokenize(resume_text)

    filtered_tokens = [w for w in word_tokens if w not in stop_words]
    filtered_tokens = [w for w in word_tokens if w.isalpha()]
    bigrams_trigrams = list(map(' '.join, nltk.everygrams(filtered_tokens, 2, 3)))
    found_skills = set()
    for token in filtered_tokens:
        if token.lower() in SKILLS_DB:
            found_skills.add(token)
    for ngram in bigrams_trigrams:
        if ngram.lower() in SKILLS_DB:
            found_skills.add(ngram)

    return found_skills


if __name__ == '__main__':
    resume = extract_text_from_pdf('./resume.pdf')
    names = extract_name(resume)
    phone_number = extract_phone_number(resume)
    emails = extract_emails(resume)
    skills = extract_skills(resume)

    if names:
        print('[NAME]', names[0], names[1])
        print('[PHONE]', phone_number)
        print('[EMAIL]', emails[0])
        print('[SKILLS]', skills)
