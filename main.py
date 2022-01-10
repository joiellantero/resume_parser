from pdfminer.high_level import extract_text
import nltk
import re
import os

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('stopwords')

PHONE_REG = re.compile(r'[\+\(]?[1-9][0-9 .\-\(\)]{8,}[0-9]')
EMAIL_REG = re.compile(r'[a-z0-9\.\-+_]+@[a-z0-9\.\-+_]+\.[a-z]+')


def convert_file_to_list(file):
    file = open(file)
    file = file.readlines()
    lines = []
    lines=[line.strip() for line in file]
    return lines


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
    skills_db = convert_file_to_list("./db/skills_db.txt")
    for token in filtered_tokens:
        if token.lower() in skills_db:
            found_skills.add(token)
    for ngram in bigrams_trigrams:
        if ngram.lower() in skills_db:
            found_skills.add(ngram)
    return found_skills


def extract_education(resume_text):
    organizations = []
    for sent in nltk.sent_tokenize(resume_text):
        for chunk in nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(sent))):
            if hasattr(chunk, 'label') and chunk.label() == 'ORGANIZATION':
                organizations.append(' '.join(c[0] for c in chunk.leaves()))
    education = set()
    education_db = convert_file_to_list("./db/education_db.txt")
    for org in organizations:
        for word in education_db:
            if org.lower().find(word) >= 0:
                education.add(org)
    return education


if __name__ == '__main__':
    filename = "resume"
    filepath = f'test-data/{filename}.pdf'

    try: 
        if os.stat(filepath).st_size > 0:
            resume = extract_text_from_pdf(filepath)
            names = extract_name(resume)
            phone_number = extract_phone_number(resume)
            emails = extract_emails(resume)
            skills = extract_skills(resume)
            education = extract_education(resume)

            if names:
                print('[NAME]', names[0])

            if phone_number:
                print('[PHONE]', phone_number)

            if emails:
                print('[EMAIL]', emails[0])

            if skills:
                print('[SKILLS]', skills)

            if education:
                print('[EDUCATION]', education)
        else:
            print("[ERROR] Empty file")
    except OSError:
         print("[ERROR] Cannot find file")

