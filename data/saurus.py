import unicodedata


def save(file_name, content):
    with open(file_name, 'w') as text_file:
        text_file.write(content)


def is_utf(text):
    try:
        text.decode('utf-8')
        return True
    except UnicodeError:
        return False


def main():
    output = ''

    with open('openthesaurus.txt', mode='r') as f:
        for line in f:
            if line.startswith('#'):
                continue

            index = line.index(';')
            begin = line[:index]
            end = line[index + 1:].strip().replace(';', ' ')

            if begin != '' and end != '':
                text = '"%s","","%s"\n' % (begin.strip(), end.strip())

                output += text.decode('ascii', 'ignore')

    save('german.csv', output)


if __name__ == '__main__':
    main()
