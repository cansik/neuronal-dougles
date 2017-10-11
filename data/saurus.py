# coding=utf-8
import unicodedata

umlauts = {u'Ä': 'Ae',
           u'Ö': 'Oe',
           u'Ü': 'Ue',
           u'ä': 'ae',
           u'ö': 'oe',
           u'ü': 'ue'
           }


def save(file_name, content):
    with open(file_name, 'w') as text_file:
        text_file.write(content.encode('utf-8'))


def is_utf(text):
    try:
        text.decode('utf-8')
        return True
    except UnicodeError:
        return False


def prepare_umlaut(text):
    for umlaut, representation in umlauts.items():
        text = text.replace(umlaut, representation)
    return text


def main():
    output = ''

    with open('openthesaurus.txt', mode='r') as f:
        for l in f:
            line = l.decode('utf-8')

            if line.startswith('#'):
                continue

            index = line.index(';')
            begin = line[:index]
            end = line[index + 1:].strip().replace(';', ' ')

            if begin != '' and end != '':
                text = '"%s","","%s"\n' % (begin.strip(), end.strip())
                output += prepare_umlaut(text)

    save('german.csv', output)


if __name__ == '__main__':
    main()
