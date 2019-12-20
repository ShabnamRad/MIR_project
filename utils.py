import xml.etree.ElementTree as ET
import csv


def parse_xml(xml_file):
    document = ET.parse(xml_file)

    NSMAP = {'mw': 'http://www.mediawiki.org/xml/export-0.10/'}

    docs = []
    index = 0

    for item in document.findall('.//mw:page', namespaces=NSMAP):

        doc = {'id': index, 'title': item.findall('.//mw:title', namespaces=NSMAP)[0].text,
               'text': item.findall('.//mw:revision/mw:text', namespaces=NSMAP)[0].text}
        docs.append(doc)
        index += 1

    return docs


def parse_csv(csv_file):
    with open(csv_file, encoding="utf8") as file:
        data = list(csv.reader(file))

    data = data[1:]
    docs = []
    index = 0
    for row in data:
        docs.append({'id': index, 'title': row[0], 'text': row[1]})
        index += 1

    return docs


def parse_tagged_csv(csv_file):
    with open(csv_file, encoding="utf8") as file:
        data = list(csv.reader(file))

    data = data[1:]
    tags = []
    docs = []
    index = 0
    for row in data:
        tags.append(row[0])
        docs.append({'id': index, 'title': row[1], 'text': row[2]})
        index += 1

    return tags, docs
