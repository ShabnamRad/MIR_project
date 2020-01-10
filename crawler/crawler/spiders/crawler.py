import scrapy
import json


class SemanticScholarSpider(scrapy.Spider):
    name = "semantic_scholar"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.to_crawl = 5000
        self.visited_ids = set()
        self.DATA_FILENAME = "crawled_papers_info.json"
        with open(self.DATA_FILENAME, mode='w', encoding='utf-8') as f:
            json.dump([], f)

    @staticmethod
    def get_url(paper_id):
        return 'https://www.semanticscholar.org/paper/%s' % paper_id

    def start_requests(self):
        start_ids = [
            'Tackling-Climate-Change-with-Machine-Learning-Rolnick-Donti/998039a4876edc440e0cabb0bc42239b0eb29644',
            'Sublinear-Algorithms-for-(%CE%94%2B-1)-Vertex-Coloring-Assadi-Chen/eb4e84b8a65a21efa904b6c30ed9555278077dd3',
            'Processing-Data-Where-It-Makes-Sense%3A-Enabling-Mutlu-Ghose/4f17bd15a6f86730ac2207167ccf36ec9e6c2391',
        ]
        urls = [str(self.get_url(paper_id)) for paper_id in start_ids]
        for paper_id in start_ids:
            self.visited_ids.add(paper_id)

        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):
        page = '/'.join(response.url.split("/")[-2:])

        references = response.css('div[id="references"] a[data-selenium-selector="title-link"]::attr(href)') \
            .re(r'paper/(.*)')

        data = {
            "id": page,
            "title": response.css('h1[data-selenium-selector="paper-detail-title"]::text').extract_first(),
            "abstract": response.css('meta[property="og:description"]::attr(content)').extract_first(),
            "date": response.css('meta[name="citation_publication_date"]::attr(content)').extract_first(),
            "authors": response.css('meta[name="citation_author"]::attr(content)').extract(),
            "references": references
        }
        with open(self.DATA_FILENAME, mode='r', encoding='utf-8') as f:
            old = json.load(f)
        with open(self.DATA_FILENAME, mode='w', encoding='utf-8') as f:
            old.append(data)
            json.dump(old, f)
        yield data

        for new_paper_id in references[:5]:
            if new_paper_id not in self.visited_ids:
                self.visited_ids.add(new_paper_id)
                if self.to_crawl > 0:
                    self.to_crawl -= 1
                    yield response.follow('/paper/' + new_paper_id, callback=self.parse)
