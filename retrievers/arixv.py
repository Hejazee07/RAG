import arxiv

client = arxiv.Client(
    page_size=2,
    delay_seconds=3,
    num_retries=5,
)

search = arxiv.Search(
    query="all:Large Language Models",
    max_results=2,
    sort_by=arxiv.SortCriterion.Relevance,
)

for paper in client.results(search):
    print("=" * 80)
    print("Title:", paper.title)
    print("Published:", paper.published)
    print("Authors:", ", ".join(a.name for a in paper.authors))
    print("Summary:")
    print(paper.summary[:400])