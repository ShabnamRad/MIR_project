import numpy as np
import json


def add_page_rank(alpha=0.85):
    with open('crawler/crawled_papers_info.json', mode='r', encoding='utf-8') as f:
        result = json.load(f)
    g = dict()
    titles = dict()
    for paper_info in result:
        g[paper_info['id']] = paper_info['references']
        titles[paper_info['id']] = paper_info['title']

    node_id = dict()
    for k in g.keys():
        node_id[k] = len(node_id)

    # create P
    n = len(node_id)
    p_matrix = np.zeros(shape=(n, n))
    for v, ref_ids in g.items():
        v_id = node_id[v]
        cnt = 0
        for ref_id in ref_ids:
            if ref_id in node_id:
                p_matrix[v_id, node_id[ref_id]] = 1
                cnt += 1
        p_matrix[v_id] = p_matrix[v_id] / cnt if cnt > 0 else 1.0 / n

    # calculate page rank
    p_matrix = (1.0 - alpha) * p_matrix + alpha / n
    a = np.zeros(shape=(1, n))
    a[0, 0] = 1
    for i in range(500):
        b = np.matmul(a, p_matrix)
        if np.abs(b - a).max() < 1e-9:
            print("converged at %d" % i)
            break
        a = b

    page_rank_data = list(zip(titles.values(), list(node_id.keys()), a[0].tolist()))
    i = 0
    with open('page_ranks.txt', mode='w', encoding='utf-8') as f:
        f.write("")
    for (title, pid, pr) in page_rank_data:
        i += 1
        print(str(i) + ") PR: " + str(pr) + ", Title: " + title + ", id: " + pid)
        with open('page_ranks.txt', mode='a+', encoding='utf-8') as f:
            f.write(str(i) + ") PR: " + str(pr) + ", Title: " + title + ", id: " + pid + "\n")


add_page_rank()
