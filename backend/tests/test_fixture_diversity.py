"""合成データの多様性を守るテスト（#51 / #52）。

`fixtures/synthetic/` は評価の土台なので、構造が退化すると
**評価セットの側で何をやっても測定にならない**。以下はその退化を検出する。

- #52: 全トピックが「回答6〜7件・文書1〜2件」と横並びだと、route をコーパスから決められない

#51 の部署横断の検査は案件と日報の合成データを読んでいたので、それらと一緒に #97 で消した。

背景は `fixtures/synthetic/README.md` と `analysis/19_評価データ設計.md`。
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

import pytest

from eval_topic_keywords import TOPIC_KEYWORDS

REPO_ROOT = Path(__file__).resolve().parents[2]
SYN = REPO_ROOT / "fixtures" / "synthetic"

pytestmark = pytest.mark.skipif(
    not (SYN / "answers" / "answers.json").exists(),
    reason="合成データが無い。生成スクリプト build_fixtures.py は company-v1 タグにある",
)


def _load(rel):
    return json.loads((SYN / rel).read_text(encoding="utf-8"))


def _topics():
    return TOPIC_KEYWORDS


@pytest.fixture(scope="module")
def topics():
    return _topics()


# ------------------------------------------------------------------ #52


def test_documents_are_concentrated(topics):
    """文書が一部のトピックに集中していること（route=document を成立させるため）。

    全トピックに一律で文書があると「文書があるトピックは document」が全件該当してしまい、
    経路判定精度を測れない。
    """
    documents = _load("documents/documents.json")
    with_docs = {t for d in documents for t in topics if d["title"].startswith(t)}
    assert len(with_docs) <= 8, f"文書を持つトピックが {len(with_docs)} 個ある（8以下にする）"
    assert with_docs, "文書を持つトピックが1つも無い"


def test_reuse_count_differs_by_topic():
    """トピック別の「有用回答の平均 reuse_count」に差があること。

    一様だと route=prior_answer をコーパスから判定できない。
    """
    answers = _load("answers/answers.json")
    reuse = defaultdict(list)
    for a in answers:
        if a.get("was_helpful"):
            reuse[a["topic"]].append(a.get("reuse_count", 0))
    means = {t: sum(v) / len(v) for t, v in reuse.items() if v}
    assert means, "有用回答が1件も無い"
    spread = max(means.values()) - min(means.values())
    assert spread >= 3.0, f"平均 reuse_count の差が {spread:.2f} しかない"


def test_some_topics_have_no_past_answers(topics):
    """過去QAを1件も持たないトピックがあること（route=person の純粋なケース）。"""
    answers = _load("answers/answers.json")
    n = Counter(a["topic"] for a in answers)
    empty = [t for t in topics if n.get(t, 0) == 0]
    assert empty, "全トピックに過去QAがある。現場判断のみの領域が無い"


def test_route_labels_come_from_corpus():
    """評価セットの route が4種類そろい、person に偏りすぎていないこと。

    評価セットの生成スクリプト（build_eval_v2.py、company-v1 タグ）は
    route をコーパスの状態（文書数・reuse）から導出する。
    fixtures が横並びに戻ると、ここが1種類に潰れる。
    """
    person = json.loads((SYN / "eval" / "eval_person.json").read_text(encoding="utf-8"))
    dist = Counter(q["gold_route"] for q in person)
    assert set(dist) == {"person", "prior_answer", "document", "none"}, dist
    assert dist["person"] / len(person) <= 0.80, f"person に偏りすぎ: {dist}"
