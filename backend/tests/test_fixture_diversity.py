"""合成データの多様性を守るテスト（#51 / #52）。

`fixtures/synthetic/` は評価の土台なので、構造が退化すると
**評価セットの側で何をやっても測定にならない**。以下はその退化を検出する。

- #51: トピック → 専門家が単一部署に閉じると、個人単位の専門性推定を評価できない
- #52: 全トピックが「回答6〜7件・文書1〜2件」と横並びだと、route をコーパスから決められない

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
    not (SYN / "projects" / "projects.json").exists(),
    reason="合成データが無い。生成スクリプト build_fixtures.py は company-v1 タグにある",
)


def _load(rel):
    return json.loads((SYN / rel).read_text(encoding="utf-8"))


def _topics():
    return TOPIC_KEYWORDS


@pytest.fixture(scope="module")
def topics():
    return _topics()


# ------------------------------------------------------------------ #51


def test_projects_cross_department(topics):
    """部署をまたぐメンバー構成の案件が一定割合あること。

    一次データは顧客接点のある4部署にしか案件を割り当てていない。
    支援部署（総務・経理・情シス等）をメンバーに入れないと、
    バックオフィスの社員は案件の証拠を一切持てず、トピック→専門家が部署に一意に決まる。
    """
    employees = {e["id"]: e for e in _load("people/employees.json")}
    members = defaultdict(list)
    for m in _load("projects/project_members.json"):
        members[m["project_id"]].append(m)

    cross = 0
    for ms in members.values():
        lead = next((m for m in ms if m["role"] == "lead"), None)
        if lead is None:
            continue
        lead_dept = employees[lead["employee_id"]]["department"]
        if any(
            employees[m["employee_id"]]["department"] != lead_dept
            for m in ms
            if m["role"] == "member"
        ):
            cross += 1
    rate = cross / len(members)
    assert rate >= 0.15, f"部署をまたぐ案件が {rate:.0%} しかない"


def test_topic_experts_are_not_all_identical(topics):
    """トピックごとの上位専門家が、トピック間で十分に散らばっていること。

    以前は開発部の4名が開発系6トピックすべてで**完全に同じ**正解集合だった。
    そうなると、クエリを何件足しても独立サンプルが増えない。
    """
    projects = {p["id"]: p for p in _load("projects/projects.json")}
    members = defaultdict(list)
    for m in _load("projects/project_members.json"):
        members[m["project_id"]].append(m)

    def match(text):
        return [t for t, kws in topics.items() if any(k in text for k in kws)]

    ev = defaultdict(lambda: defaultdict(float))
    for pid, p in projects.items():
        ts = match(f"{p['subject']} {p['client_issue']} {p['product']} {p.get('remarks', '')}")
        for m in members[pid]:
            w = 1.0 if m["role"] == "lead" else 0.6
            for t in ts:
                ev[m["employee_id"]][t] += w
    for d in _load("daily_reports/daily_reports.json"):
        for t in match(f"{d['content']} {d.get('issue', '')}"):
            ev[d["employee_id"]][t] += 0.15

    def top(t, k=3):
        r = sorted(((e, ev[e][t]) for e in ev if ev[e][t] > 0), key=lambda x: (-x[1], x[0]))
        return tuple(e for e, _ in r[:k])

    uniq = {top(t) for t in topics if top(t)}
    assert len(uniq) >= 18, f"トピック→上位3名 のユニーク集合が {len(uniq)} 種類しかない"


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


# ------------------------------------------------------------------ #83


def test_every_employee_branch_is_in_the_branch_vocabulary():
    """社員の拠点はすべて `BRANCH_VOCABULARY` に載っていること（#83）。

    `BRANCH_VOCABULARY` は `REGION_OF_BRANCH`（proximity 用に手で保守している地図）
    から導出している。DB に拠点を足して地図に足し忘れると、C1 はその拠点を
    **guided decoding で生成できなくなり**、利用者が明示した希望が黙って落ちる。
    `analyze` の警告はモデルが語彙外を出したときにしか出ないので、そこでは気づけない。
    """

    from tekijin.scorer.weights import BRANCH_VOCABULARY

    branches = {e.get("branch") for e in _load("people/employees.json")} - {None}
    unknown = branches - set(BRANCH_VOCABULARY)
    assert not unknown, f"REGION_OF_BRANCH に無い拠点: {sorted(unknown)}"
