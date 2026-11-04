"""評価セットの「妥当性」を守るテスト（Issue #43）。

普通のテストは実装の正しさを守る。このテストが守るのは**評価セットが測定として成立していること**。
旧 eval_queries.json は FK 整合もトピック網羅も満たしていたが、
route はキーワード5語で100%当たり、正解は「answers を数えるだけ」で100%再現でき、
異常系は0件だった。つまり整合していても測定になっていなかった。

以下の4本が落ちたら、評価セットが「自分に甘いテスト」に戻っている。
設計の根拠は analysis/19_評価データ設計.md。
"""

import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import pytest

from eval_topic_keywords import TOPIC_KEYWORDS

REPO_ROOT = Path(__file__).resolve().parents[2]
SYN = REPO_ROOT / "fixtures" / "synthetic"
EVAL = SYN / "eval"

# 評価セットが無い環境ではスキップする（生成スクリプトは #89 で消し、company-v1 タグに残る）
pytestmark = pytest.mark.skipif(
    not (EVAL / "eval_person.json").exists(),
    reason="評価セットが無い。生成スクリプト build_eval_v2.py は company-v1 タグにある",
)


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _topics():
    return TOPIC_KEYWORDS


@pytest.fixture(scope="module")
def person():
    return _load(EVAL / "eval_person.json")


@pytest.fixture(scope="module")
def robustness():
    return _load(EVAL / "eval_robustness.json")


@pytest.fixture(scope="module")
def topics():
    return _topics()


# ---------------------------------------------------------------- 妥当性の4本


def test_no_topic_word_leak_in_hard_layers(person, topics):
    """L2/L3 のクエリに正解トピック名・そのキーワードが混入していないこと。

    混入していると BM25 だけで上位に来てしまい、埋め込みモデルの横並び比較で
    全モデルが飽和して差が出ない（旧セットは 24/40 = 60% が混入していた）。
    """
    leaked = []
    for q in person:
        if q["difficulty"] not in ("L2", "L3"):
            continue
        for t in q["gold_topics"]:
            hits = [w for w in [t, *topics[t]] if w in q["query"]]
            if hits:
                leaked.append((q["id"], t, hits))
    assert not leaked, f"L2/L3 にトピック語が漏れている: {leaked}"


def test_route_not_guessable_from_keywords(person):
    """表層キーワードが route の情報を持っていないこと。

    旧セットは「手順書/FAQ/マニュアル/過去/以前」の5語で 40/40 = 100% 当たった。
    それは経路判定の精度ではなくキーワード検出を測っている。

    判定は**多数クラス（常に person と答える）のベースラインとの差**で見る。
    絶対値で見ると、route 分布が person 偏重なだけで閾値を割ってしまい、
    「表層に情報があるか」を測れない。
    """
    kw = {
        "document": ["手順書", "FAQ", "マニュアル", "ドキュメント", "規程"],
        "prior_answer": ["過去", "以前", "流用", "前回"],
        "none": ["いますか", "探しています", "経験のある方"],
    }
    hit = 0
    for q in person:
        guess = "person"
        for route, words in kw.items():
            if any(w in q["query"] for w in words):
                guess = route
                break
        hit += guess == q["gold_route"]
    keyword_acc = hit / len(person)

    dist = Counter(q["gold_route"] for q in person)
    majority_acc = dist.most_common(1)[0][1] / len(person)

    assert keyword_acc <= majority_acc + 0.10, (
        f"キーワードだけで route が {keyword_acc:.0%}（多数クラス {majority_acc:.0%}）。"
        "表層の言い回しが route を漏らしている"
    )


def test_answers_only_baseline_is_weak(person, topics):
    """「過去回答を topic で数えるだけ」のベースラインの Recall@3 が 0.6 未満であること。

    旧セットではこのベースラインが正解と 100% 一致していた。つまり20行の実装が満点を取れた。
    これは専門性推定の良し悪しを一切測っていない。
    """
    answers = _load(SYN / "answers" / "answers.json")
    by_topic = defaultdict(Counter)
    for a in answers:
        by_topic[a["topic"]][a["responder_id"]] += 1
    global_top = [e for e, _ in Counter(a["responder_id"] for a in answers).most_common(3)]

    scores = []
    for q in person:
        if q["difficulty"] == "L4" or not q["gold_experts"]:
            continue
        hits = [
            t for t, kws in topics.items() if t in q["query"] or any(k in q["query"] for k in kws)
        ]
        if hits:
            c = Counter()
            for t in hits:
                c.update(by_topic[t])
            pred = [e for e, _ in c.most_common(3)]
        else:
            pred = global_top
        gold = set(q["gold_experts"])
        scores.append(len(set(pred) & gold) / min(3, len(gold)))

    recall = sum(scores) / len(scores)
    assert recall < 0.60, (
        f"answers を数えるだけで Recall@3={recall:.3f}。正解と入力が同じ源になっている"
    )


def test_enough_independent_samples(person):
    """独立サンプル数（ユニークな正解集合）が 40 以上であること。

    正解がトピック単位で固定だと、同じトピックのクエリは全て同じ正解になり、
    件数を増やしても統計的なサンプルは増えない（旧セットは40件で実質21件）。

    しきい値の変遷（**assert が実際に強制していた値**）: 25（#47）→ 30（#53）→ **40**（#54）。
    この docstring は #47 以来ずっと「20 以上」と書いていたが、初版の assert は
    最初から 25 だった。文章と assert が食い違っていたので、実際の値に直す。

    #51 の時点では 35 しか無く、「一次データで案件を持つのは4部署16名だけ」という
    上限が効いていた。その後 評価セットが 87件へ育ったことで、**一次データを
    触らないまま 50 に到達している**ので、DoD が求めていた 40 へ引き上げる。

    50 のうち空集合（gold_experts が空 = L3型番/L4棄却の21行）は**1種類だけ**。
    set で潰れるので何行あっても寄与は +1 で、残り 49 は実体のある組み合わせ。
    つまり余裕 10 は空集合による水増しではない。

    **上限そのものは動いていない。** `fixtures/source/case_history_dummy.json` の
    120件は今も4部署（営業/開発/カスタマーサポート/マーケティング）の **16名**しか
    リードを務めない。残り24名が推薦上位に来る根拠が日報（重み 0.15）だけ、という
    偏りは #54 の残り2条件として open のまま。
    """
    uniq = {tuple(sorted(q["gold_experts"])) for q in person}
    assert len(uniq) >= 40, f"独立サンプルが {len(uniq)} 種類しかない"


# ---------------------------------------------------------------- 構成の担保


def test_difficulty_layers_present(person):
    dist = Counter(q["difficulty"] for q in person)
    # #296: L3 に型番/製品名クエリ6件を追加（20→26・route=document）。
    assert dist == {"L1": 10, "L2": 36, "L3": 26, "L4": 15}, dist


def test_l4_expects_abstain(person):
    for q in person:
        if q["difficulty"] == "L4":
            assert not q["gold_experts"], f"L4 #{q['id']} に正解がある"
            assert q["expect_abstain"], f"L4 #{q['id']} が abstain でない"
            assert q["gold_route"] == "none"


def test_robustness_covers_all_categories(robustness):
    """「答えてはいけない/聞き返すべき」が5類型そろっていること。

    評価項目に「誤回答や想定外の入力への対策」が明文化されている（doc00 §2）。
    旧セットにはこの種の問題が1件も無かった。
    """
    assert len(robustness) == 20
    cats = Counter(r["category"] for r in robustness)
    assert set(cats) == {"out_of_scope", "pii", "insufficient", "no_expert", "adversarial"}, cats
    assert all(r["expect_abstain"] for r in robustness)


def test_foreign_keys_resolve(person):
    emp_ids = {e["id"] for e in _load(SYN / "people" / "employees.json")}
    unknown = sorted({e for q in person for e in q["gold_experts"] if e not in emp_ids})
    assert not unknown, f"未知の employee_id: {unknown}"


def test_queries_are_distinct(person):
    """クエリが実質的に重複していないこと（旧セットは6文型の穴埋めで38/40）。"""
    queries = [q["query"] for q in person]
    assert len(set(queries)) == len(queries)


def test_gold_source_present_only_on_data_routes(person):
    """#296: gold_source は自己回答できるデータ由来経路(document/prior_answer)にのみ付く。

    person(取次ぎ)・none(棄却)は自己回答の出典を持たない。自己回答経路は必ず出典を持つ
    （空だと source recall(#297) の分母が壊れる）。
    """
    for q in person:
        gs = q.get("gold_source", [])
        if q["gold_route"] in ("document", "prior_answer"):
            assert gs, f"id={q['id']} ({q['gold_route']}) に gold_source が無い"
        else:
            assert gs == [], f"id={q['id']} ({q['gold_route']}) に不要な gold_source: {gs}"


def test_product_docs_do_not_pollute_unrelated_gold_source(person):
    """#296: 製品文書(型番付き doc_031〜)は「その型番クエリの gold_source」以外に混入しない。

    トピック接頭辞ルール（title.startswith(topic)）は、製品カテゴリ名がトピック名を接頭辞に
    含むと無関係クエリの gold を汚す（例「セキュリティゲートウェイ」.startswith("セキュリティ")）。
    製品文書は gold_source_override で当該型番行にだけ単独で付くべきで、他行に現れてはいけない。
    """
    docs = _load(SYN / "documents" / "documents.json")
    product_ids = {d["id"] for d in docs if d.get("product_model")}
    assert product_ids, "製品文書(product_model 付き)が無い。型番eval の前提が崩れている"
    for q in person:
        gs = set(q.get("gold_source", []))
        overlap = gs & product_ids
        if overlap:
            # 製品文書を含むなら単独の gold_source（＝その型番クエリ本体）でなければならない。
            assert gs == overlap and len(gs) == 1, (
                f"id={q['id']} の gold_source に製品文書が混入/混合している: {sorted(gs)}"
            )


def test_gold_source_ids_resolve_in_corpus(person):
    """#296: gold_source の各IDが実在する文書/過去回答であること（出典リンク先の担保）。"""
    doc_ids = {d["id"] for d in _load(SYN / "documents" / "documents.json")}
    ans_ids = {a["id"] for a in _load(SYN / "answers" / "answers.json")}
    known = doc_ids | ans_ids
    unknown = sorted({s for q in person for s in q.get("gold_source", []) if s not in known})
    assert not unknown, f"未知の gold_source id: {unknown}"


def test_label_source_recorded(person):
    """どの正解が自動でどれが著述かを、後から説明できること。"""
    allowed = {"auto:project_daily", "authored", "human:pr46"}
    bad = {q["label_source"] for q in person} - allowed
    assert not bad, f"未知の label_source: {bad}"


def test_human_labeled_slice_present(person):
    """PR #46 の人手ラベル由来の項目が取り込まれていること。

    gold が全て自動導出だと「合成データの中の別ルール」でしかない。
    独立に人が付けたラベルを一部に入れておくことで、
    eval_label_agreement.py（company-v1 タグ）による外部検証が成立する。
    """
    human = [q for q in person if q["label_source"] == "human:pr46"]
    assert len(human) == 21, f"人手ラベル由来が {len(human)} 件"
    assert all(q["source_topic"] for q in human), "source_topic（PR #46 側のトピック名）が無い"
    assert all(q["gold_experts"] for q in human)
    # うち一定数は「自前22トピック体系に無い領域」であること（営業事務・庶務など）
    outside = [q for q in human if not q["gold_topics"]]
    assert len(outside) >= 4, "自前トピック体系の穴を埋める項目が足りない"


def test_retrieval_set_aligned_with_person(person):
    """層1（検索）が層2（推薦）と対応していること。

    人を外したとき、検索が悪いのかスコアリングが悪いのかを切り分けるために要る。
    """
    retrieval = _load(EVAL / "eval_retrieval.json")
    person_ids = {q["id"] for q in person if q["difficulty"] != "L4"}
    assert {r["id"] for r in retrieval} == person_ids
    assert all(r["gold_chunks"] for r in retrieval), "根拠チャンクが空の項目がある"
    assert not math.isnan(len(retrieval))


def test_alt_gold_is_an_independent_derivation(person):
    """第2の正解（`gold_experts_alt`）が、主 gold と別経路で作られていること（#73）。

    主 gold は `projects` + `daily_reports`、第2の正解は `answers` のみから作る。
    完全一致してしまうなら経路を分けた意味が無く、まったく重ならないならどちらかが壊れている。
    段B（トピックが分かった後の人の並び）の検証は、この2本の差でしか測れない。
    """
    scored = [q for q in person if q["difficulty"] != "L4" and q["gold_experts"]]
    assert scored, "採点対象が空"
    assert all("gold_experts_alt" in q for q in scored), "gold_experts_alt が無い項目がある"

    both = [q for q in scored if q["gold_experts_alt"]]
    assert len(both) >= 40, f"第2の正解を持つ項目が {len(both)} 件しかない"

    def jaccard(a, b):
        a, b = set(a), set(b)
        return len(a & b) / len(a | b)

    scores = [jaccard(q["gold_experts"], q["gold_experts_alt"]) for q in both]
    mean = sum(scores) / len(scores)
    assert 0.3 < mean < 0.95, f"主 gold との平均 Jaccard が {mean:.2f}（別経路として不適切）"


def test_abstention_layer_is_large_enough(person):
    """棄却の閾値を決めるには L4 が5件では足りなかった（#65 §6 の実測）。"""
    l4 = [q for q in person if q["difficulty"] == "L4"]
    assert len(l4) >= 15, f"L4 が {len(l4)} 件"
    assert all(q["gold_route"] == "none" and not q["gold_experts"] for q in l4)
