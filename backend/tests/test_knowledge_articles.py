"""The KCS knowledge store: drafts, review, organization-scoped search (#112, R6).

``knowledge_articles`` holds one piece of knowledge in the KCS shape (issue,
environment, resolution, cause) with a state (wip, draft, verified, retired), and
``knowledge_sources`` ties it to what it was made from. Only verified knowledge is
searched, and only within the asker's organization and what the asker may read.
"""

from __future__ import annotations

import datetime as dt

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from tekijin.data.db import get_sessionmaker
from tekijin.data.knowledge import (
    ArticleNotFound,
    ArticleStateError,
    SourceRef,
    article_sources,
    create_draft,
    get_article,
    list_articles,
    matching_drafts,
    record_demand,
    record_use,
    retire_article,
    search_verified,
    send_back_article,
    update_article,
    verify_article,
)
from tekijin.data.visibility import EVERYTHING, PUBLIC, SourceScope
from tekijin.knowledge.index import article_text, embed_articles
from tekijin.models.tables import (
    ChannelMessage,
    DiscordChannel,
    Employee,
    KnowledgeArticle,
    Organization,
    SourceItem,
)

NOW = dt.datetime(2026, 11, 5, 12, 0, 0)
PUBLIC_CHANNEL = 7001
PRIVATE_CHANNEL = 7002


@pytest.fixture
def world(engine):
    factory = get_sessionmaker(engine)
    with factory() as session:
        org = Organization(name="天文サークル", discord_guild_id="881")
        other = Organization(name="ほかの会", discord_guild_id="882")
        session.add_all([org, other])
        session.flush()
        alice = Employee(name="アリス", org_id=org.id)
        bob = Employee(name="ボブ", org_id=other.id)
        session.add_all([alice, bob])
        session.flush()
        manual = SourceItem(
            org_id=org.id,
            provider="manual",
            external_id="manual:1",
            title="合宿の手引き",
            url="https://ex.com/camp",
            body="鍵は部室の箱",
            author_id=alice.id,
            content_hash="h1",
            updated_at=NOW,
        )
        foreign = SourceItem(
            org_id=other.id,
            provider="manual",
            external_id="manual:2",
            title="よそ",
            body="x",
            content_hash="h2",
            updated_at=NOW,
        )
        session.add_all([manual, foreign])
        session.add_all(
            [
                DiscordChannel(
                    org_id=org.id, channel_id=PUBLIC_CHANNEL, public=True, updated_at=NOW
                ),
                DiscordChannel(
                    org_id=org.id, channel_id=PRIVATE_CHANNEL, public=False, updated_at=NOW
                ),
            ]
        )
        said = ChannelMessage(
            org_id=org.id,
            message_id=9001,
            channel_id=PUBLIC_CHANNEL,
            channel_name="質問",
            author_id=alice.id,
            content="望遠鏡は倉庫の右の棚です",
            created_at=NOW,
        )
        secret = ChannelMessage(
            org_id=org.id,
            message_id=9002,
            channel_id=PRIVATE_CHANNEL,
            channel_name="運営",
            author_id=alice.id,
            content="会計の暗証番号は会計係に聞く",
            created_at=NOW,
        )
        session.add_all([said, secret])
        session.commit()
        ids = {
            "org": org.id,
            "other": other.id,
            "alice": alice.id,
            "bob": bob.id,
            "manual": manual.id,
            "foreign": foreign.id,
            "said": said.id,
            "secret": secret.id,
        }
    yield ids
    with factory() as session:
        for table in (
            "knowledge_sources",
            "knowledge_articles",
            "channel_messages",
            "discord_channels",
            "source_items",
        ):
            session.execute(text(f"DELETE FROM {table}"))  # noqa: S608 - fixed names
        session.execute(text("DELETE FROM employees WHERE org_id IS NOT NULL"))
        session.execute(text("DELETE FROM organizations"))
        session.commit()


@pytest.fixture
def db(engine, world):
    # Depends on ``world`` so its rollback runs before ``world`` deletes the rows.
    factory = get_sessionmaker(engine)
    with factory() as session:
        yield session
        session.rollback()


def _verified(db, world, fake_embedder, issue, resolution, *, sources=(), org=None) -> int:
    article = create_draft(
        db,
        org_id=org or world["org"],
        kind="procedure",
        issue=issue,
        resolution=resolution,
        sources=sources,
        now=NOW,
    )
    verify_article(db, article.id, org_id=org or world["org"], verifier_id=None, now=NOW)
    embed_articles(db, fake_embedder, article_ids=[article.id])
    return article.id


# --------------------------------------------------------------------------- #
# creating drafts
# --------------------------------------------------------------------------- #
def test_a_draft_with_a_resolution_starts_as_draft(db, world) -> None:
    article = create_draft(
        db,
        org_id=world["org"],
        kind="case",
        issue="部室の鍵が見つからない",
        environment="合宿の前",
        resolution="部室の箱に入っている",
        cause="前の担当が戻し忘れた",
        author_id=world["alice"],
        sources=[SourceRef(source_item_id=world["manual"], quote="鍵は部室の箱")],
        now=NOW,
    )
    assert article.state == "draft"
    assert (article.kind, article.issue, article.environment) == (
        "case",
        "部室の鍵が見つからない",
        "合宿の前",
    )
    assert article.cause == "前の担当が戻し忘れた"
    assert article.author_id == world["alice"] and article.verifier_id is None
    assert article.reuse_count == 0 and article.last_used_at is None
    assert article.created_at == NOW and article.updated_at == NOW


def test_a_draft_without_a_resolution_is_wip(db, world) -> None:
    for blank in (None, "", "   "):
        article = create_draft(
            db,
            org_id=world["org"],
            kind="case",
            issue="観測会の雨天時は？",
            resolution=blank,
            now=NOW,
        )
        assert article.state == "wip"
        assert article.resolution is None


def test_one_source_can_back_several_articles(db, world) -> None:
    ref = SourceRef(source_item_id=world["manual"], quote="鍵は部室の箱")
    first = create_draft(
        db,
        org_id=world["org"],
        kind="procedure",
        issue="鍵",
        resolution="箱",
        sources=[ref],
        now=NOW,
    )
    second = create_draft(
        db, org_id=world["org"], kind="fact", issue="箱", resolution="部室", sources=[ref], now=NOW
    )
    found = article_sources(db, [first.id, second.id], org_id=world["org"], scope=EVERYTHING)
    assert [s.source_item_id for s in found[first.id]] == [world["manual"]]
    assert [s.source_item_id for s in found[second.id]] == [world["manual"]]
    assert found[first.id][0].quote == "鍵は部室の箱"
    assert found[first.id][0].title == "合宿の手引き"
    assert found[first.id][0].url == "https://ex.com/camp"


def test_an_article_can_have_several_sources(db, world) -> None:
    article = create_draft(
        db,
        org_id=world["org"],
        kind="case",
        issue="望遠鏡の置き場所",
        resolution="倉庫の右の棚",
        sources=[
            SourceRef(source_item_id=world["manual"]),
            SourceRef(channel_message_id=world["said"], quote="倉庫の右の棚"),
        ],
        now=NOW,
    )
    found = article_sources(db, [article.id], org_id=world["org"], scope=EVERYTHING)[article.id]
    assert {(s.source_item_id, s.channel_message_id) for s in found} == {
        (world["manual"], None),
        (None, world["said"]),
    }
    message = next(s for s in found if s.channel_message_id is not None)
    assert message.channel_name == "質問"
    assert message.url == f"https://discord.com/channels/881/{PUBLIC_CHANNEL}/9001"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"kind": "rumor"},
        {"issue": "   "},
    ],
)
def test_bad_values_are_refused(db, world, kwargs) -> None:
    args = {
        "org_id": world["org"],
        "kind": "case",
        "issue": "問い",
        "resolution": "答え",
        "now": NOW,
    }
    with pytest.raises(ValueError):
        create_draft(db, **{**args, **kwargs})


def test_a_source_ref_names_exactly_one_thing() -> None:
    with pytest.raises(ValueError):
        SourceRef()
    with pytest.raises(ValueError):
        SourceRef(source_item_id=1, channel_message_id=2)


def test_another_organizations_source_is_refused(db, world) -> None:
    with pytest.raises(ValueError):
        create_draft(
            db,
            org_id=world["org"],
            kind="fact",
            issue="よその話",
            resolution="x",
            sources=[SourceRef(source_item_id=world["foreign"])],
            now=NOW,
        )


def test_the_table_refuses_a_non_wip_article_without_a_resolution(db, world) -> None:
    db.add(
        KnowledgeArticle(
            org_id=world["org"],
            kind="fact",
            issue="問い",
            resolution=None,
            state="draft",
            created_at=NOW,
            updated_at=NOW,
        )
    )
    with pytest.raises(IntegrityError):
        db.flush()


# --------------------------------------------------------------------------- #
# review: edit, verify, send back, retire
# --------------------------------------------------------------------------- #
def test_verify_stamps_the_verifier_and_time(db, world) -> None:
    article = create_draft(
        db, org_id=world["org"], kind="fact", issue="会費", resolution="月500円", now=NOW
    )
    later = NOW + dt.timedelta(hours=1)
    done = verify_article(
        db, article.id, org_id=world["org"], verifier_id=world["alice"], now=later
    )
    assert done.state == "verified"
    assert done.verifier_id == world["alice"] and done.verified_at == later


def test_wip_cannot_be_verified(db, world) -> None:
    article = create_draft(db, org_id=world["org"], kind="fact", issue="会費", now=NOW)
    with pytest.raises(ArticleStateError):
        verify_article(db, article.id, org_id=world["org"], verifier_id=None, now=NOW)


def test_editing_in_an_answer_turns_wip_into_draft(db, world) -> None:
    article = create_draft(db, org_id=world["org"], kind="fact", issue="会費", now=NOW)
    later = NOW + dt.timedelta(minutes=5)
    edited = update_article(
        db, article.id, org_id=world["org"], changes={"resolution": "月500円"}, now=later
    )
    assert edited.state == "draft" and edited.resolution == "月500円"
    assert edited.updated_at == later


def test_clearing_the_answer_turns_it_back_into_wip(db, world) -> None:
    article = create_draft(
        db, org_id=world["org"], kind="fact", issue="会費", resolution="月500円", now=NOW
    )
    verify_article(db, article.id, org_id=world["org"], verifier_id=None, now=NOW)
    edited = update_article(
        db, article.id, org_id=world["org"], changes={"resolution": " "}, now=NOW
    )
    assert edited.state == "wip" and edited.resolution is None and edited.verified_at is None


def test_editing_clears_the_embedding_so_it_is_rebuilt(db, world, fake_embedder) -> None:
    article_id = _verified(db, world, fake_embedder, "会費", "月500円")
    assert db.get(KnowledgeArticle, article_id).embedding is not None
    update_article(db, article_id, org_id=world["org"], changes={"issue": "会費はいくら"}, now=NOW)
    assert db.get(KnowledgeArticle, article_id).embedding is None


def test_unknown_or_bad_edits_are_refused(db, world) -> None:
    article = create_draft(
        db, org_id=world["org"], kind="fact", issue="会費", resolution="月500円", now=NOW
    )
    with pytest.raises(ValueError):
        update_article(db, article.id, org_id=world["org"], changes={"state": "verified"}, now=NOW)
    with pytest.raises(ValueError):
        update_article(db, article.id, org_id=world["org"], changes={"kind": "rumor"}, now=NOW)
    with pytest.raises(ValueError):
        update_article(db, article.id, org_id=world["org"], changes={"issue": ""}, now=NOW)


def test_send_back_returns_it_to_wip(db, world) -> None:
    article = create_draft(
        db, org_id=world["org"], kind="fact", issue="会費", resolution="月500円", now=NOW
    )
    verify_article(db, article.id, org_id=world["org"], verifier_id=world["alice"], now=NOW)
    back = send_back_article(db, article.id, org_id=world["org"], now=NOW)
    assert back.state == "wip" and back.verifier_id is None and back.verified_at is None
    assert back.resolution == "月500円"  # kept, so the next edit can build on it


def test_a_sent_back_article_with_an_answer_can_be_verified_again(db, world) -> None:
    article = create_draft(
        db, org_id=world["org"], kind="fact", issue="会費", resolution="月500円", now=NOW
    )
    send_back_article(db, article.id, org_id=world["org"], now=NOW)
    again = verify_article(db, article.id, org_id=world["org"], verifier_id=None, now=NOW)
    assert again.state == "verified"


def test_an_unanswered_article_can_be_retired(db, world) -> None:
    article = create_draft(db, org_id=world["org"], kind="fact", issue="雨天時は？", now=NOW)
    retired = retire_article(db, article.id, org_id=world["org"], now=NOW)
    db.flush()
    assert retired.state == "retired" and retired.resolution is None
    # Editing it keeps it retired.
    kept = update_article(db, article.id, org_id=world["org"], changes={"cause": "x"}, now=NOW)
    assert kept.state == "retired"


def test_retire_can_point_at_the_replacement(db, world) -> None:
    old = create_draft(
        db, org_id=world["org"], kind="fact", issue="会費", resolution="月400円", now=NOW
    )
    new = create_draft(
        db, org_id=world["org"], kind="fact", issue="会費", resolution="月500円", now=NOW
    )
    retired = retire_article(db, old.id, org_id=world["org"], superseded_by=new.id, now=NOW)
    assert retired.state == "retired" and retired.superseded_by == new.id
    with pytest.raises(ValueError):
        retire_article(db, new.id, org_id=world["org"], superseded_by=new.id, now=NOW)


def test_a_retired_article_can_be_verified_again(db, world) -> None:
    article = create_draft(
        db, org_id=world["org"], kind="fact", issue="会費", resolution="月500円", now=NOW
    )
    retire_article(db, article.id, org_id=world["org"], now=NOW)
    assert (
        verify_article(db, article.id, org_id=world["org"], verifier_id=None, now=NOW).state
        == "verified"
    )


def test_review_actions_stay_inside_the_organization(db, world) -> None:
    article = create_draft(
        db, org_id=world["org"], kind="fact", issue="会費", resolution="月500円", now=NOW
    )
    for action in (
        lambda: verify_article(db, article.id, org_id=world["other"], verifier_id=None, now=NOW),
        lambda: send_back_article(db, article.id, org_id=world["other"], now=NOW),
        lambda: retire_article(db, article.id, org_id=world["other"], now=NOW),
        lambda: update_article(
            db, article.id, org_id=world["other"], changes={"issue": "x"}, now=NOW
        ),
    ):
        with pytest.raises(ArticleNotFound):
            action()


# --------------------------------------------------------------------------- #
# reading and listing
# --------------------------------------------------------------------------- #
def test_members_list_verified_only(db, world) -> None:
    draft = create_draft(
        db, org_id=world["org"], kind="fact", issue="会費", resolution="月500円", now=NOW
    )
    shown = create_draft(
        db, org_id=world["org"], kind="fact", issue="部費", resolution="年1000円", now=NOW
    )
    verify_article(db, shown.id, org_id=world["org"], verifier_id=None, now=NOW)
    items, total = list_articles(db, org_id=world["org"], states=("verified",), scope=PUBLIC)
    assert [a.id for a in items] == [shown.id] and total == 1
    assert (
        get_article(db, draft.id, org_id=world["org"], scope=PUBLIC, states=("verified",)) is None
    )


def test_another_organization_sees_nothing(db, world) -> None:
    article = create_draft(
        db, org_id=world["org"], kind="fact", issue="会費", resolution="月500円", now=NOW
    )
    verify_article(db, article.id, org_id=world["org"], verifier_id=None, now=NOW)
    assert list_articles(db, org_id=world["other"], states=("verified",), scope=EVERYTHING) == (
        [],
        0,
    )
    assert get_article(db, article.id, org_id=world["other"], scope=EVERYTHING) is None


def test_review_queue_puts_the_most_wanted_draft_first(db, world) -> None:
    quiet = create_draft(
        db, org_id=world["org"], kind="fact", issue="会費", resolution="月500円", now=NOW
    )
    wanted = create_draft(
        db,
        org_id=world["org"],
        kind="fact",
        issue="部費",
        resolution="年1000円",
        now=NOW - dt.timedelta(days=1),
    )
    wip = create_draft(db, org_id=world["org"], kind="fact", issue="雨天時", now=NOW)
    record_demand(db, [wanted.id])
    record_demand(db, [wanted.id])
    items, total = list_articles(db, org_id=world["org"], states=("wip", "draft"), scope=EVERYTHING)
    assert total == 3
    assert [a.id for a in items][0] == wanted.id
    assert {a.id for a in items} == {quiet.id, wanted.id, wip.id}
    assert items[0].demand_count == 2


def test_listing_searches_the_text(db, world) -> None:
    for issue in ("会費はいくら", "望遠鏡の場所"):
        a = create_draft(
            db, org_id=world["org"], kind="fact", issue=issue, resolution="答え", now=NOW
        )
        verify_article(db, a.id, org_id=world["org"], verifier_id=None, now=NOW)
    items, total = list_articles(
        db, org_id=world["org"], states=("verified",), scope=PUBLIC, q="望遠鏡"
    )
    assert [a.issue for a in items] == ["望遠鏡の場所"] and total == 1


def test_knowledge_from_a_private_channel_is_hidden_from_members(db, world) -> None:
    secret = create_draft(
        db,
        org_id=world["org"],
        kind="fact",
        issue="暗証番号",
        resolution="会計係に聞く",
        sources=[SourceRef(channel_message_id=world["secret"])],
        now=NOW,
    )
    verify_article(db, secret.id, org_id=world["org"], verifier_id=None, now=NOW)
    assert list_articles(db, org_id=world["org"], states=("verified",), scope=PUBLIC) == ([], 0)
    assert get_article(db, secret.id, org_id=world["org"], scope=PUBLIC) is None
    readers = SourceScope(channels=frozenset({PRIVATE_CHANNEL}))
    assert get_article(db, secret.id, org_id=world["org"], scope=readers) is not None
    assert get_article(db, secret.id, org_id=world["org"], scope=EVERYTHING) is not None


def test_a_hidden_message_hides_the_knowledge_made_from_it(db, world) -> None:
    article = create_draft(
        db,
        org_id=world["org"],
        kind="fact",
        issue="望遠鏡",
        resolution="倉庫",
        sources=[SourceRef(channel_message_id=world["said"])],
        now=NOW,
    )
    verify_article(db, article.id, org_id=world["org"], verifier_id=None, now=NOW)
    assert get_article(db, article.id, org_id=world["org"], scope=PUBLIC) is not None
    db.get(ChannelMessage, world["said"]).hidden_at = NOW
    db.flush()
    assert get_article(db, article.id, org_id=world["org"], scope=PUBLIC) is None


def test_deleting_a_source_hides_the_knowledge_from_members(db, world) -> None:
    """A deleted conversation must not leave its knowledge open to everyone."""

    article = create_draft(
        db,
        org_id=world["org"],
        kind="fact",
        issue="暗証番号",
        resolution="会計係に聞く",
        sources=[SourceRef(channel_message_id=world["secret"])],
        now=NOW,
    )
    verify_article(db, article.id, org_id=world["org"], verifier_id=None, now=NOW)
    db.delete(db.get(ChannelMessage, world["secret"]))
    db.flush()
    db.expire_all()
    assert get_article(db, article.id, org_id=world["org"], scope=PUBLIC) is None
    lost = article_sources(db, [article.id], org_id=world["org"], scope=EVERYTHING)[article.id]
    assert [(s.source_item_id, s.channel_message_id, s.question_id) for s in lost] == [
        (None, None, None)
    ]


# --------------------------------------------------------------------------- #
# search and use
# --------------------------------------------------------------------------- #
def test_search_returns_verified_knowledge_of_the_organization(db, world, fake_embedder) -> None:
    hit = _verified(db, world, fake_embedder, "部室 の 鍵 は どこ", "部室 の 箱")
    draft = create_draft(
        db, org_id=world["org"], kind="fact", issue="部室 の 鍵 は どこ", resolution="x", now=NOW
    )
    embed_articles(db, fake_embedder, article_ids=[draft.id])
    _verified(db, world, fake_embedder, "部室 の 鍵 は どこ", "よその 答え", org=world["other"])
    query = fake_embedder.encode(["部室 の 鍵 は どこ"])[0]
    found = search_verified(db, org_id=world["org"], query_vec=query, scope=PUBLIC, top_k=5)
    assert [a.id for a, _ in found] == [hit]
    assert found[0][1] > 0.8  # the resolution is embedded too


def test_search_skips_retired_and_below_the_floor(db, world, fake_embedder) -> None:
    gone = _verified(db, world, fake_embedder, "部室 の 鍵", "箱")
    retire_article(db, gone, org_id=world["org"], now=NOW)
    _verified(db, world, fake_embedder, "まったく 別 の 話", "x")
    query = fake_embedder.encode(["部室 の 鍵"])[0]
    assert (
        search_verified(db, org_id=world["org"], query_vec=query, scope=PUBLIC, min_similarity=0.5)
        == []
    )


def test_search_respects_what_the_reader_may_see(db, world, fake_embedder) -> None:
    _verified(
        db,
        world,
        fake_embedder,
        "暗証 番号",
        "会計係",
        sources=[SourceRef(channel_message_id=world["secret"])],
    )
    query = fake_embedder.encode(["暗証 番号"])[0]
    assert search_verified(db, org_id=world["org"], query_vec=query, scope=PUBLIC) == []
    assert len(search_verified(db, org_id=world["org"], query_vec=query, scope=EVERYTHING)) == 1


def test_matching_drafts_finds_drafts_near_the_question(db, world, fake_embedder) -> None:
    draft = create_draft(
        db, org_id=world["org"], kind="fact", issue="合宿 の 持ち物", resolution="寝袋", now=NOW
    )
    embed_articles(db, fake_embedder, article_ids=[draft.id])
    _verified(db, world, fake_embedder, "合宿 の 持ち物", "寝袋")
    query = fake_embedder.encode(["合宿 の 持ち物"])[0]
    assert matching_drafts(db, org_id=world["org"], query_vec=query, min_similarity=0.5) == [
        draft.id
    ]
    assert matching_drafts(db, org_id=world["other"], query_vec=query, min_similarity=0.5) == []


def test_record_use_counts_and_stamps(db, world, fake_embedder) -> None:
    article_id = _verified(db, world, fake_embedder, "会費", "月500円")
    later = NOW + dt.timedelta(days=2)
    record_use(db, [article_id], now=NOW)
    record_use(db, [article_id], now=later)
    row = db.get(KnowledgeArticle, article_id)
    db.refresh(row)
    assert row.reuse_count == 2 and row.last_used_at == later


def test_embedding_text_uses_the_kcs_fields(db, world) -> None:
    article = create_draft(
        db,
        org_id=world["org"],
        kind="case",
        issue="鍵がない",
        environment="合宿前",
        resolution="箱を見る",
        cause="戻し忘れ",
        now=NOW,
    )
    row = db.get(KnowledgeArticle, article.id)
    assert article_text(row) == "鍵がない\n合宿前\n箱を見る\n戻し忘れ"


def test_embedding_skips_retired_and_fills_only_missing(db, world, fake_embedder) -> None:
    keep = create_draft(
        db, org_id=world["org"], kind="fact", issue="会費", resolution="月500円", now=NOW
    )
    gone = create_draft(
        db, org_id=world["org"], kind="fact", issue="部費", resolution="年1000円", now=NOW
    )
    retire_article(db, gone.id, org_id=world["org"], now=NOW)
    assert embed_articles(db, fake_embedder) == 1
    assert embed_articles(db, fake_embedder) == 0
    assert db.get(KnowledgeArticle, keep.id).embedding is not None
    assert db.get(KnowledgeArticle, gone.id).embedding is None
