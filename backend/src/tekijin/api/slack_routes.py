"""Slack account linking, login and events.

Endpoints: ``GET /slack/authorize-url``, ``POST /slack/login-url``,
``GET /slack/oauth/start``, ``GET /slack/oauth/callback``,
``POST /slack/link/complete``, ``GET /slack/status``, ``POST /slack/unlink``,
``POST /slack/events``, ``POST /slack/interactivity``.

"Sign in with Slack" account linking, independent of whether DM notifications
are turned on (``Settings.slack_notifications_enabled``) — an employee can link
before a bot token exists; notifications simply stay off until one does. The
callback is a plain browser redirect from Slack with no ``Authorization``
header, so it cannot tell who is linking — and it does not try. It hands the
frontend a short-lived pending token, redeemed with the frontend's own bearer
token (``POST /slack/link/complete``).

Both halves of a link name an identity: the ``state`` names the employee who
STARTED it, the pending token names the Slack account that CONSENTED, and the
bearer names who is FINISHING it. Every attack this flow has seen took one half
from the attacker and the other from the victim — in both directions — so the
starter and the finisher must be the same person (#494). Login is different: it
has no session to finish with, so it is bound to the browser by a nonce cookie
issued at ``/slack/oauth/start``, on the callback's own origin.

``POST /slack/events`` is the Slack -> TEKIJIN direction (#388, #hand-off-chat):
Slack's Events API calls this directly (no TEKIJIN principal — it
authenticates itself via a request signature instead,
:func:`tekijin.slack.verify.verify_signature`). A message posted in a
hand-off's shared Slack channel is routed to whichever TEKIJIN thread that
channel's ``current_thread_id`` currently names (see
``tekijin.slack.notify``'s module docstring) — no relay back into Slack is
needed here, since both parties already see each other's messages natively
in the same channel; this only mirrors them into TEKIJIN.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import hmac
import json
import logging
import secrets
import threading
from urllib.parse import parse_qs, urlparse

import jwt
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, PlainTextResponse, RedirectResponse, Response
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from tekijin.api import schemas
from tekijin.api.service import AgentService
from tekijin.auth.deps import require_admin, require_principal
from tekijin.auth.principal import Principal
from tekijin.auth.tokens import create_access_token
from tekijin.config import Settings, get_settings
from tekijin.data.db import session_scope
from tekijin.data.messages import create_message, thread_parties
from tekijin.data.slack_channel_links import get_channel_link_by_channel_id
from tekijin.data.slack_directory import apply_sync_plan, load_directory_state
from tekijin.data.slack_links import (
    delete_slack_link,
    get_slack_link,
    get_slack_link_by_slack_user_id,
    upsert_slack_link,
)
from tekijin.data.slack_message_anchors import record_message_anchor
from tekijin.models.tables import Employee
from tekijin.slack.capture import (
    KNOWLEDGE_ACTION_IDS,
    KNOWLEDGE_APPROVE_ACTION,
    KNOWLEDGE_DISCARD_ACTION,
    KNOWLEDGE_KEEP_ACTION,
    SOLVE_REACTIONS,
    approve_thread_draft,
    discard_thread_draft,
    is_solve_utterance,
    is_thanks_utterance,
    review_blocks,
    schedule_solve_capture,
    schedule_solve_prompt,
    thread_draft,
)
from tekijin.slack.client import (
    SlackIdentity,
    build_authorize_url,
    exchange_code,
    list_users,
    respond_to_response_url,
)
from tekijin.slack.user_sync import parse_members, plan_user_sync
from tekijin.slack.verify import verify_signature

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/slack", tags=["slack"])

_STATE_PURPOSE = "slack_link"
# A SECOND purpose over the same secret. Without this claim a link state — which
# any signed-in user can mint from /slack/authorize-url — would be replayable at
# the callback to obtain a bearer token (#406).
_LOGIN_STATE_PURPOSE = "slack_login"
# Handed to the frontend after a LINK callback, exchanged in-session for the
# actual link. Short-lived and single-purpose.
_PENDING_LINK_PURPOSE = "slack_link_pending"
_PENDING_TTL_MINUTES = 5.0
_STATE_TTL_MINUTES = 10.0

# Binds the OAuth flow to the browser that started it (#494). A signed, unexpired
# `state` only proves WE minted it; without this, an attacker mints a state
# carrying their own employee id and has a victim complete consent against it,
# and the callback attaches the victim's Slack identity to the attacker's row.
#
# NOT a session cookie — this project keeps sessions in bearer tokens on purpose.
# This is a single-use CSRF nonce with the same lifetime as the state.
#
# `SameSite=Lax` is required, not incidental: Slack's callback is a top-level GET
# navigation from slack.com, which Lax allows and Strict would block. The state
# carries only the nonce's HASH, so seeing a state (it travels through Slack and
# lands in logs) does not let anyone forge the cookie.
_STATE_COOKIE = "tekijin_oauth_state"


def _public_origin(settings: Settings) -> str:
    """The origin Slack sends the browser back to.

    The nonce cookie MUST be issued by this origin: cookies are scoped by host,
    and the app calls the API on a different host entirely (Tailscale IP vs the
    public tunnel), so a cookie set on the API origin is never sent to the
    callback. Deriving it from ``slack_redirect_uri`` keeps the two in step by
    construction.
    """

    parsed = urlparse(settings.slack_redirect_uri)
    return f"{parsed.scheme}://{parsed.netloc}"


def _new_nonce() -> tuple[str, str]:
    """A fresh nonce and the digest to embed in the state."""

    nonce = secrets.token_urlsafe(32)
    return nonce, hashlib.sha256(nonce.encode()).hexdigest()


def _set_nonce_cookie(response: Response, nonce: str, *, settings: Settings) -> None:
    response.set_cookie(
        _STATE_COOKIE,
        nonce,
        max_age=int(_STATE_TTL_MINUTES * 60),
        httponly=True,
        samesite="lax",
        # Judged by the origin that ISSUES the cookie, which is now the same
        # origin as the callback (see _public_origin). A Secure cookie is simply
        # dropped by the browser over plain HTTP, so a local setup must not get one.
        secure=settings.slack_redirect_uri.startswith("https://"),
        path="/slack",
    )


def _nonce_matches(request: Request, state_digest: object) -> bool:
    """True when this browser holds the nonce the state was minted with."""

    presented = request.cookies.get(_STATE_COOKIE)
    if not presented or not isinstance(state_digest, str):
        return False
    return hmac.compare_digest(hashlib.sha256(presented.encode()).hexdigest(), state_digest)


def _require_linkable_employee(principal: Principal) -> int:
    """An admin has no employee id and never receives chat messages — nothing
    for them to link."""

    if principal.employee_id is None:
        raise HTTPException(status_code=403, detail="管理者はSlack連携を利用できません。")
    return principal.employee_id


def _encode_state(
    *,
    purpose: str,
    secret: str,
    nonce_digest: str | None = None,
    employee_id: int | None = None,
) -> str:
    now = dt.datetime.now(dt.UTC)
    payload: dict[str, object] = {
        "purpose": purpose,
        "iat": int(now.timestamp()),
        "exp": int((now + dt.timedelta(minutes=_STATE_TTL_MINUTES)).timestamp()),
    }
    if nonce_digest is not None:
        payload["nonce"] = nonce_digest
    if employee_id is not None:
        # Who STARTED the link. Never used to choose a row — only compared with
        # who finishes it (see link_complete). Every attack found on this flow has
        # been the same shape: one half from the attacker, the other from the
        # victim. Requiring both halves to name the same person is what closes it.
        payload["employee_id"] = employee_id
    return jwt.encode(payload, secret, algorithm="HS256")


def _spent[R: Response](response: R) -> R:
    """Burn the nonce. Single-use, so even the browser that started the flow
    cannot replay a captured ``code``+``state`` pair a second time."""

    response.delete_cookie(_STATE_COOKIE, path="/slack")
    return response


def _verified_state(state: str, *, request: Request, secret: str) -> dict:
    """The decoded state, once it is ours AND this browser started the flow.

    Both halves matter. The signature says we minted it; the nonce says the
    browser now presenting it is the one we minted it for (#494).
    """

    try:
        payload = jwt.decode(state, secret, algorithms=["HS256"])
    except jwt.PyJWTError as exc:
        raise HTTPException(status_code=400, detail="invalid or expired state") from exc
    purpose = payload.get("purpose")
    if purpose not in (_STATE_PURPOSE, _LOGIN_STATE_PURPOSE):
        raise HTTPException(status_code=400, detail="invalid state")
    # Only LOGIN mints a session out of thin air, so only LOGIN needs to prove
    # the browser started the flow. LINK is protected structurally instead: the
    # callback cannot attach anything on its own — completion is authenticated.
    if purpose == _LOGIN_STATE_PURPOSE and not _nonce_matches(request, payload.get("nonce")):
        raise HTTPException(status_code=400, detail="state was not started by this browser")
    return payload


@router.get("/authorize-url", response_model=schemas.SlackAuthorizeUrlResponse)
def authorize_url(
    principal: Principal = Depends(require_principal),
) -> schemas.SlackAuthorizeUrlResponse:
    """The "Sign in with Slack" URL for the frontend to navigate to.

    503 (not a client error) while no Slack App is registered yet
    (``TEKIJIN_SLACK_CLIENT_ID`` / ``_SECRET`` / ``_REDIRECT_URI`` unset) — the
    button is a temporarily-unavailable feature, not a caller mistake.
    """

    employee_id = _require_linkable_employee(principal)
    settings = get_settings()
    if not settings.slack_configured():
        raise HTTPException(status_code=503, detail="Slack連携は現在利用できません。")
    state = _encode_state(
        purpose=_STATE_PURPOSE, employee_id=employee_id, secret=settings.auth_secret
    )
    url = build_authorize_url(
        client_id=settings.slack_client_id,
        redirect_uri=settings.slack_redirect_uri,
        state=state,
    )
    return schemas.SlackAuthorizeUrlResponse(url=url)


@router.post("/login-url", response_model=schemas.SlackAuthorizeUrlResponse)
def login_url() -> schemas.SlackAuthorizeUrlResponse:
    """ "Sign in with Slack" for someone who has NO session yet (#406 案A).

    Deliberately unauthenticated — that is the whole point — and it reveals
    nothing: which employee the login becomes is decided by the Slack identity
    the callback resolves.

    POST rather than GET so a third-party page cannot trigger it with
    ``<img src>``; a cross-site POST needs a preflight this origin allowlist
    refuses.
    """

    settings = get_settings()
    if not settings.slack_login_enabled or not settings.slack_configured():
        raise HTTPException(status_code=503, detail="Slackログインは現在利用できません。")
    # Points at OUR start endpoint, not at Slack. The browser has to visit the
    # callback's own origin first so the nonce cookie is issued there; issuing it
    # from this response would put it on the API origin, which the callback host
    # never sees (#494).
    return schemas.SlackAuthorizeUrlResponse(url=f"{_public_origin(settings)}/slack/oauth/start")


@router.get("/oauth/start")
def oauth_start(response: Response) -> RedirectResponse:
    """Begin a Slack LOGIN: issue the nonce, then bounce to Slack.

    A plain top-level navigation, so there is no CORS and no third-party cookie
    involved — the cookie is first-party to this origin, and Slack's callback
    returns to the same one.
    """

    settings = get_settings()
    if not settings.slack_login_enabled or not settings.slack_configured():
        raise HTTPException(status_code=503, detail="Slackログインは現在利用できません。")
    nonce, digest = _new_nonce()
    redirect = RedirectResponse(
        build_authorize_url(
            client_id=settings.slack_client_id,
            redirect_uri=settings.slack_redirect_uri,
            state=_encode_state(
                purpose=_LOGIN_STATE_PURPOSE, nonce_digest=digest, secret=settings.auth_secret
            ),
        )
    )
    _set_nonce_cookie(redirect, nonce, settings=settings)
    return redirect


def _login_redirect(settings: Settings, identity: SlackIdentity, session: Session) -> str:
    """Where to send the browser after a Slack LOGIN attempt.

    The token rides in the URL **fragment**, never the query string: a query
    parameter is written verbatim into the server access log (the OAuth ``code``
    already appears there), while a fragment is never sent to any server.
    """

    frontend = settings.slack_frontend_url.rstrip("/")
    link = get_slack_link_by_slack_user_id(
        session, identity.slack_user_id, expected_team_id=settings.slack_team_id
    )
    if link is None:
        # Authenticated by Slack, but no employee claims this identity. Sync
        # (#406 step 3) is what normally fills this in.
        logger.info("Slack login for an unlinked slack_user %s", identity.slack_user_id)
        return f"{frontend}/login?slack=unlinked"
    employee = session.get(Employee, link.employee_id)
    if employee is None:
        return f"{frontend}/login?slack=error"
    token = create_access_token(
        Principal(
            employee_id=employee.id,
            name=employee.name,
            dept=employee.department,
            is_admin=False,
        ),
        secret=settings.auth_secret,
        ttl_hours=settings.auth_token_ttl_hours,
    )
    return f"{frontend}/login#slack_token={token}"


@router.get("/oauth/callback")
def oauth_callback(
    request: Request,
    code: str | None = Query(default=None),
    state: str | None = Query(default=None),
    error: str | None = Query(default=None),
) -> RedirectResponse:
    """Slack's OAuth redirect target. No ``Authorization`` header is available
    here (this is a top-level browser navigation, not a fetch from the SPA), so
    the employee comes from the signed ``state`` instead (#slack-integration).

    Always ends in a redirect back to the frontend chat page — success or
    failure — never a bare JSON error, since a human is looking at the browser
    at this point, not calling the API.
    """

    settings = get_settings()
    frontend_chat_url = f"{settings.slack_frontend_url.rstrip('/')}/chat"
    if error or not code or not state:
        return _spent(RedirectResponse(f"{frontend_chat_url}?slack=error"))
    try:
        payload = _verified_state(state, request=request, secret=settings.auth_secret)
        purpose = payload["purpose"]
        identity = exchange_code(
            client_id=settings.slack_client_id,
            client_secret=settings.slack_client_secret,
            redirect_uri=settings.slack_redirect_uri,
            code=code,
        )
        if purpose == _LOGIN_STATE_PURPOSE and not settings.slack_login_enabled:
            # A state minted while the feature was on must not outlive it.
            logger.warning("Slack login attempted while disabled")
            return _spent(RedirectResponse(f"{frontend_chat_url}?slack=error"))
        if settings.slack_team_id and identity.slack_team_id != settings.slack_team_id:
            # Same redirect as any other failure — the person is not one of ours,
            # so telling them WHICH workspace we expect would leak it.
            logger.warning(
                "Slack OAuth from unexpected workspace %s (expected %s)",
                identity.slack_team_id,
                settings.slack_team_id,
            )
            return _spent(RedirectResponse(f"{frontend_chat_url}?slack=error"))
        service = request.app.state.agent_service
        if purpose == _LOGIN_STATE_PURPOSE:
            with service.session_factory() as session:
                return _spent(RedirectResponse(_login_redirect(settings, identity, session)))
        # The callback CANNOT attach the identity by itself: it has no session, so
        # it does not know who is linking. It hands the frontend a short-lived
        # pending token, which the frontend redeems with its own bearer token
        # (POST /slack/link/complete). That is what makes a link URL harmless to
        # forward to someone else (#494).
        pending = jwt.encode(
            {
                "purpose": _PENDING_LINK_PURPOSE,
                "employee_id": payload.get("employee_id"),
                "slack_user_id": identity.slack_user_id,
                "slack_team_id": identity.slack_team_id,
                "exp": int(
                    (
                        dt.datetime.now(dt.UTC) + dt.timedelta(minutes=_PENDING_TTL_MINUTES)
                    ).timestamp()
                ),
            },
            settings.auth_secret,
            algorithm="HS256",
        )
        return _spent(RedirectResponse(f"{frontend_chat_url}#slack_pending={pending}"))
    except Exception:  # noqa: BLE001 - browser redirect boundary, never surfaces a bare 4xx/5xx
        logger.warning("Slack OAuth callback failed", exc_info=True)
        return _spent(RedirectResponse(f"{frontend_chat_url}?slack=error"))


@router.post("/link/complete", response_model=schemas.SlackStatusResponse)
def link_complete(
    request: Request,
    body: schemas.SlackLinkCompleteRequest,
    principal: Principal = Depends(require_principal),
) -> schemas.SlackStatusResponse:
    """Redeem a pending Slack link against the CALLER's own session (#494).

    The employee comes from the bearer token, never from anything that travelled
    through the browser — so a link URL forwarded to someone else can only ever
    attach that person's Slack account to their own row.
    """

    employee_id = _require_linkable_employee(principal)
    settings = get_settings()
    try:
        payload = jwt.decode(body.pending_token, settings.auth_secret, algorithms=["HS256"])
    except jwt.PyJWTError as exc:
        raise HTTPException(status_code=400, detail="連携の有効期限が切れました。") from exc
    if payload.get("purpose") != _PENDING_LINK_PURPOSE:
        raise HTTPException(status_code=400, detail="連携情報が正しくありません。")
    # THE check. The pending token names the Slack account that consented and the
    # employee who started the flow; the bearer names who is finishing it. Every
    # attack on this flow has been "one half from each person", in both
    # directions — so the two must name the same employee.
    if payload.get("employee_id") != employee_id:
        logger.warning(
            "Slack link redeemed by employee %s but started by %s",
            employee_id,
            payload.get("employee_id"),
        )
        raise HTTPException(
            status_code=403, detail="この連携はあなたが開始したものではありません。"
        )
    team = str(payload.get("slack_team_id", ""))
    if settings.slack_team_id and team != settings.slack_team_id:
        raise HTTPException(status_code=400, detail="このワークスペースは利用できません。")

    service = request.app.state.agent_service
    try:
        with session_scope(service.session_factory) as session:
            upsert_slack_link(
                session,
                employee_id,
                slack_user_id=str(payload["slack_user_id"]),
                slack_team_id=team,
                now=dt.datetime.now(),  # noqa: DTZ005 - naive is intentional, matches created_at
            )
    except IntegrityError as exc:
        # `slack_user_id` is unique: that Slack account already belongs to a
        # different employee. Say so — "error" leaves the user with no next step.
        raise HTTPException(
            status_code=409, detail="このSlackアカウントは既に他の社員と連携されています。"
        ) from exc
    return schemas.SlackStatusResponse(linked=True)


@router.get("/status", response_model=schemas.SlackStatusResponse)
def status(
    request: Request,
    principal: Principal = Depends(require_principal),
) -> schemas.SlackStatusResponse:
    if principal.employee_id is None:
        return schemas.SlackStatusResponse(linked=False)
    service = request.app.state.agent_service
    with service.session_factory() as session:
        link = get_slack_link(
            session, principal.employee_id, expected_team_id=get_settings().slack_team_id
        )
    return schemas.SlackStatusResponse(linked=link is not None)


@router.post("/unlink", response_model=schemas.SlackUnlinkResponse)
def unlink(
    request: Request,
    principal: Principal = Depends(require_principal),
) -> schemas.SlackUnlinkResponse:
    employee_id = _require_linkable_employee(principal)
    service = request.app.state.agent_service
    with session_scope(service.session_factory) as session:
        delete_slack_link(session, employee_id)
    return schemas.SlackUnlinkResponse()


@router.post("/sync-users", response_model=schemas.SlackUserSyncResponse)
def sync_users(
    request: Request,
    principal: Principal = Depends(require_admin),
) -> schemas.SlackUserSyncResponse:
    """Reconcile ``slack_links`` against the Slack workspace roster (#406 step 3).

    Admin-only and off by default. Run it on a schedule with cron — an hourly
    poll is enough, and the sync is self-healing, so a missed run costs nothing
    but a delay. It is an endpoint rather than an in-process timer on purpose:
    a background thread in this codebase has already swallowed a failure
    unnoticed, and this one writes the table that decides who can log in.

    Every decision is made by the pure planner in ``tekijin.slack.user_sync``;
    this handler only supplies the inputs and reports the outcome. In
    particular it never overwrites an existing link and never re-points a Slack
    account at a different employee — see that module for why.
    """

    settings = get_settings()
    if not settings.slack_user_sync_enabled:
        raise HTTPException(status_code=503, detail="Slackユーザー同期は現在無効です。")
    if not settings.slack_notifications_enabled():
        # The roster comes from the bot token; without one there is nothing to
        # read. Saying so beats reporting a successful sync of nobody.
        raise HTTPException(status_code=503, detail="Slackのボットトークンが設定されていません。")
    if not settings.slack_team_id:
        # A blank workspace makes the planner match nobody, which would look
        # exactly like a workspace where nothing needed doing.
        raise HTTPException(status_code=503, detail="TEKIJIN_SLACK_TEAM_ID が設定されていません。")

    try:
        raw = list_users(bot_token=settings.slack_bot_token)
    except Exception as exc:  # noqa: BLE001 - upstream failure, reported as 502
        logger.warning("Slack users.list failed", exc_info=True)
        raise HTTPException(
            status_code=502, detail="Slackのユーザー一覧を取得できませんでした。"
        ) from exc

    members = parse_members(raw)
    service = request.app.state.agent_service
    now = dt.datetime.now(dt.UTC).replace(tzinfo=None)
    try:
        with session_scope(service.session_factory) as session:
            state = load_directory_state(session)
            plan = plan_user_sync(
                members,
                employee_id_by_email=state.employee_id_by_email,
                linked_slack_user_by_employee=state.linked_slack_user_by_employee,
                employee_by_slack_user=state.employee_by_slack_user,
                expected_team_id=settings.slack_team_id,
                admin_email=settings.admin_email,
                create_missing=settings.slack_user_sync_create_employees,
                allowed_create_domains=settings.slack_user_sync_allowed_domains,
            )
            applied = apply_sync_plan(session, plan, team_id=settings.slack_team_id, now=now)
    except (ValueError, IntegrityError) as exc:
        # The planner refuses ambiguous pairs and the applier refuses a duplicated
        # plan, so reaching here means the first guard has a hole. The batch is
        # rolled back whole — including any departure unlinks in it — so this has
        # to be legible enough that someone goes and looks, not a bare traceback.
        logger.error("Slack directory sync could not be applied", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Slackユーザー同期を適用できませんでした。管理者に連絡してください。",
        ) from exc

    logger.info(
        "Slack directory sync by admin: %s members, created=%s linked=%s unlinked=%s skipped=%s",
        len(members),
        applied["created"],
        applied["linked"],
        applied["unlinked"],
        plan.skipped,
    )
    return schemas.SlackUserSyncResponse(
        created=applied["created"],
        linked=applied["linked"],
        unlinked=applied["unlinked"],
        skipped=dict(plan.skipped),
    )


def _handle_message_event(session_factory: sessionmaker[Session], event: dict) -> None:
    """Mirror one message posted in a hand-off's shared Slack channel into
    the TEKIJIN thread it currently belongs to (#hand-off-chat).

    Silently drops anything it cannot confidently route (channel isn't one of
    ours, unknown sender, sender no longer a party) rather than erroring —
    this runs inside a webhook Slack will retry on any non-2xx, so "nothing
    to do" must look identical to "handled" from Slack's side. Synchronous
    (plain ``def``, run via ``run_in_threadpool`` by the caller): does
    blocking DB I/O, which must never run directly on the asyncio event loop
    that ``POST /slack/events`` (an ``async def`` handler) shares with every
    other concurrent request this worker is serving.
    """

    if event.get("type") != "message":
        return
    # Never the bot's own post landing back in the same channel it was sent
    # to (that would loop), and message_changed / message_deleted / channel
    # joins etc. are not a new human message.
    if event.get("bot_id") or event.get("subtype"):
        return
    channel_id = event.get("channel")
    slack_user_id = event.get("user")
    text = event.get("text")
    message_ts = event.get("ts")
    if not channel_id or not slack_user_id or not text or not text.strip():
        return

    prompt_thread_id: int | None = None
    with session_scope(session_factory) as session:
        channel_link = get_channel_link_by_channel_id(session, channel_id)
        if channel_link is None:
            return  # not a channel TEKIJIN created — ignore
        sender_link = get_slack_link_by_slack_user_id(
            session, slack_user_id, expected_team_id=get_settings().slack_team_id
        )
        if sender_link is None:
            return
        thread_id = channel_link.current_thread_id
        sender_id = sender_link.employee_id
        parties = thread_parties(session, thread_id)
        if parties is None or sender_id not in (parties["asker_id"], parties["responder_id"]):
            return
        now = dt.datetime.now()  # noqa: DTZ005 - naive is intentional, matches created_at elsewhere
        create_message(session, thread_id, sender_id, text, now)
        if get_settings().slack_solve_capture_enabled:
            # #476/#508: remember which thread THIS message belonged to, so a later ✅
            # reaction on it is attributed to the right thread even after the channel
            # is reused for a newer hand-off. Only while solve-capture is on, so the
            # flag-off path stays byte-identical (no extra write).
            if message_ts:
                record_message_anchor(
                    session,
                    slack_channel_id=channel_id,
                    slack_ts=message_ts,
                    thread_id=thread_id,
                    now=now,
                )
            # #476/#537: an explicit resolution from either party, or an explicit
            # thanks from the ASKER, is a capture trigger. Sender identity matters
            # for thanks: a responder's polite closing is not evidence that the
            # asker's problem is resolved. Defer the draft + in-thread prompt to a
            # daemon thread (LLM + Slack post exceed the ~3s budget); scheduled
            # AFTER this transaction commits.
            if is_solve_utterance(text) or (
                sender_id == parties["asker_id"] and is_thanks_utterance(text)
            ):
                prompt_thread_id = thread_id

    if prompt_thread_id is not None:
        schedule_solve_prompt(session_factory, channel_id=channel_id, thread_id=prompt_thread_id)


def _handle_reaction_event(session_factory: sessionmaker[Session], event: dict) -> None:
    """Solve-capture (#476): a ✅ reaction on a pair-channel thread queues a
    knowledge draft of the resolved Q&A.

    Cheap, synchronous gating only (event shape + the feature flag); the actual
    channel/party resolution and the LLM extraction happen in
    :func:`schedule_solve_capture`'s daemon thread so the webhook still acks within
    Slack's ~3s budget. A no-op — never an error — for anything it should not act
    on (wrong reaction, flag off, missing ids), so Slack never sees a 5xx and never
    retries. When the flag is OFF (default) this returns before touching anything,
    keeping the events path byte-identical to before #476.
    """

    if event.get("type") != "reaction_added":
        return
    if event.get("reaction") not in SOLVE_REACTIONS:
        return
    if not get_settings().slack_solve_capture_enabled:
        return
    item = event.get("item") or {}
    channel_id = item.get("channel")
    message_ts = item.get("ts")
    reactor_slack_user_id = event.get("user")
    if not channel_id or not reactor_slack_user_id:
        return
    schedule_solve_capture(
        session_factory,
        channel_id=channel_id,
        message_ts=message_ts,
        reactor_slack_user_id=reactor_slack_user_id,
    )


@router.post("/events")
async def events(request: Request) -> Response:
    """Slack Events API endpoint: the URL-verification handshake, plus inbound
    hand-off-channel messages mirrored into the matching TEKIJIN chat thread
    (#hand-off-chat).

    No TEKIJIN auth — Slack calls this directly, authenticated instead by its
    own request signature. Always acks within Slack's 3s budget and never
    raises past a signature failure: anything else it can't route is just a
    no-op (see :func:`_handle_message_event`), so a malformed or duplicate
    delivery never turns into a 5xx that trains Slack to keep retrying.

    ``event_id`` de-dup guards against Slack's OWN retries (it re-delivers
    whenever an earlier attempt didn't ack in time — plausible before the
    event-loop fix above, but Slack does not guarantee exactly-once even
    otherwise) so a retried delivery can never insert the same message twice.
    """

    settings = get_settings()
    body = await request.body()
    if not verify_signature(
        signing_secret=settings.slack_signing_secret,
        timestamp=request.headers.get("X-Slack-Request-Timestamp", ""),
        signature=request.headers.get("X-Slack-Signature", ""),
        body=body,
    ):
        raise HTTPException(status_code=401, detail="invalid signature")

    payload = json.loads(body)
    if payload.get("type") == "url_verification":
        # Slack's documented handshake: echo the challenge back as plain text.
        return PlainTextResponse(str(payload.get("challenge", "")))

    if payload.get("type") == "event_callback":
        event_id = payload.get("event_id")
        seen_events = request.app.state.slack_seen_event_ids
        if event_id is None or not seen_events.seen_before(event_id):
            event = payload.get("event") or {}
            service = request.app.state.agent_service
            event_type = event.get("type")
            if event_type == "reaction_added":
                # Solve-capture (#476): cheap gating on the loop, then a daemon
                # thread does the DB resolution + LLM extraction (never blocks ack).
                _handle_reaction_event(service.session_factory, event)
            else:
                # Message mirroring (#388): a quick DB write, safe to run in the
                # threadpool while the request awaits it.
                await run_in_threadpool(_handle_message_event, service.session_factory, event)

    return JSONResponse({"ok": True})


_INTERACTIVITY_FAILURE_TEXT = "この依頼は処理できませんでした。TEKIJINで状態を確認してください。"

_RESULT_TEXT_BY_ACTION_ID = {
    "tekijin_accept": "承諾しました。TEKIJINの依頼状態を更新しています。",
    "tekijin_decline": "辞退しました。依頼者へ通知します。",
    "tekijin_refer": "「自分より適任がいる」として辞退しました。依頼者へ通知します。",
}


def _advance_after_resume(service: AgentService, session_id: str) -> None:
    """Drain the just-queued resume so the paused run actually advances.

    ``submit_resume`` only queues a ``Command`` (service.py); the frontend's
    own button click always follows it with ``advanceSession()`` ->
    ``GET /events/{session_id}`` to consume that queue and drive the graph
    forward (accept -> done, decline -> reroute to the next candidate) — see
    that function's doc comment. A Slack click has no such stream of its own,
    so without this the hand-off would sit parked until the asker's tab
    happens to reconnect. Runs fire-and-forget in its own thread (mirrors
    ``notify.schedule_pending_handoff``): draining can invoke LLM calls (e.g.
    the reroute), which routinely runs well past Slack's ~3s interactivity
    budget.
    """

    try:
        if service.is_streamable(session_id):
            for _ in service.stream_events(session_id):
                pass
    except Exception:
        logger.warning(
            "Slack interactivity: failed to advance session %s", session_id, exc_info=True
        )


def _handle_knowledge_action(
    service: AgentService,
    action_id: str,
    value: dict,
    slack_user_id: str | None,
    response_url: str | None,
    original_blocks: list[dict],
) -> Response:
    """Apply a knowledge-prompt click, running the whole review loop inside Slack (#527).

    Flow: 残す → show the extracted draft in-thread with 承認する / 破棄する → 承認する
    marks it ``verified`` (it may answer questions) → 破棄する retires it (durable — a
    later capture never makes a new one). No web hop: the reviewer reads and decides
    without leaving Slack.

    Authorization: any PARTY of the thread (asker or responder) may act on the shared
    draft — unlike accept/decline, not the responder's sole call. The prompt only exists
    in the pair's private 2-member channel, so membership is already the gate; the party
    check is the same belt-and-suspenders the reaction path uses. Never raises past a
    friendly Slack message (same contract as the hand-off path).
    """

    try:
        thread_id = int(value["thread_id"])
    except (KeyError, TypeError, ValueError):
        if response_url:
            respond_to_response_url(
                response_url, {"response_type": "ephemeral", "text": _INTERACTIVITY_FAILURE_TEXT}
            )
        return JSONResponse({"text": _INTERACTIVITY_FAILURE_TEXT})

    with session_scope(service.session_factory) as session:
        parties = thread_parties(session, thread_id)
        clicker = (
            get_slack_link_by_slack_user_id(
                session, str(slack_user_id), expected_team_id=get_settings().slack_team_id
            )
            if slack_user_id
            else None
        )
        if (
            parties is None
            or clicker is None
            or clicker.employee_id not in (parties["asker_id"], parties["responder_id"])
        ):
            if response_url:
                respond_to_response_url(
                    response_url,
                    {"response_type": "ephemeral", "text": "この操作を行う権限がありません。"},
                )
            return JSONResponse({"text": "この操作を行う権限がありません。"})

        approved = False
        if action_id == KNOWLEDGE_KEEP_ACTION:
            # 残す → show the draft's content in-thread so it can be approved/discarded
            # right here, replacing the keep/discard prompt (the "確認" surface, #527).
            article = thread_draft(session, thread_id)
            if article is not None:
                if response_url:
                    respond_to_response_url(
                        response_url,
                        {
                            "replace_original": True,
                            "text": "この下書きを確認してください。",
                            "blocks": review_blocks(thread_id=thread_id, article=article),
                        },
                    )
                return JSONResponse({"text": "この下書きを確認してください。"})
            text = "下書きが見つかりませんでした。もう一度お試しください。"
        elif action_id == KNOWLEDGE_APPROVE_ACTION:
            if approve_thread_draft(session, thread_id, verifier_id=clicker.employee_id):
                approved = True
                text = "承認しました。今後の回答でこの知識が活用されます。"
            else:
                text = "下書きが見つかりませんでした。もう一度お試しください。"
        elif action_id == KNOWLEDGE_DISCARD_ACTION:
            discard_thread_draft(session, thread_id)
            text = "この会話は知識として残しません。"
        else:  # unreachable: KNOWLEDGE_ACTION_IDS gates entry, but fail closed
            logger.warning("Slack interactivity: unhandled knowledge action_id %s", action_id)
            text = _INTERACTIVITY_FAILURE_TEXT

    if approved:
        # After the approve commits, embed the unit so knowledge_answer can find it
        # immediately — nothing else indexes knowledge units in the running app (#531).
        service.schedule_knowledge_embed()

    if response_url:
        kept_blocks = [b for b in original_blocks if b.get("type") != "actions"]
        kept_blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": f"*{text}*"}})
        respond_to_response_url(
            response_url, {"replace_original": True, "text": text, "blocks": kept_blocks}
        )
    return JSONResponse({"text": text})


def _handle_interactivity_action(service: AgentService, raw: str) -> Response:
    """Apply one Slack 承諾/辞退/自分より適任がいる button click.

    Synchronous — does blocking DB I/O and can hold ``submit_resume``'s
    per-session lock — so the caller runs it via ``run_in_threadpool`` rather
    than inline on the event loop (matching ``_handle_message_event`` above).

    Never lets an exception (including an auth mismatch) escape as a raw
    non-2xx: that is exactly what makes Slack mark the message with its
    "processing failed" warning triangle, so every failure here instead
    resolves to a 200 with a friendly Slack-facing message.

    For a ``block_actions`` payload (a button click), that 200 JSON body is
    otherwise inert — Slack only looks at its status code, so returning
    ``{"text": ...}`` here does NOT update the message or remove the buttons.
    ``response_url`` (below) is what actually changes what the message shows.
    """

    response_url: str | None = None
    original_blocks: list[dict] = []
    try:
        payload = json.loads(raw)
        response_url = payload.get("response_url")
        # The message this action was attached to (Slack echoes it back in the
        # payload) — kept so a successful click can drop just the button row
        # instead of wiping the whole message, including the consultation
        # text, when it replaces the message below.
        original_blocks = (payload.get("message") or {}).get("blocks") or []
        action = (payload.get("actions") or [])[0]
        action_id = action.get("action_id")
        value = json.loads(action.get("value", "{}"))
        slack_user_id = (payload.get("user") or {}).get("id")
        # #476 Screen 02: the knowledge keep/discard buttons carry {thread_id}, not the
        # hand-off {session_id, outcome, ...}, so branch BEFORE parsing those. Handled
        # in its own function (a different auth: any thread party may decide, not only
        # the assigned responder).
        if action_id in KNOWLEDGE_ACTION_IDS:
            return _handle_knowledge_action(
                service, action_id, value, slack_user_id, response_url, original_blocks
            )
        session_id = value["session_id"]
        outcome = value["outcome"]
        recommendation_id = int(value["recommendation_id"])
        # A payload without `user.id` cannot identify a responder, so decide it here
        # rather than sending the None to SQL. `slack_links.slack_user_id` is
        # NOT NULL, so `WHERE slack_user_id IS NULL` could never have matched — the
        # old code was not unsafe, it just spent a round-trip proving that and made
        # the outcome depend on a constraint stated nowhere near here (#441).
        # Truthiness rather than `is not None`: an empty-string id is equally
        # unusable, and this way both fail closed at the same place.
        responder_id = None
        if slack_user_id:
            with service.session_factory() as session:
                link = get_slack_link_by_slack_user_id(
                    session, str(slack_user_id), expected_team_id=get_settings().slack_team_id
                )
                responder_id = link.employee_id if link else None
        _, current_responder_id = service.session_participants(session_id)
        if responder_id is None or responder_id != current_responder_id:
            logger.info(
                "Slack interactivity: slack_user %s is not session %s's assigned responder",
                slack_user_id,
                session_id,
            )
            # Ephemeral (visible only to the clicker), NOT replace_original: the
            # real responder still needs the buttons intact on this message.
            if response_url:
                respond_to_response_url(
                    response_url,
                    {"response_type": "ephemeral", "text": "この操作を行う権限がありません。"},
                )
            return JSONResponse({"text": "この操作を行う権限がありません。"})
        service.submit_resume(session_id, outcome=outcome, recommendation_id=recommendation_id)
    except Exception:
        logger.warning("Slack interactivity handling failed", exc_info=True)
        if response_url:
            respond_to_response_url(
                response_url, {"response_type": "ephemeral", "text": _INTERACTIVITY_FAILURE_TEXT}
            )
        return JSONResponse({"text": _INTERACTIVITY_FAILURE_TEXT})

    threading.Thread(
        target=_advance_after_resume,
        args=(service, session_id),
        daemon=True,
        name="slack-interactivity-advance",
    ).start()
    text = _RESULT_TEXT_BY_ACTION_ID.get(
        action_id,
        "承諾しました。" if outcome == "accepted" else "辞退しました。依頼者へ通知します。",
    )
    # Handled -> rewrite the message dropping only the button row (so it can't
    # be clicked, or double-clicked, again) while keeping the original
    # consultation text intact, with the outcome appended as its own line.
    if response_url:
        kept_blocks = [b for b in original_blocks if b.get("type") != "actions"]
        kept_blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": f"*{text}*"}})
        respond_to_response_url(
            response_url,
            {"replace_original": True, "text": text, "blocks": kept_blocks},
        )
    return JSONResponse({"text": text})


@router.post("/interactivity")
async def interactivity(request: Request) -> Response:
    """Handle Slack Block Kitの承諾・辞退・自分より適任がいるボタン for pending hand-offs."""
    settings = get_settings()
    body = await request.body()
    if not verify_signature(
        signing_secret=settings.slack_signing_secret,
        timestamp=request.headers.get("X-Slack-Request-Timestamp", ""),
        signature=request.headers.get("X-Slack-Signature", ""),
        body=body,
    ):
        raise HTTPException(status_code=401, detail="invalid signature")
    raw = parse_qs(body.decode("utf-8"), keep_blank_values=True).get("payload", [""])[0]
    service = request.app.state.agent_service
    return await run_in_threadpool(_handle_interactivity_action, service, raw)
