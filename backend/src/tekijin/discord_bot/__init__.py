"""Discord bot: a separate process that relays ``/ask`` to the backend (Discord D1, #60).

Run with ``python -m tekijin.discord_bot``. Only ``__main__`` imports discord.py; the
reply formatting in :mod:`tekijin.discord_bot.format` is plain Python and unit-tested.
"""
