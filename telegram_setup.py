"""
Helper una-tantum per trovare il tuo TELEGRAM_CHAT_ID.

1. Crea il bot via @BotFather e ottieni il TOKEN
2. Manda un qualsiasi messaggio al tuo bot
3. Esegui: python telegram_setup.py --token <IL_TUO_TOKEN>
   Il chat_id verrà stampato e scritto nel file .env
"""

import sys
import argparse
import requests
from pathlib import Path


def get_chat_id(token: str) -> int | None:
    url  = f"https://api.telegram.org/bot{token}/getUpdates"
    resp = requests.get(url, timeout=10)
    data = resp.json()

    if not data.get("ok"):
        print(f"Errore API Telegram: {data}")
        return None

    updates = data.get("result", [])
    if not updates:
        print("Nessun messaggio trovato. Assicurati di aver mandato un messaggio al bot e riprova.")
        return None

    chat_id = updates[-1]["message"]["chat"]["id"]
    return chat_id


def write_env(token: str, chat_id: int):
    env_path = Path(__file__).parent / ".env"
    existing = env_path.read_text(encoding="utf-8") if env_path.exists() else ""

    lines = [l for l in existing.splitlines() if not l.startswith("TELEGRAM_")]
    lines += [
        f"TELEGRAM_BOT_TOKEN={token}",
        f"TELEGRAM_CHAT_ID={chat_id}",
    ]
    env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f".env aggiornato: {env_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", required=True, help="Token del bot Telegram")
    args = parser.parse_args()

    print(f"Recupero aggiornamenti per il bot...")
    chat_id = get_chat_id(args.token)
    if chat_id is None:
        sys.exit(1)

    print(f"\nTrovato! TELEGRAM_CHAT_ID = {chat_id}")
    write_env(args.token, chat_id)
    print("\nOra puoi lanciare la pipeline con:")
    print("  python pipeline_runner.py")
    print("  python pipeline_runner.py --stop-on-error  # si ferma al primo errore")


if __name__ == "__main__":
    main()
