import json
import os
import time
import hashlib
import yagmail
from threading import Thread

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

USERS_FILE = "users.json"
NEWS_FILE = "news.json"

EMAIL_USER = ""
EMAIL_PASS = ""


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def init_files():
    if not os.path.exists(USERS_FILE):
        with open(USERS_FILE, "w") as f:
            json.dump([], f)

init_files()

@app.post("/signup")
async def signup(request: Request):
    data = await request.json()
    email = data.get("email")

    with open(USERS_FILE, "r") as f:
        users = json.load(f)

    if email in users:
        return {"message": "User already exists"}

    users.append(email)

    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=4)

    return {"message": "Signup successful"}


@app.get("/news")
def get_news():
    if not os.path.exists(NEWS_FILE):
        return []

    with open(NEWS_FILE, "r") as f:
        return json.load(f)


last_hash = None
last_sent_titles = set()

def get_file_hash():
    try:
        with open(NEWS_FILE, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()
    except:
        return None


def should_send(data):
    global last_sent_titles

    if not data:
        return False

    
    if data[0].get("score", 0) < 40:
        print("⚠️ Not important → skip email")
        return False

    current_titles = {item["title"] for item in data[:5]}

    if current_titles == last_sent_titles:
        print("⚠️ Duplicate news → skip email")
        return False

    last_sent_titles = current_titles
    return True


def send_email(data):
    try:
        with open(USERS_FILE, "r") as f:
            users = json.load(f)

        if not users:
            print("⚠️ No users to send email")
            return

        yag = yagmail.SMTP(EMAIL_USER, EMAIL_PASS)

        content = "<h2>📈 Market News Update</h2>"

        for item in data[:10]:
            content += f"""
            <p>
                <b>{item['title']}</b><br>
                Score: {item.get('score', 'N/A')}<br>
                <a href="{item['link']}">Read more</a>
            </p>
            <hr>
            """

        yag.send(
            to=users,
            subject="🚨 Market Alert",
            contents=content
        )

        print("📧 Emails sent!")

    except Exception as e:
        print("❌ Email error:", e)

def watch_news():
    global last_hash

    print("👀 Watching news.json for real changes...")

    last_hash = get_file_hash()

    while True:
        time.sleep(5)  # check every 5 sec

        new_hash = get_file_hash()

        if new_hash == last_hash:
            continue

        last_hash = new_hash

        print("\n🔥 REAL CHANGE DETECTED")

        try:
            with open(NEWS_FILE, "r") as f:
                data = json.load(f)

            if should_send(data):
                send_email(data)

        except Exception as e:
            print("❌ Error:", e)


@app.on_event("startup")
def start_watcher():
    thread = Thread(target=watch_news, daemon=True)
    thread.start()
