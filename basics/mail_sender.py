
import time
import json
import hashlib
import os
import yagmail
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler


FILE_TO_WATCH = "news.json"

SENDER_EMAIL = ""
APP_PASSWORD = ""

RECIPIENTS = [
    "peoplewatching41@gmail.com",
    "vishwanaathh4@gmail.com",
    "kancherlasriram2006@gmail.com"
    
]


last_hash = None
last_sent_titles = set()

def get_file_hash():
    try:
        with open(FILE_TO_WATCH, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()
    except:
        return None

def send_email(data):
    try:
        yag = yagmail.SMTP(SENDER_EMAIL, APP_PASSWORD)

        content = "<h2>📈 Market News Alert</h2>"

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
            to=RECIPIENTS,
            subject="🚨 Market News Update",
            contents=content
        )

        print("📧 Email sent successfully!")

    except Exception as e:
        print("❌ Email error:", e)

def should_send(data):
    global last_sent_titles

    if not data:
        return False

    # only send if high impact news exists
    if data[0].get("score", 0) < 40:
        print("⚠️ No high-impact news → skipping email")
        return False

    # avoid duplicate emails
    current_titles = {item["title"] for item in data[:5]}

    if current_titles == last_sent_titles:
        print("⚠️ Same news → skipping email")
        return False

    last_sent_titles = current_titles
    return True

class JSONChangeHandler(FileSystemEventHandler):
    def on_modified(self, event):
        global last_hash

        if os.path.basename(event.src_path) == FILE_TO_WATCH:
            new_hash = get_file_hash()

            # ignore if same content
            if new_hash == last_hash:
                return

            last_hash = new_hash

            print("\n🔥 REAL JSON CHANGE DETECTED!")

            try:
                with open(FILE_TO_WATCH, "r", encoding="utf-8") as f:
                    data = json.load(f)

                print(f"Total items: {len(data)}")

                # preview top 3
                for item in data[:3]:
                    print("-", item["title"])

                # ✅ trigger email if conditions met
                if should_send(data):
                    send_email(data)

            except Exception as e:
                print("❌ Error reading JSON:", e)

def watch():
    global last_hash

    last_hash = get_file_hash()

    event_handler = JSONChangeHandler()
    observer = Observer()

    observer.schedule(event_handler, path=".", recursive=False)
    observer.start()

    print("👀 Watching for REAL changes in news.json...\n")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()

    observer.join()


if __name__ == "__main__":
    watch()
