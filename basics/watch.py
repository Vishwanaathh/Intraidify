import time
import json
import hashlib
import os
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

FILE_TO_WATCH = "news.json"

last_hash = None  


def get_file_hash():
    try:
        with open(FILE_TO_WATCH, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()
    except:
        return None


class JSONChangeHandler(FileSystemEventHandler):
    def on_modified(self, event):
        global last_hash

        if os.path.basename(event.src_path) == FILE_TO_WATCH:
            new_hash = get_file_hash()


            if new_hash == last_hash:
                return

            last_hash = new_hash

            print("\n REAL JSON CHANGE DETECTED!")

            try:
                with open(FILE_TO_WATCH, "r", encoding="utf-8") as f:
                    data = json.load(f)

               

            except Exception as e:
                print("Error reading JSON:", e)


if __name__ == "__main__":

    last_hash = get_file_hash()

    event_handler = JSONChangeHandler()
    observer = Observer()

    observer.schedule(event_handler, path=".", recursive=False)
    observer.start()

    print(" Watching for REAL changes in news.json...\n")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()

    observer.join()