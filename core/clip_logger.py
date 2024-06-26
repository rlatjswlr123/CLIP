import logging
from pathlib import Path
import datetime


def log_setting(log_save_dir):
    # log_path = Path("logs")
    # log_path.mkdir(exist_ok=True)
    log_dir = Path(log_save_dir)
    log_dir.parent.mkdir(exist_ok=True)
    log_dir.mkdir(exist_ok=True)
    log_filename = f"log_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.txt"
    log_filepath = log_dir / log_filename
    logging.basicConfig(filename=log_filepath, level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True)  # 추가된 부분
