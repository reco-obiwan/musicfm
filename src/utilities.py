"""yaml config to dict"""

import os
from datetime import datetime

from pathlib import Path
import yaml


class Config:
    """
    yaml config parser class
    """

    def __init__(self, config_path: str) -> None:
        self.config = {}
        with open(config_path) as file:
            for key, value in yaml.load(file, Loader=yaml.FullLoader).items():
                self.config[key] = value

    def __getitem__(self, key):
        return self.config[key]

    def __setitem__(self, key, value):
        self.config[key] = value

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return str(self.config)




def _get_files_from_directory(directory_path):
    """디렉토리에서 .pt 확장자를 가진 파일 리스트를 가져옵니다."""
    try:
        return [f.name for f in Path(directory_path).glob("*.pt") if f.is_file()]
    except Exception:
        return []


def find_most_recent_file(directory_path):
    """yyyymmddhh 형식의 날짜가 포함된 파일 중 가장 최근 파일을 반환합니다."""
    files = _get_files_from_directory(directory_path)
    date_format = "%Y%m%d%H"
    file_date_map = {}

    for file in files:
        try:
            # 파일명에서 날짜 부분 추출
            date_time_str = file.split("_")[-1].replace(".pt", "")
            file_date_map[file] = datetime.strptime(date_time_str, date_format)
        except ValueError:
            continue  # 날짜 형식이 맞지 않으면 무시

    if file_date_map:
        # 가장 최근 날짜의 파일 찾기
        most_recent_file = max(file_date_map, key=file_date_map.get)
        return most_recent_file
    else:
        return None  # 유효한 파일이 없을 경우
