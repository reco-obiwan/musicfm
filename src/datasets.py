import os
import random

import shutil
import requests
import torch
import torchaudio
from torch.utils.data import Dataset
from transformers.utils import logging

logging.set_verbosity_info()
logger = logging.get_logger("transformers")

workdir = os.environ["WORKDIR"]


class DatasetBase(Dataset):
    def __init__(self, config):
        super().__init__()

        self.local_music_path = os.path.join(workdir, "music")

        self.config = config
        self.track_download_url = self.config["datasets"]["track_download_url"]
        self.track_list_path = self.config["datasets"]["track_list"]

        with open(self.track_list_path, "r") as f:
            self.track_list = [line.strip() for line in f]

        self.total_track_cnt = len(self.track_list)

        self.session = requests.Session()

    def __len__(self):
        return len(self.track_list)

    def __getitem__(self, _):
        tracks = list(range(self.total_track_cnt))

        for _ in range(20):  # Limit attempts to 10
            idx = random.choice(tracks)
            wav = self._get_wav(self.track_list[idx])
            if wav is not None:
                return wav

        raise RuntimeError(
            "Unable to retrieve a valid waveform from the dataset after 10 attempts"
        )

    def _get_wav(self, track_id, freq=24000, seconds=30):
        path = self._get_music_path(track_id)

        if path is None:
            return None

        waveform, orig_freq = torchaudio.load(path)
        # 스테레오를 모노로 변환
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        resampler = torchaudio.transforms.Resample(orig_freq=orig_freq, new_freq=freq)
        resampled_waveform = resampler(waveform)

        length = seconds * freq
        start = random.randint(0, resampled_waveform.shape[1] - length)
        end = start + length

        output_waveform = resampled_waveform[:, start:end].squeeze(0)
        logger.debug("track_id: %s, shape: %s", track_id, output_waveform.shape)

        return output_waveform

    def _get_music_path(self, track_id):
        for ext in ["aac", "m4a"]:
            music_path = os.path.join(self.local_music_path, f"{track_id}.{ext}")
            if os.path.exists(music_path):
                return music_path

        return self._download_track(track_id)

    def _download_track(self, track_id):
        try:
            response = self.session.get(f"{self.track_download_url}/{track_id}")
            response.raise_for_status()

            body = response.json()
            logger.debug("Track metadata retrieved successfully: %s", body)
            
            track_download_url = body["data"]["url"].strip()
            track_extension = track_download_url.rsplit(".", maxsplit=1)[-1]

            track_download_name = os.path.join(
                self.local_music_path, f"{track_id}.{track_extension}"
            )

            if not os.path.exists(track_download_name):
                with self.session.get(
                    track_download_url, stream=True
                ) as download_response:
                    download_response.raise_for_status()  # Automatically raises an exception for HTTP errors
                    with open(track_download_name, "wb") as file:
                        shutil.copyfileobj(download_response.raw, file)
                logger.debug(f"Downloaded track {track_id} successfully")

            return track_download_name
        except requests.RequestException as req_err:
            error_message = f"HTTP request error for track {track_id}: {req_err}"
            logger.debug(error_message)
        except Exception as exc:
            error_message = (
                f"Failed to download {track_id} from {track_download_url}: {exc}"
            )
            logger.warning(error_message)

        return None


class TrainDataset(DatasetBase):
    def __init__(self, config):
        super().__init__(config=config)
        with open(self.track_list_path, "r") as f:
            self.track_list = [line.strip() for line in f]

        

class ValidationDataset(DatasetBase):
    def __init__(self, config, num_samples=10000):
        super().__init__(config=config)
        with open(self.track_list_path, "r") as f:
            c = [line.strip() for line in f]

        self.track_list = random.sample(self.track_list, num_samples)

        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples


# if __name__ == "__main__":
#     db = DatasetBase()
#     track_id = "520736792"
#     logger.info(f"Track {track_id} already exists")
#     db.get_wav(track_id)
