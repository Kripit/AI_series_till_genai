import pandas as pd
import requests
import hashlib
import json
import time
from pathlib import Path
from datetime import datetime, timezone
from io import StringIO

from ..config import Settings, settings
from ..logging_setup import logger


class DataLoader:
    """_summary_
    Downloads data from a live source, caches locally and tracks
    exactly which version od the data was used for every run
    
    WHY CACHE? In production you do no want every pipeline run to re-hit
    an external API or data lake -  it is slow, costs money on metered APIs,
    and if source goes down your pipeline still needds to run on the last known- good data
    
    WHY HASH? "Which dataset produced this model?" is on of the most
    common questions in production ML debugginh. A content hash answers it precisely
    - if the hash matches, it is provably the same data
    """
    
    def __init__(self, settings: Settings):
        self.settings = settings  # Store the Settings object inside the loader so every method can access config.
        #settings → DataLoader → self.settings

    def _cache_path(self, url: str) -> Path:
        url_hash = hashlib.md5(url.encode()).hexdigest()[:10]
        #Turn the URL into a short, deterministic fingerprint.
        return self.settings.raw_data_dir / f"raw_{url_hash}.csv"
        # Take the raw-data folder + create the filename raw_<hash>.csv and return that path
        
#self.settings.raw_data_dir
#       ↓
#data/raw/

#url_hash
#        ↓
#abc123def4

#f"raw_{url_hash}.csv"
#        ↓
#raw_abc123def4.csv

    def load(self, force_refresh: bool = False) -> pd.DataFrame:
        """_summary_

        Load data. Uses local cache unless force_refresh = True or cache missing
        
        Args:
            force_refresh (bool, optional): _description_. Defaults to False.

        Returns:
            pd.DataFrame: _description_
        """
        cache_path = self._cache_path(self.settings.data_url)
        
        if cache_path.exists() and not force_refresh:
            logger.info(f"Loading from cache: {cache_path}")
            df = pd.read_csv(cache_path)
        
        else:
            logger.info(f"Downloading from source {self.settings.data_url}")
            df = self._download()
            df.to_csv(cache_path, index = False)
            logger.info(f"Cached to: {cache_path}")
        content_hash = self._compute_hash(df)
        logger.info(f"Dataset hash: {content_hash} | shape: {df.shape}")
        
        self._write_manifest(df, content_hash, cache_path)
        
        return df

    
    def _download(self) -> pd.DataFrame:
        """
        Download with retries and a real timeout — production APIs and
        data sources are unreliable, and a pipeline that hangs forever
        on a stalled connection is a real incident, not a hypothetical.
        """
        for attempt in range(1, 4):
            try:
                response = requests.get(self.settings.data_url, timeout=30)
                response.raise_for_status()
                # raise_for_status(): raises an exception on 4xx/5xx responses.
                # Without this, a 404 or 500 page (which is valid text!) would
                # get parsed as if it were CSV data - and fail confusingly
                # three steps later instead of right here where the real
                # problem is.
                return pd.read_csv(StringIO(response.text))

            except (requests.RequestException, pd.errors.ParserError) as error:
                logger.warning(f"Download attempt {attempt}/3 failed: {error}")
                if attempt == 3:
                    raise RuntimeError(
                        f"Failed to download data after 3 attempts from "
                        f"{self.settings.data_url}"
                    ) from error

                time.sleep(2**attempt)
                # Exponential backoff: wait 2s, then 4s, then give up.
                # This is the standard retry pattern for any network call
                # in production - you will write this exact loop constantly.

    def _compute_hash(self, df: pd.DataFrame) -> str:
         # Hash the actual content, not the file — so two runs that
        # download identical data always get the identical hash,
        # regardless of download timestamp or file metadata.
        content = pd.util.hash_pandas_object(df, index=True).values
        return hashlib.sha256(content.tobytes()).hexdigest()[:16]
    
    def _write_manifest(self, df: pd.DataFrame, content_hash: str, cache_path: Path):
        """
        A manifest file records exactly what data was used, when.
        This is a lightweight substitute for full data versioning tools
        like DVC — good enough for most teams, and the concept is identical.
        """
        manifest = {
            "loaded_at": datetime.now(timezone.utc).isoformat(),
            "source_url": self.settings.data_url,
            "cache_path": str(cache_path),
            "content_hash": content_hash,
            "n_rows": len(df),
            "n_cols": len(df.columns),
            "columns": list(df.columns),
        }
        manifest_path = self.settings.processed_data_dir / "data_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as file:
            json.dump(manifest, file, indent=2)
        logger.info(f"Manifest written: {manifest_path}")


if __name__ == "__main__":
    loader = DataLoader(settings)
    df = loader.load()
    print(df.shape)
    print(df.dtypes)