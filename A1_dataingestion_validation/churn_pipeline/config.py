from dataclasses import dataclass, field
from pathlib import Path
import os

@dataclass(frozen=True)
    
#frozen = true: make the dataclass immutable after creation
#Once Settings() is constructed, nobody can accidentally so
# Settings.data_url = "something else " halfway througgh a pipeline run.
# This bites you badly in production if mutable - a background thread or a careless line 400 lines down silently changes cnofig,
# and now half your pipeline ran with different setting than other half

class Settings:
    """
    Central configuration for the churn pipeline.
    Every path, URL and threshold lives here - nowhere else in the codebase.
    WHY: when this pipeline runs in docker, in a cron job, and on your laptop
    , only the environment variables change.  THE code never does
    """

    data_url: str = os.environ.get(
        "CHURN_DATA_URL",
        "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
    )
    #os.environ.get(key, default): read from environment variable if set,
    # otherwise use the default. This is the pattern for config in production.
    #In Docker: -e CHURN_DATA_URL = https:// smtg .. overrides this
    #on your laptop: falls back to the public url. same code, diff environment  
    
    raw_data_dir: Path = Path(os.environ.get("DATA_DIR", "./data/raw"))
    processed_data_dir: Path = Path(os.environ.get("PROCESSED_DIR","./data/processed"))
    artifacts_dir: Path = Path(os.environ.get("ARTIFACTS_DIR","./artifacts"))
    
    target_column: str = "Churn"
    id_column: str = "customerID"
    
    #Validation thresholds - the pipeline REFUSES to proceed if these are violated.
    max_missing_fraction: float = 0.4
    #if any column is more than 40 percent missing, something is structurally wrong
    #with the data source - stop and alert, dont silently impute and continue
    # Step 1 - Look at the expected data volume
    #
    # Suppose you're building a daily churn pipeline.
    #
    # Yesterday: 98,500 rows
    #
    # Past 30 days: 97k, 101k, 99k, 98k, 100k
    #
    # Then suddenly: Today: 1,200 rows
    #
    # You immediately know something is wrong.
    #
    # So you can calculate a historical lower bound:
    # historical minimum ~= 95,000
    #
    # Then set a safety threshold, e.g.:
    # min_dataset_rows = 80_000
    #
    # Now you're not randomly choosing 1000.

    min_columns: int = 10
    max_missing_ratio: float = 0.20
    min_target_class_count: int = 50

    min_rows: int = 1000
    #IF the source returns fewer rows than this, the data source is probably broken
    
    random_seed: int = 42
    
    def __post_init__(self):
        # this runs after the dataclass fields are set
        #we use it to create directories immediately -fail fast if the filesystem is read-only or permissions are wrong
        
        self.raw_data_dir.mkdir(parents=True, exist_ok = True)
        self.processed_data_dir.mkdir(parents = True, exist_ok=True)
        self.artifacts_dir.mkdir(parents =True, exist_ok = True)
        
settings = Settings()
# Module-level singleton. Import this everywhere: `from config import settings`.
# Every file in the codebase reads settings.X — never a hardcoded path again.