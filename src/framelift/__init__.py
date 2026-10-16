"""framelift: extract structured data from screen recordings and screenshots with vision LLMs.

The public API is `Profile` and `Pipeline` (plus the `Observation` and `Item` records they
produce).
Everything under ``framelift._internal`` may change in any release.
"""

from importlib.metadata import version as _version

from framelift.item import Item
from framelift.observation import Observation
from framelift.pipeline import Pipeline, RunResult
from framelift.profile import Profile, Validator

__all__ = ["Item", "Observation", "Pipeline", "Profile", "RunResult", "Validator", "__version__"]

__version__ = _version("framelift")
