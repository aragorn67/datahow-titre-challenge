"""Column names and prefix conventions of the supplied CSV files.

Every column name used anywhere in the package is declared here, so that a change to
the data format is a change to one file. Nothing downstream hardcodes a column name.

The dataset uses a prefix convention, also used by the OpenAPI payload:

    Z:  process design scalars, constant within a run
    W:  control profiles over time (temperature, pH, feeds)
    X:  measured observations
    Y:  the target
"""

from collections.abc import Sequence

# --- Index columns -----------------------------------------------------------------

ROW_ID_COLUMN = "RowID"
EXPERIMENT_COLUMN = "Exp"
TIME_COLUMN = "Time[day]"

# --- Prefixes ----------------------------------------------------------------------

DESIGN_SCALAR_PREFIX = "Z:"
CONTROL_PROFILE_PREFIX = "W:"
OBSERVATION_PREFIX = "X:"
TARGET_PREFIX = "Y:"

# --- Target ------------------------------------------------------------------------

TARGET_COLUMN = "Y:Titer"

# --- Design scalars set once per run and never varied in time ----------------------

DESIGN_DISSOLVED_OXYGEN = "Z:DO"
DESIGN_STIRRING = "Z:Stir"

# --- Design scalars used when reconstructing the control profiles ------------------

DESIGN_FEED_START = "Z:FeedStart"
DESIGN_FEED_END = "Z:FeedEnd"
DESIGN_FEED_RATE_GLUCOSE = "Z:FeedRateGlc"
DESIGN_FEED_RATE_GLUTAMINE = "Z:FeedRateGln"
DESIGN_PH_START = "Z:phStart"
DESIGN_PH_END = "Z:phEnd"
DESIGN_PH_SHIFT = "Z:phShift"
DESIGN_TEMPERATURE_START = "Z:tempStart"
DESIGN_TEMPERATURE_END = "Z:tempEnd"
DESIGN_TEMPERATURE_SHIFT = "Z:tempShift"
DESIGN_EXPERIMENT_DURATION = "Z:ExpDuration"

# --- Control profiles --------------------------------------------------------------

CONTROL_TEMPERATURE = "W:temp"
CONTROL_PH = "W:pH"
CONTROL_FEED_GLUCOSE = "W:FeedGlc"
CONTROL_FEED_GLUTAMINE = "W:FeedGln"

# --- Observations ------------------------------------------------------------------

OBSERVATION_VIABLE_CELL_DENSITY = "X:VCD"
OBSERVATION_GLUCOSE = "X:Glc"
OBSERVATION_GLUTAMINE = "X:Gln"
OBSERVATION_AMMONIA = "X:Amm"
OBSERVATION_LACTATE = "X:Lac"
OBSERVATION_LYSED_CELLS = "X:Lysed"


def columns_with_prefix(columns: Sequence[str], prefix: str) -> list[str]:
    """Return the members of ``columns`` beginning with ``prefix``, order preserved.

    Args:
        columns: column names to filter, e.g. ``dataframe.columns``.
        prefix: one of DESIGN_SCALAR_PREFIX | CONTROL_PROFILE_PREFIX |
            OBSERVATION_PREFIX | TARGET_PREFIX.
    """
    return [column for column in columns if column.startswith(prefix)]
