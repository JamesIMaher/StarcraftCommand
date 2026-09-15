"""Runs before pytest imports any test module. Several test files import
`pysc2.lib.actions` directly (for assertions against real FunctionCall
objects) before importing anything from `sc2rl`, which would otherwise import
google.protobuf before sc2rl/__init__.py gets a chance to set this -- see that
file for why it's needed.
"""

import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
