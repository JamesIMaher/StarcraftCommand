"""sc2rl package.

Sets PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python before anything else runs.
pysc2's bundled s2clientprotocol/*_pb2.py files were generated with an old
protoc and raise ("Descriptors cannot be created directly") under the
protobuf version tensorboard requires (>=6.31.1); forcing the pure-Python
protobuf backend lets both coexist. This has to happen before the first
import of google.protobuf anywhere in the process (the backend is locked in
at that point), so it lives here rather than in whichever submodule happens
to import pysc2 first -- any `import sc2rl...` runs this __init__ before its
submodules, which is the one ordering guarantee we can rely on.

Caveat: this only helps if nothing imports `pysc2` directly before importing
anything from `sc2rl` in the same process -- e.g. tests/conftest.py sets the
same env var for exactly that reason.
"""

import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
