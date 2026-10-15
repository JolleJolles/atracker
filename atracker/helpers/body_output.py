"""Stream optional per-frame geometry without retaining it for a whole video."""
import gzip
import json
import os
import tempfile
import uuid


class BodyWriter:
    def __init__(self, path, metadata):
        self.path = path
        self.run = uuid.uuid4().hex
        fd, self.temporary = tempfile.mkstemp(prefix='.body-', suffix='.tmp', dir=os.path.dirname(path))
        os.close(fd)
        self.stream = gzip.open(self.temporary, 'wt', encoding='utf-8')
        self.write(dict(type='metadata', schema_version=1, tracking_run=self.run, **metadata))

    def write(self, record):
        self.stream.write(json.dumps(record, allow_nan=False) + '\n')

    def commit(self):
        self.stream.close()
        os.replace(self.temporary, self.path)

    def abort(self):
        self.stream.close()
        if os.path.exists(self.temporary):
            os.unlink(self.temporary)
