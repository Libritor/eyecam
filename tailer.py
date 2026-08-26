"""Windows-safe incremental CSV reader.

Follows a CSV file that another process is actively writing (line-buffered).
Handles: the file not existing yet (open retried on each poll), partial lines
at EOF (buffered until the newline arrives), and appends becoming visible on
re-read of the same handle.
"""

import csv
import io
import os


class CsvTail:
    def __init__(self, path, skip_header=True):
        self.path = path
        self.skip_header = skip_header
        self._fh = None
        self._buf = ""
        self._header_done = not skip_header
        self._nread = 0

    def _reset(self):
        if self._fh is not None:
            self._fh.close()
        self._fh = None
        self._buf = ""
        self._header_done = not self.skip_header
        self._nread = 0

    def poll(self):
        """Return a list of parsed rows (lists of str) appended since last poll."""
        if self._fh is not None:
            # Writer re-created the file (mode "w" on a reused session dir):
            # our offset is past the new EOF — start over. ASCII content and
            # newline="" keep len(chunk) == bytes consumed.
            try:
                if os.path.getsize(self.path) < self._nread:
                    self._reset()
            except OSError:
                self._reset()
        if self._fh is None:
            if not os.path.exists(self.path):
                return []
            try:
                self._fh = open(self.path, "r", newline="")
            except OSError:
                return []  # transiently locked (AV/indexer); retry next poll
        try:
            chunk = self._fh.read()
        except OSError:
            return []
        if not chunk:
            return []
        self._nread += len(chunk)
        self._buf += chunk
        # Only parse complete lines; keep any trailing fragment.
        last_nl = self._buf.rfind("\n")
        if last_nl < 0:
            return []
        complete, self._buf = self._buf[:last_nl + 1], self._buf[last_nl + 1:]
        rows = list(csv.reader(io.StringIO(complete)))
        if rows and not self._header_done:
            rows = rows[1:]
            self._header_done = True
        return [r for r in rows if r]

    def close(self):
        if self._fh is not None:
            self._fh.close()
            self._fh = None
