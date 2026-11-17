"""Shared, durable reservations before any paid operation is dispatched."""
from contextlib import contextmanager
from datetime import datetime, timezone
from decimal import Decimal, ROUND_CEILING
import json
import os
from pathlib import Path
import resource
import shutil
import sqlite3
from threading import Lock
import uuid

from sft_data import ARTIFACTS

DEFAULT_LEDGER = ARTIFACTS / 'rl-eight-config-2026-10-23/usage.sqlite3'
_SCALE = 1_000_000_000
_transaction_lock = Lock()


class BudgetExceeded(RuntimeError):
    pass


def resource_snapshot(path):
    parent = Path(path).absolute().parent
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    value = {'ledger_path': str(Path(path).absolute()), 'nofile_soft': soft, 'nofile_hard': hard,
             'ledger_parent_writable': os.access(parent, os.W_OK | os.X_OK)}
    try:
        descriptors = '/proc/self/fd' if Path('/proc/self/fd').is_dir() else '/dev/fd'
        # Directory enumeration includes its own transient descriptor.
        value['open_fds'] = len(os.listdir(descriptors)) - 1
    except OSError as error:
        value.update(open_fds=None, fd_count_error=str(error))
    try:
        value['ledger_free_bytes'] = shutil.disk_usage(parent).free
    except OSError as error:
        value.update(ledger_free_bytes=None, disk_error=str(error))
    return value


def _units(value):
    number = Decimal(str(value))
    if not number.is_finite() or number < 0:
        raise ValueError('Budget amounts must be finite and nonnegative')
    return int((number * _SCALE).to_integral_value(rounding=ROUND_CEILING))


class Budget:
    def __init__(self, path=DEFAULT_LEDGER, run_id=None, ceiling_usd=None, reserve_usd=0):
        self.path = Path(path)
        self.run_id = run_id
        self.ceiling = None if ceiling_usd is None else _units(ceiling_usd)
        self.reserve_units = _units(reserve_usd)
        if self.ceiling is not None and self.reserve_units > self.ceiling:
            raise ValueError('Contingency exceeds total budget')
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._transaction() as connection:
            connection.execute("""CREATE TABLE IF NOT EXISTS metadata (
                id INTEGER PRIMARY KEY CHECK (id = 1), ceiling_units INTEGER NOT NULL,
                contingency_units INTEGER NOT NULL, charged_units INTEGER NOT NULL,
                halted INTEGER NOT NULL DEFAULT 0)""")
            connection.execute("""CREATE TABLE IF NOT EXISTS reservations (
                id TEXT PRIMARY KEY, operation TEXT NOT NULL, run_id TEXT,
                reserved_units INTEGER NOT NULL, charged_units INTEGER NOT NULL,
                status TEXT NOT NULL, created_utc TEXT NOT NULL, settled_utc TEXT)""")
            connection.execute('INSERT OR IGNORE INTO metadata VALUES (1, ?, ?, 0, 0)',
                               (-1 if self.ceiling is None else self.ceiling, self.reserve_units))
            metadata = connection.execute('SELECT * FROM metadata WHERE id = 1').fetchone()
            if (metadata['ceiling_units'], metadata['contingency_units']) != (-1 if self.ceiling is None else self.ceiling, self.reserve_units):
                raise ValueError('Existing budget limits differ; refusing to reset shared budget')

    @contextmanager
    def _transaction(self):
        with _transaction_lock:
            connection = None
            try:
                connection = sqlite3.connect(self.path, timeout=60, isolation_level=None)
                connection.row_factory = sqlite3.Row
                connection.execute('PRAGMA synchronous = FULL')
                connection.execute('BEGIN IMMEDIATE')
                yield connection
                connection.execute('COMMIT')
            except BaseException as error:
                if isinstance(error, sqlite3.Error):
                    error.add_note('SQLite budget diagnostics: ' + json.dumps({
                        **resource_snapshot(self.path),
                        'sqlite_errorcode': getattr(error, 'sqlite_errorcode', None),
                        'sqlite_errorname': getattr(error, 'sqlite_errorname', None)}))
                if connection is not None and connection.in_transaction:
                    connection.execute('ROLLBACK')
                raise
            finally:
                if connection is not None:
                    connection.close()

    def reserve(self, operation, upper_usd):
        amount = _units(upper_usd)
        with self._transaction() as connection:
            metadata = connection.execute('SELECT * FROM metadata WHERE id = 1').fetchone()
            charged = metadata['charged_units']
            if metadata['halted']:
                raise BudgetExceeded('Budget halted after a cost bound violation; reconciliation required')
            if self.ceiling is not None and charged + amount + self.reserve_units > self.ceiling:
                raise BudgetExceeded(f'Budget prevents {operation}: requested ${amount / _SCALE:.6f}; '
                                     f'available ${(self.ceiling - self.reserve_units - charged) / _SCALE:.6f}')
            reservation = uuid.uuid4().hex
            connection.execute('INSERT INTO reservations VALUES (?, ?, ?, ?, ?, ?, ?, NULL)',
                (reservation, operation, self.run_id, amount, amount, 'reserved',
                 datetime.now(timezone.utc).isoformat()))
            connection.execute('UPDATE metadata SET charged_units = charged_units + ? WHERE id = 1', (amount,))
        return reservation

    def settle(self, reservation, upper_usd=None):
        if upper_usd is None:
            return
        amount = _units(upper_usd)
        with self._transaction() as connection:
            row = connection.execute('SELECT * FROM reservations WHERE id = ?', (reservation,)).fetchone()
            if row is None:
                raise KeyError(reservation)
            if row['status'] == 'settled':
                if row['charged_units'] != amount:
                    raise ValueError('Reservation already settled with a different amount')
                return
            exceeded = self.ceiling is not None and amount > row['reserved_units']
            connection.execute("UPDATE reservations SET charged_units = ?, status = 'settled', settled_utc = ? WHERE id = ?",
                               (amount, datetime.now(timezone.utc).isoformat(), reservation))
            connection.execute('UPDATE metadata SET charged_units = charged_units + ?, halted = MAX(halted, ?) WHERE id = 1',
                               (amount - row['charged_units'], int(exceeded)))
        if exceeded:
            raise BudgetExceeded('Reported cost exceeded predispatch bound; stop and reconcile pricing')

    def snapshot(self, include_reservations=False):
        with self._transaction() as connection:
            metadata = dict(connection.execute('SELECT * FROM metadata WHERE id = 1').fetchone())
            snapshot = {**metadata, 'version': 2, 'unit_usd': 1 / _SCALE,
                        'committed_usd': metadata['charged_units'] / _SCALE,
                        'spending_cap_usd': None if self.ceiling is None else self.ceiling / _SCALE,
                        'available_usd': None if self.ceiling is None else (self.ceiling - self.reserve_units - metadata['charged_units']) / _SCALE}
            if include_reservations:
                snapshot['reservations'] = {row['id']: dict(row) for row in connection.execute('SELECT * FROM reservations')}
            return snapshot


def sampling_upper(prompt_tokens, max_tokens, num_samples=1):
    if min(prompt_tokens, max_tokens, num_samples) < 0:
        raise ValueError('Token counts cannot be negative')
    return num_samples * (prompt_tokens * .33 + max_tokens * 1.005) / 1e6


def training_upper(all_tokens, passes=1):
    if all_tokens < 0 or passes < 1:
        raise ValueError('Token counts cannot be negative')
    return all_tokens * passes * .737 / 1e6


def formatter_upper(messages, max_tokens=8192):
    # UTF-8 bytes plus serialized framing conservatively bound input tokens.
    input_bytes = len(json.dumps(messages, ensure_ascii=False).encode('utf-8'))
    return (input_bytes * .27 + max_tokens * .41) / 1e6
