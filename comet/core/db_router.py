import asyncio
import contextvars
import socket
from contextlib import contextmanager
from typing import List, Optional, Sequence
from urllib.parse import quote_plus

from databases import Database
from sqlalchemy.engine.url import make_url

from comet.core.logger import logger


class ReplicaAwareDatabase:
    """Routes read queries to replicas while keeping writes on the primary."""

    def __init__(
        self,
        primary: Database,
        replicas: Optional[Sequence[Database]] = None,
        force_ipv4: bool = False,
    ):
        self._primary = primary
        self._configured_replicas = list(replicas or [])
        self._force_ipv4 = force_ipv4
        self._active_replicas: List[Database] = []
        self._replica_index = 0
        self._transaction_depth = contextvars.ContextVar(
            "comet_db_replica_tx_depth", default=0
        )
        self._force_primary_context = contextvars.ContextVar(
            "comet_db_replica_force_primary", default=False
        )

    @property
    def has_replicas(self):
        return bool(self._active_replicas)

    @property
    def is_connected(self):
        return self._primary.is_connected

    async def connect(self):
        # Handle IP resolution override
        if self._force_ipv4:
            self._primary = await self._resolve_and_recreate(self._primary)
            self._configured_replicas = [
                await self._resolve_and_recreate(replica)
                for replica in self._configured_replicas
            ]

        resolved_ip = await self._wait_for_dns(self._primary)
        if resolved_ip:
            # Reconstruct the primary database instance with the resolved IP
            # This bypasses asyncpg's internal resolution which seems flaky
            # IMPORTANT: Use .replace() on the URL object to preserve credentials.
            # Convert to string forces masking (***), which caused authentication failures.
            current_url = self._primary.url
            if current_url.hostname != resolved_ip:
                logger.log("DATABASE", f"Forcing connection to IP: {resolved_ip}")
                
                # Manual URL reconstruction to guarantee credentials are preserved
                # We explicitely rebuild the string to avoid library masking (***) issues
                user_part = ""
                if current_url.username:
                    user = quote_plus(current_url.username)
                    user_part = f"{user}"
                    if current_url.password:
                        pwd = quote_plus(current_url.password)
                        user_part += f":{pwd}"
                    user_part += "@"

                port_part = f":{current_url.port}" if current_url.port else ""
                path_part = f"/{current_url.database}" if current_url.database else ""
                query_part = f"?{current_url.query}" if current_url.query else ""
                
                new_conn_str = f"{current_url.scheme}://{user_part}{resolved_ip}{port_part}{path_part}{query_part}"
                
                self._primary = Database(new_conn_str)

        # Retry logic for primary database connection
        max_retries = 5
        for attempt in range(max_retries):
            try:
                await self._primary.connect()
                break
            except (socket.gaierror, OSError) as e:
                if attempt == max_retries - 1:
                    logger.error(f"Failed to connect to primary database after {max_retries} attempts: {e}")
                    raise e
                
                wait_time = 2 ** attempt  # Exponential backoff: 1, 2, 4, 8, 16 seconds
                logger.warning(f"Database connection failed ({e}), retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)

        healthy_replicas: List[Database] = []
        for replica in self._configured_replicas:
            try:
                await replica.connect()
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.log(
                    "DATABASE",
                    f"Read replica connection failed ({getattr(replica, 'url', 'replica')}): {exc}",
                )
            else:
                healthy_replicas.append(replica)

        self._active_replicas = healthy_replicas

        if self._active_replicas:
            logger.log(
                "DATABASE",
                f"Read replicas enabled ({len(self._active_replicas)} healthy)",
            )

    async def disconnect(self):
        for db in [self._primary, *self._configured_replicas]:
            if not db.is_connected:
                continue
            try:
                await db.disconnect()
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.log("DATABASE", f"Error disconnecting database: {exc}")

    def transaction(self, *args, **kwargs):
        primary_transaction = self._primary.transaction(*args, **kwargs)
        return _ReplicaAwareTransaction(self, primary_transaction)

    async def execute(self, query, values=None, *, force_primary: bool = False):
        return await self._primary.execute(query, values)

    async def execute_many(self, query, values, *, force_primary: bool = False):
        return await self._primary.execute_many(query, values)

    async def fetch_all(self, query, values=None, *, force_primary: bool = False):
        return await self._run_read("fetch_all", force_primary, query, values)

    async def fetch_one(self, query, values=None, *, force_primary: bool = False):
        return await self._run_read("fetch_one", force_primary, query, values)

    async def fetch_val(
        self, query, values=None, column: int = 0, *, force_primary: bool = False
    ):
        return await self._run_read("fetch_val", force_primary, query, values, column)

    def _should_use_primary(self, explicit_force: bool):
        if explicit_force or self._force_primary_context.get():
            return True

        if not self._active_replicas:
            return True

        if self._transaction_depth.get() > 0:
            return True

        return False

    def _next_replica(self):
        replica = self._active_replicas[
            self._replica_index % len(self._active_replicas)
        ]
        self._replica_index = (self._replica_index + 1) % len(self._active_replicas)
        return replica

    async def _run_read(self, method_name: str, force_primary: bool, *args):
        target = (
            self._primary
            if self._should_use_primary(force_primary)
            else self._next_replica()
        )

        method = getattr(target, method_name)
        try:
            return await method(*args)
        except asyncio.CancelledError:  # pragma: no cover - propagate cancellations
            raise
        except Exception as exc:
            if target is not self._primary and self._primary.is_connected:
                logger.log(
                    "DATABASE",
                    f"Replica {method_name} failed, retrying on primary: {exc}",
                )
                fallback = getattr(self._primary, method_name)
                return await fallback(*args)
            raise

    @contextmanager
    def force_primary(self):
        token = self._force_primary_context.set(True)
        try:
            yield self
        finally:
            self._force_primary_context.reset(token)

    async def _resolve_and_recreate(self, db_instance: Database) -> Database:
        """
        Resolves the hostname of the given Database instance to an IP address.
        If resolution succeeds and changes the host, returns a NEW Database instance.
        Otherwise, returns the original instance.
        """
        try:
            original_url_str = str(db_instance.url)
            url_obj = make_url(original_url_str)

            if not url_obj.host:
                return db_instance

            # Resolve
            logger.log("DATABASE", f"Resolving hostname for DB: {url_obj.host}")
            resolved_ip = socket.gethostbyname(url_obj.host)

            if resolved_ip == url_obj.host:
                return db_instance  # No change

            logger.log(
                "DATABASE", f"Resolved {url_obj.host} -> {resolved_ip} (forcing IPv4)"
            )
            # Reconstruct URL with IP
            url_obj = url_obj.set(host=resolved_ip)
            new_url_str = url_obj.render_as_string(hide_password=False)

            return Database(new_url_str)
        except Exception as e:
            logger.warning(
                f"Failed to resolve hostname for {db_instance.url}: {e}. Keeping original."
            )
            return db_instance

    def __getattr__(self, item):
        return getattr(self._primary, item)

    async def _wait_for_dns(self, db: Database):
        try:
            url = make_url(str(db.url))
            if not url.host:
                return

            logger.log("DATABASE", f"Checking DNS resolution for {url.host}...")
            
            # Retry indefinitely (or for a very long time) until DNS is ready
            # This prevents the app from crashing during startup race conditions
            retries = 0
            while True:
                try:
                    # Use blocking call to verify system resolver is ready
                    ip = socket.gethostbyname(url.host)
                    logger.log("DATABASE", f"DNS resolved for {url.host} -> {ip}")
                    return ip
                except socket.gaierror:
                    retries += 1
                    wait = min(2 * retries, 15)  # Cap wait at 15s
                    logger.warning(f"DNS resolution for {url.host} failed (Attempt {retries}), retrying in {wait}s...")
                    await asyncio.sleep(wait)
                    
                    if retries > 20: # ~5 minutes max wait then try anyway
                         logger.error("DNS resolution timing out, attempting connection anyway...")
                         return None
        except Exception as e:
            logger.warning(f"Error during DNS pre-check: {e}")
            return None



class _ReplicaAwareTransaction:
    def __init__(self, router: ReplicaAwareDatabase, transaction_cm):
        self._router = router
        self._transaction_cm = transaction_cm
        self._token = None

    async def __aenter__(self):
        current_depth = self._router._transaction_depth.get()
        self._token = self._router._transaction_depth.set(current_depth + 1)
        return await self._transaction_cm.__aenter__()

    async def __aexit__(self, exc_type, exc, tb):
        try:
            return await self._transaction_cm.__aexit__(exc_type, exc, tb)
        finally:
            if self._token is not None:
                self._router._transaction_depth.reset(self._token)
