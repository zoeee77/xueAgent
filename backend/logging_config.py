import logging
import sys
from contextvars import ContextVar
from logging import Filter, LogRecord

# Context variable for per-request trace IDs
trace_id_ctx: ContextVar[str] = ContextVar("trace_id", default="")


class TraceIdFilter(Filter):
    """Logging filter that injects trace_id from the current context into each log record."""

    def filter(self, record: LogRecord) -> bool:
        record.trace_id = trace_id_ctx.get() or "no-trace"
        return True


def setup_logging(level: int = logging.INFO) -> None:
    """Configure structured logging with trace_id support."""
    log_format = "[%(asctime)s] [%(levelname)s] [trace_id:%(trace_id)s] %(message)s"

    formatter = logging.Formatter(log_format, datefmt="%Y-%m-%d %H:%M:%S")

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    handler.addFilter(TraceIdFilter())

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.addHandler(handler)
