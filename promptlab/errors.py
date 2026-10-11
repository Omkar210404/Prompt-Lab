"""Typed error hierarchy. Provider errors carry a `retryable` flag that the retry loop reads."""


class PromptLabError(Exception):
    """Base class for all expected errors. The CLI prints these without a stack trace."""

    error_type = "error"


class ConfigError(PromptLabError):
    """A problem in an experiment file, a test set, a price table or the environment."""

    error_type = "config_error"


class RunNotFoundError(PromptLabError):
    error_type = "run_not_found"


class ProviderError(PromptLabError):
    """Base class for errors coming from a model provider."""

    error_type = "provider_error"
    retryable = False
    # Set by the retry loop: how many attempts were made before giving up.
    attempts = 1


class ProviderAuthError(ProviderError):
    """401/403. Never retried: a wrong key will not start working."""

    error_type = "auth_error"


class ProviderTimeoutError(ProviderError):
    error_type = "timeout"
    retryable = True


class ProviderRateLimitError(ProviderError):
    error_type = "rate_limited"
    retryable = True

    def __init__(self, message: str, retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after


class ProviderServerError(ProviderError):
    """5xx or a connection failure."""

    error_type = "server_error"
    retryable = True


class ProviderRequestError(ProviderError):
    """Other 4xx (bad model name, bad payload). Retrying the same request will not help."""

    error_type = "bad_request"


class MalformedResponseError(ProviderError):
    """The provider answered 2xx but the body is not a chat completion."""

    error_type = "malformed_response"
