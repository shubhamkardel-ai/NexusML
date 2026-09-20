from config.settings import settings


def health_check() -> dict:
    """Return the current health status of NexusML."""
    return {
        "status": "healthy",
        "application": settings.APP_NAME,
        "environment": settings.APP_ENV,
        "debug": settings.DEBUG,
    }


if __name__ == "__main__":
    result = health_check()

    print("=" * 50)
    print("NexusML Health Check")
    print("=" * 50)

    for key, value in result.items():
        print(f"{key}: {value}")

    print("=" * 50)