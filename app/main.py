from services.logging_config import configure_logging, get_logger


configure_logging()

logger = get_logger("NexusML")


def health_check():
    logger.info("NexusML application health check executed")

    return {
        "status": "healthy",
        "application": "NexusML",
        "environment": "development",
        "debug": True,
    }


if __name__ == "__main__":
    result = health_check()

    print(result)