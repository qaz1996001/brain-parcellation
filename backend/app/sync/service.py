from advanced_alchemy.extensions.fastapi import (
    AdvancedAlchemy,
    AsyncSessionConfig,
    SQLAlchemyAsyncConfig,
    base,
    filters,
    repository,
    service,
)
from .model import DCOPEventModel


class DCOPEventDicomService(service.SQLAlchemyAsyncRepositoryService[DCOPEventModel]):
    """Author repository."""

    class Repo(repository.SQLAlchemyAsyncRepository[DCOPEventModel]):
        """Author repository."""

        model_type = DCOPEventModel

    repository_type = Repo
