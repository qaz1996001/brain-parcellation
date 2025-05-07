# tests/test_backup_config.py
from sqlalchemy.orm import Session
from app.crud import backup_config_crud
from app.schemas import BackupConfigCreate, BackupConfigUpdate
from app.models import BackupConfig


def test_create_backup_config(db: Session):
    """
    Test creating a new backup configuration.
    """
    # Prepare test data
    config_data = BackupConfigCreate(
        config_key="test_config_key",
        config_value="test_config_value",
        description="Test configuration"
    )

    # Create configuration
    created_config = backup_config_crud.create(db, obj_in=config_data)

    # Assertions
    assert created_config is not None
    assert created_config.config_key == "test_config_key"
    assert created_config.config_value == "test_config_value"
    assert created_config.description == "Test configuration"


def test_get_backup_config_by_key(db: Session):
    """
    Test retrieving a backup configuration by its key.
    """
    # Create a test configuration first
    config_data = BackupConfigCreate(
        config_key="retrieval_test_key",
        config_value="retrieval_test_value",
        description="Retrieval test configuration"
    )
    created_config = backup_config_crud.create(db, obj_in=config_data)

    # Retrieve the configuration
    retrieved_config = backup_config_crud.get_by_key(db, config_key="retrieval_test_key")

    # Assertions
    assert retrieved_config is not None
    assert retrieved_config.config_key == "retrieval_test_key"
    assert retrieved_config.config_value == "retrieval_test_value"


def test_update_backup_config(db: Session):
    """
    Test updating an existing backup configuration.
    """
    # Create initial configuration
    config_data = BackupConfigCreate(
        config_key="update_test_key",
        config_value="initial_value",
        description="Initial description"
    )
    created_config = backup_config_crud.create(db, obj_in=config_data)

    # Prepare update data
    update_data = BackupConfigUpdate(
        config_value="updated_value",
        description="Updated description"
    )

    # Update configuration
    updated_config = backup_config_crud.update(
        db,
        db_obj=created_config,
        obj_in=update_data
    )

    # Assertions
    assert updated_config.config_value == "updated_value"
    assert updated_config.description == "Updated description"
    assert updated_config.config_key == "update_test_key"


def test_upsert_backup_config(db: Session):
    """
    Test upserting (inserting or updating) a backup configuration.
    """
    # First upsert - should create
    config1 = backup_config_crud.upsert(
        db,
        config_key="upsert_test_key",
        config_value="initial_value",
        description="Initial description"
    )

    # Assertions for first upsert
    assert config1.config_key == "upsert_test_key"
    assert config1.config_value == "initial_value"
    assert config1.description == "Initial description"

    # Second upsert - should update
    config2 = backup_config_crud.upsert(
        db,
        config_key="upsert_test_key",
        config_value="updated_value",
        description="Updated description"
    )

    # Assertions for second upsert
    assert config2.config_key == "upsert_test_key"
    assert config2.config_value == "updated_value"
    assert config2.description == "Updated description"
    assert config2.config_id == config1.config_id  # Ensure same record was updated


def test_delete_backup_config(db: Session):
    """
    Test deleting a backup configuration.
    """
    # Create a configuration to delete
    config_data = BackupConfigCreate(
        config_key="delete_test_key",
        config_value="delete_test_value",
        description="Delete test configuration"
    )
    created_config = backup_config_crud.create(db, obj_in=config_data)

    # Delete the configuration
    deleted_config = backup_config_crud.remove(db, id=created_config.config_id)

    # Assertions
    assert deleted_config is not None
    assert deleted_config.config_key == "delete_test_key"

    # Verify config is actually deleted
    retrieved_config = backup_config_crud.get(db, id=created_config.config_id)
    assert retrieved_config is None