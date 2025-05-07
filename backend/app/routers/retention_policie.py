# app/routers/retention_policie.py
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List

from app.database import get_db
from app.schemas import RetentionPolicyCreate, RetentionPolicyResponse, RetentionPolicyUpdate
from app.crud import retention_policy_crud

router = APIRouter()


@router.post("/", response_model=RetentionPolicyResponse, status_code=201)
def create_retention_policy(
        policy: RetentionPolicyCreate,
        db: Session = Depends(get_db)
) -> RetentionPolicyResponse:
    """Create a new retention policy."""
    # Check if policy with the same name already exists
    existing_policy = retention_policy_crud.get_by_name(db=db, name=policy.policy_name)
    if existing_policy:
        raise HTTPException(status_code=400, detail="Retention policy with this name already exists")

    return retention_policy_crud.create(db=db, obj_in=policy)


@router.get("/", response_model=List[RetentionPolicyResponse])
def read_retention_policies(
        skip: int = 0,
        limit: int = 100,
        mode: str = None,
        db: Session = Depends(get_db)
) -> List[RetentionPolicyResponse]:
    """Get a list of retention policies."""
    if mode:
        if mode not in ["governance", "compliance"]:
            raise HTTPException(status_code=400, detail="Mode must be either 'governance' or 'compliance'")
        return retention_policy_crud.get_by_mode(db=db, mode=mode, skip=skip, limit=limit)
    return retention_policy_crud.get_multi(db=db, skip=skip, limit=limit)


@router.get("/{policy_id}", response_model=RetentionPolicyResponse)
def read_retention_policy(
        policy_id: int,
        db: Session = Depends(get_db)
) -> RetentionPolicyResponse:
    """Get a specific retention policy by ID."""
    policy = retention_policy_crud.get(db=db, id=policy_id)
    if policy is None:
        raise HTTPException(status_code=404, detail="Retention policy not found")
    return policy


@router.get("/by-name/{policy_name}", response_model=RetentionPolicyResponse)
def read_retention_policy_by_name(
        policy_name: str,
        db: Session = Depends(get_db)
) -> RetentionPolicyResponse:
    """Get a specific retention policy by name."""
    policy = retention_policy_crud.get_by_name(db=db, name=policy_name)
    if policy is None:
        raise HTTPException(status_code=404, detail="Retention policy not found")
    return policy


@router.put("/{policy_id}", response_model=RetentionPolicyResponse)
def update_retention_policy(
        policy_id: int,
        policy: RetentionPolicyUpdate,
        db: Session = Depends(get_db)
) -> RetentionPolicyResponse:
    """Update a specific retention policy."""
    db_policy = retention_policy_crud.get(db=db, id=policy_id)
    if db_policy is None:
        raise HTTPException(status_code=404, detail="Retention policy not found")

    # If name is being updated, check if it already exists
    if policy.policy_name and policy.policy_name != db_policy.policy_name:
        existing_policy = retention_policy_crud.get_by_name(db=db, name=policy.policy_name)
        if existing_policy:
            raise HTTPException(status_code=400, detail="Retention policy with this name already exists")

    return retention_policy_crud.update(db=db, db_obj=db_policy, obj_in=policy)


@router.delete("/{policy_id}", response_model=RetentionPolicyResponse)
def delete_retention_policy(
        policy_id: int,
        db: Session = Depends(get_db)
) -> RetentionPolicyResponse:
    """Delete a specific retention policy."""
    policy = retention_policy_crud.get(db=db, id=policy_id)
    if policy is None:
        raise HTTPException(status_code=404, detail="Retention policy not found")

    # Check if the policy is in use by any object retention entries
    # This would require a more complex check in a real implementation

    return retention_policy_crud.remove(db=db, id=policy_id)