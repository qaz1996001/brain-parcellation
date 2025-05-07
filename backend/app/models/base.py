from sqlalchemy import (
    Boolean, Column, DateTime, ForeignKey, Integer,
    String, Float, Text, UniqueConstraint, Index
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, Mapped
from typing import List, Optional
from datetime import datetime

# Base class for all models
Base = declarative_base()