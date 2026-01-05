# app/inference/model.py
# Reuse DCOP models from sync module - no duplication, following YAGNI principle

from backend.app.sync.model import DCOPConfModel, DCOPEventModel

__all__ = ["DCOPConfModel", "DCOPEventModel"]
