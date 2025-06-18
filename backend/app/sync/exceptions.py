class DCOPEventError(Exception):
    """DCOP 事件處理基礎異常"""
    pass


class DuplicateEventError(DCOPEventError):
    """重複事件異常"""
    pass


class InvalidStateTransitionError(DCOPEventError):
    """無效狀態轉換異常"""
    pass


class ExternalServiceError(DCOPEventError):
    """外部服務調用異常"""
    pass
