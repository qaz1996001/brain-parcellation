from typing import Any, Optional
from sqlalchemy import func
from sqlalchemy.engine import Connection
from sqlalchemy.sql import Select
from fastapi_pagination.bases import AbstractParams
from fastapi_pagination.utils import verify_params
from fastapi_pagination.ext.sqlalchemy import create_count_query # paginate,




def paginate_items(
    conn: Connection,
    stmt: Select,
    params: Optional[AbstractParams] = None,
) -> Any:
    params, raw_params = verify_params(params, "limit-offset")

    # total = conn.scalar(stmt.with_only_columns(func.count()))
    total = conn.scalar(create_count_query(stmt, use_subquery=True))
    q = stmt.offset(raw_params.offset).limit(raw_params.limit)
    items = conn.execute(q).all()
    # print('params',params)
    # print('raw_params',raw_params)
    # print('total',total)
    # print('items len',len(items))
    # print('stmt.with_only_columns(func.count()) ',stmt.with_only_columns(func.count()))
    # print('create_count_query',create_count_query(stmt, use_subquery=True))
    # print('stmt', stmt)
    return items,total,raw_params,params