# app/routers/sync/deps.py

import datetime
import inspect
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Callable,
    Literal,
    NamedTuple,
    Optional,
    TypeVar,
    Union,
    cast,
    overload,
)
from uuid import UUID

from fastapi import Depends, Query
from fastapi.exceptions import RequestValidationError
from advanced_alchemy.filters import (
    BeforeAfter,
    CollectionFilter,
    FilterTypes,
    LimitOffset,
    NotInCollectionFilter,
    OrderBy,
    SearchFilter,
)

from advanced_alchemy.utils.text import camelize
from typing import TYPE_CHECKING, Callable

from advanced_alchemy.filters import (
    BeforeAfter,
    CollectionFilter,
    FilterTypes,
)
from advanced_alchemy.extensions.fastapi.providers import DEPENDENCY_DEFAULTS, FilterConfig, _make_hashable
from advanced_alchemy.extensions.fastapi.providers import DependencyCache, SortOrder, FieldNameType
from advanced_alchemy.extensions.fastapi.providers import _aggregate_filter_function,DependencyDefaults



dep_cache = DependencyCache()


def _create_filter_aggregate_function_fastapi(  # noqa: C901, PLR0915
    config: FilterConfig,
    dep_defaults: "DependencyDefaults" = DEPENDENCY_DEFAULTS,
) -> Callable[..., list[FilterTypes]]:
    """Create a FastAPI dependency provider function that aggregates multiple filter dependencies.

    Returns:
        A FastAPI dependency provider function that aggregates multiple filter dependencies.
    """
    params: list[inspect.Parameter] = []
    annotations: dict[str, Any] = {}

    # Add id filter providers
    if (id_filter := config.get("id_filter", False)) is not False:

        def provide_id_filter(  # pyright: ignore[reportUnknownParameterType]
            ids: Annotated[  # type: ignore
                Optional[list[id_filter]],  # pyright: ignore
                Query(
                    alias="ids",
                    required=False,
                    description="IDs to filter by.",
                ),
            ] = None,
        ) -> Optional[CollectionFilter[id_filter]]:  # type: ignore
            return CollectionFilter[id_filter](field_name=config.get("id_field", "id"), values=ids) if ids else None  # type: ignore

        params.append(
            inspect.Parameter(
                name=dep_defaults.ID_FILTER_DEPENDENCY_KEY,
                kind=inspect.Parameter.KEYWORD_ONLY,
                annotation=Annotated[Optional[CollectionFilter[id_filter]], Depends(provide_id_filter)],  # type: ignore
            )
        )
        annotations[dep_defaults.ID_FILTER_DEPENDENCY_KEY] = Annotated[
            Optional[CollectionFilter[id_filter]], Depends(provide_id_filter)  # type: ignore
        ]

    # Add created_at filter providers
    if config.get("created_at", False):

        def provide_created_at_filter(
            before: Annotated[
                Optional[str],
                Query(
                    alias="createdBefore",
                    description="Filter by created date before this timestamp.",
                    json_schema_extra={"format": "date-time"},
                ),
            ] = None,
            after: Annotated[
                Optional[str],
                Query(
                    alias="createdAfter",
                    description="Filter by created date after this timestamp.",
                    json_schema_extra={"format": "date-time"},
                ),
            ] = None,
        ) -> Optional[BeforeAfter]:
            before_dt = None
            after_dt = None

            # Validate both parameters regardless of endpoint path
            if before is not None:
                try:
                    before_dt = datetime.datetime.fromisoformat(before.replace("Z", "+00:00"))
                except (ValueError, TypeError, AttributeError) as e:
                    raise RequestValidationError(
                        errors=[{"loc": ["query", "createdBefore"], "msg": "Invalid date format"}]
                    ) from e

            if after is not None:
                try:
                    after_dt = datetime.datetime.fromisoformat(after.replace("Z", "+00:00"))
                except (ValueError, TypeError, AttributeError) as e:
                    raise RequestValidationError(
                        errors=[{"loc": ["query", "createdAfter"], "msg": "Invalid date format"}]
                    ) from e

            return (
                BeforeAfter(field_name="created_at", before=before_dt, after=after_dt)
                if before_dt or after_dt
                else None  # pyright: ignore
            )

        param_name = dep_defaults.CREATED_FILTER_DEPENDENCY_KEY
        params.append(
            inspect.Parameter(
                name=param_name,
                kind=inspect.Parameter.KEYWORD_ONLY,
                annotation=Annotated[Optional[BeforeAfter], Depends(provide_created_at_filter)],
            )
        )
        annotations[param_name] = Annotated[Optional[BeforeAfter], Depends(provide_created_at_filter)]

    # Add updated_at filter providers
    if config.get("updated_at", False):

        def provide_updated_at_filter(
            before: Annotated[
                Optional[str],
                Query(
                    alias="updatedBefore",
                    description="Filter by updated date before this timestamp.",
                    json_schema_extra={"format": "date-time"},
                ),
            ] = None,
            after: Annotated[
                Optional[str],
                Query(
                    alias="updatedAfter",
                    description="Filter by updated date after this timestamp.",
                    json_schema_extra={"format": "date-time"},
                ),
            ] = None,
        ) -> Optional[BeforeAfter]:
            before_dt = None
            after_dt = None

            # Validate both parameters regardless of endpoint path
            if before is not None:
                try:
                    before_dt = datetime.datetime.fromisoformat(before.replace("Z", "+00:00"))
                except (ValueError, TypeError, AttributeError) as e:
                    raise RequestValidationError(
                        errors=[{"loc": ["query", "updatedBefore"], "msg": "Invalid date format"}]
                    ) from e

            if after is not None:
                try:
                    after_dt = datetime.datetime.fromisoformat(after.replace("Z", "+00:00"))
                except (ValueError, TypeError, AttributeError) as e:
                    raise RequestValidationError(
                        errors=[{"loc": ["query", "updatedAfter"], "msg": "Invalid date format"}]
                    ) from e

            return (
                BeforeAfter(field_name="updated_at", before=before_dt, after=after_dt)
                if before_dt or after_dt
                else None  # pyright: ignore
            )

        param_name = dep_defaults.UPDATED_FILTER_DEPENDENCY_KEY
        params.append(
            inspect.Parameter(
                name=param_name,
                kind=inspect.Parameter.KEYWORD_ONLY,
                annotation=Annotated[Optional[BeforeAfter], Depends(provide_updated_at_filter)],
            )
        )
        annotations[param_name] = Annotated[Optional[BeforeAfter], Depends(provide_updated_at_filter)]

    if before_after_fields := config.get("before_after_fields"):
        before_after_fields = {before_after_fields} if isinstance(before_after_fields,
                                                                  (str, FieldNameType)) else before_after_fields

        for field_def in before_after_fields:
            def create_before_after_filter_provider(
                    local_field_name: str,
            ) -> Callable[..., Optional[BeforeAfter]]:
                def provide_before_after_filter(
                        before: Annotated[
                            Optional[str],
                            Query(
                                alias=f"{camelize(local_field_name)}Before",
                                description=f"Filter by {local_field_name} before this timestamp.",
                                json_schema_extra={"format": "date-time"},
                            ),
                        ] = None,
                        after: Annotated[
                            Optional[str],
                            Query(
                                alias=f"{camelize(local_field_name)}After",
                                description=f"Filter by {local_field_name} after this timestamp.",
                                json_schema_extra={"format": "date-time"},
                            ),
                        ] = None,
                ) -> Optional[BeforeAfter]:
                    before_dt = None
                    after_dt = None

                    # Validate both parameters regardless of endpoint path
                    if before is not None:
                        try:
                            before_dt = datetime.datetime.fromisoformat(before.replace("Z", "+00:00"))
                        except (ValueError, TypeError, AttributeError) as e:
                            raise RequestValidationError(
                                errors=[{"loc": ["query", f"{camelize(local_field_name)}Before"],
                                         "msg": "Invalid date format"}]
                            ) from e

                    if after is not None:
                        try:
                            after_dt = datetime.datetime.fromisoformat(after.replace("Z", "+00:00"))
                        except (ValueError, TypeError, AttributeError) as e:
                            raise RequestValidationError(
                                errors=[{"loc": ["query", f"{camelize(local_field_name)}After"],
                                         "msg": "Invalid date format"}]
                            ) from e

                    return (
                        BeforeAfter(field_name=local_field_name, before=before_dt, after=after_dt)
                        if before_dt or after_dt
                        else None
                    )

                return provide_before_after_filter

            provider = create_before_after_filter_provider(field_def.name)
            param_name = f"{field_def.name}_before_after_filter"

            params.append(
                inspect.Parameter(
                    name=param_name,
                    kind=inspect.Parameter.KEYWORD_ONLY,
                    annotation=Annotated[Optional[BeforeAfter], Depends(provider)],
                )
            )
            annotations[param_name] = Annotated[Optional[BeforeAfter], Depends(provider)]

    # Add pagination filter providers
    if config.get("pagination_type") == "limit_offset":

        def provide_limit_offset_pagination(
            current_page: Annotated[
                int,
                Query(
                    ge=1,
                    alias="currentPage",
                    description="Page number for pagination.",
                ),
            ] = 1,
            page_size: Annotated[
                int,
                Query(
                    ge=1,
                    alias="pageSize",
                    description="Number of items per page.",
                ),
            ] = config.get("pagination_size", dep_defaults.DEFAULT_PAGINATION_SIZE),
        ) -> LimitOffset:
            return LimitOffset(limit=page_size, offset=page_size * (current_page - 1))

        param_name = dep_defaults.LIMIT_OFFSET_FILTER_DEPENDENCY_KEY
        params.append(
            inspect.Parameter(
                name=param_name,
                kind=inspect.Parameter.KEYWORD_ONLY,
                annotation=Annotated[LimitOffset, Depends(provide_limit_offset_pagination)],
            )
        )
        annotations[param_name] = Annotated[LimitOffset, Depends(provide_limit_offset_pagination)]

    # Add search filter providers
    if search_fields := config.get("search"):

        def provide_search_filter(
            search_string: Annotated[
                Optional[str],
                Query(
                    required=False,
                    alias="searchString",
                    description="Search term.",
                ),
            ] = None,
            ignore_case: Annotated[
                Optional[bool],
                Query(
                    required=False,
                    alias="searchIgnoreCase",
                    description="Whether search should be case-insensitive.",
                ),
            ] = config.get("search_ignore_case", False),
        ) -> SearchFilter:
            field_names = set(search_fields.split(",")) if isinstance(search_fields, str) else search_fields

            return SearchFilter(
                field_name=field_names,
                value=search_string,  # type: ignore[arg-type]
                ignore_case=ignore_case or False,
            )

        param_name = dep_defaults.SEARCH_FILTER_DEPENDENCY_KEY
        params.append(
            inspect.Parameter(
                name=param_name,
                kind=inspect.Parameter.KEYWORD_ONLY,
                annotation=Annotated[Optional[SearchFilter], Depends(provide_search_filter)],
            )
        )
        annotations[param_name] = Annotated[Optional[SearchFilter], Depends(provide_search_filter)]

    # Add sort filter providers
    if sort_field := config.get("sort_field"):
        sort_order_default = config.get("sort_order", "desc")

        def provide_order_by(
            field_name: Annotated[
                str,
                Query(
                    alias="orderBy",
                    description="Field to order by.",
                    required=False,
                ),
            ] = sort_field,  # type: ignore[assignment]
            sort_order: Annotated[
                Optional[SortOrder],
                Query(
                    alias="sortOrder",
                    description="Sort order ('asc' or 'desc').",
                    required=False,
                ),
            ] = sort_order_default,
        ) -> OrderBy:
            return OrderBy(field_name=field_name, sort_order=sort_order or sort_order_default)

        param_name = dep_defaults.ORDER_BY_FILTER_DEPENDENCY_KEY
        params.append(
            inspect.Parameter(
                name=param_name,
                kind=inspect.Parameter.KEYWORD_ONLY,
                annotation=Annotated[OrderBy, Depends(provide_order_by)],
            )
        )
        annotations[param_name] = Annotated[OrderBy, Depends(provide_order_by)]

    # Add not_in filter providers
    if not_in_fields := config.get("not_in_fields"):
        not_in_fields = {not_in_fields} if isinstance(not_in_fields, (str, FieldNameType)) else not_in_fields
        for field_def in not_in_fields:

            def create_not_in_filter_provider(  # pyright: ignore
                local_field_name: str,
                local_field_type: type[Any],
            ) -> Callable[..., Optional[NotInCollectionFilter[field_def.type_hint]]]:  # type: ignore
                def provide_not_in_filter(  # pyright: ignore
                    values: Annotated[  # type: ignore
                        Optional[set[local_field_type]],  # pyright: ignore
                        Query(
                            alias=camelize(f"{local_field_name}_not_in"),
                            description=f"Filter {local_field_name} not in values",
                        ),
                    ] = None,
                ) -> Optional[NotInCollectionFilter[local_field_type]]:  # type: ignore
                    return NotInCollectionFilter(field_name=local_field_name, values=values) if values else None  # pyright: ignore

                return provide_not_in_filter  # pyright: ignore

            provider = create_not_in_filter_provider(field_def.name, field_def.type_hint)  # pyright: ignore
            param_name = f"{field_def.name}_not_in_filter"
            params.append(
                inspect.Parameter(
                    name=param_name,
                    kind=inspect.Parameter.KEYWORD_ONLY,
                    annotation=Annotated[Optional[NotInCollectionFilter[field_def.type_hint]], Depends(provider)],  # type: ignore
                )
            )
            annotations[param_name] = Annotated[Optional[NotInCollectionFilter[field_def.type_hint]], Depends(provider)]  # type: ignore

    # Add in filter providers
    if in_fields := config.get("in_fields"):
        in_fields = {in_fields} if isinstance(in_fields, (str, FieldNameType)) else in_fields
        for field_def in in_fields:

            def create_in_filter_provider(  # pyright: ignore
                local_field_name: str,
                local_field_type: type[Any],
            ) -> Callable[..., Optional[CollectionFilter[field_def.type_hint]]]:  # type: ignore
                def provide_in_filter(  # pyright: ignore
                    values: Annotated[  # type: ignore
                        Optional[set[local_field_type]],  # pyright: ignore
                        Query(
                            alias=camelize(f"{local_field_name}_in"),
                            description=f"Filter {local_field_name} in values",
                        ),
                    ] = None,
                ) -> Optional[CollectionFilter[local_field_type]]:  # type: ignore
                    return CollectionFilter(field_name=local_field_name, values=values) if values else None  # pyright: ignore

                return provide_in_filter  # pyright: ignore

            provider = create_in_filter_provider(field_def.name, field_def.type_hint)  # type: ignore
            param_name = f"{field_def.name}_in_filter"
            params.append(
                inspect.Parameter(
                    name=param_name,
                    kind=inspect.Parameter.KEYWORD_ONLY,
                    annotation=Annotated[Optional[CollectionFilter[field_def.type_hint]], Depends(provider)],  # type: ignore
                )
            )
            annotations[param_name] = Annotated[Optional[CollectionFilter[field_def.type_hint]], Depends(provider)]  # type: ignore

    _aggregate_filter_function.__signature__ = inspect.Signature(  # type: ignore
        parameters=params,
        return_annotation=Annotated[list[FilterTypes], Depends(_aggregate_filter_function)],
    )

    return _aggregate_filter_function


def provide_filters(
    config: FilterConfig,
    dep_defaults: DependencyDefaults = DEPENDENCY_DEFAULTS,
) -> Callable[..., list[FilterTypes]]:
    """Create FastAPI dependency providers for filters based on the provided configuration.

    Returns:
        A FastAPI dependency provider function that aggregates multiple filter dependencies.
    """
    # Check if any filters are actually requested in the config
    filter_keys = {
        "id_filter",
        "created_at",
        "updated_at",
        "before_after_fields",  # 添加這行
        "pagination_type",
        "search",
        "sort_field",
        "not_in_fields",
        "in_fields",
    }

    has_filters = False
    for key in filter_keys:
        value = config.get(key)
        if value is not None and value is not False and value != []:
            has_filters = True
            break

    if not has_filters:
        return list

    # Calculate cache key using hashable version of config
    cache_key = hash(_make_hashable(config))

    # Check cache first
    cached_dep = dep_cache.get_dependencies(cache_key)
    if cached_dep is not None:
        return cached_dep

    dep = _create_filter_aggregate_function_fastapi(config, dep_defaults)
    dep_cache.add_dependencies(cache_key, dep)
    return dep