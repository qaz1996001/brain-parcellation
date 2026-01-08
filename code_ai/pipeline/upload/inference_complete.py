#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Upload inference complete notification to platform API.

支援成功與失敗兩種情況:
- 成功: 讀取 JSON，提取 inference_id 並發送 InferenceSuccessRequest
- 失敗: 發送 InferenceFailedRequest

@author: sean Ho
"""
import json
import logging
from typing import Optional, Literal, Union

import httpx

from code_ai.pipeline.dicomseg.schema import CmbDetectionResponse
from .schema import InferenceSuccessRequest, InferenceFailedRequest

logger = logging.getLogger(__name__)

ModelName = Literal["cmb_model", "aneurysm_model"]


def upload_inference_success(
    url: str,
    output_json_path: str,
    model_name: ModelName = "cmb_model",
) -> Optional[InferenceSuccessRequest]:
    """
    上傳推論成功通知。

    Args:
        url: API endpoint URL
        output_json_path: 推論結果 JSON 檔案路徑
        model_name: 模型名稱 ("cmb_model" 或 "aneurysm_model")

    Returns:
        InferenceSuccessRequest if successful, None if failed
    """
    try:
        with open(output_json_path, mode='r', encoding='utf-8') as f:
            data = json.load(f)

        # 驗證並解析 CMB 檢測結果
        cmb_response = CmbDetectionResponse.model_validate(data)
        logger.info(f"CMB detection response validated: inference_id={cmb_response.inference_id}")

        # 取得 study_instance_uid (優先從 input_study_instance_uid 列表取第一個)
        if cmb_response.input_study_instance_uid:
            study_uid = cmb_response.input_study_instance_uid[0]
        else:
            study_uid = cmb_response.study_instance_uid or ""

        # 建立成功請求
        request = InferenceSuccessRequest(
            studyInstanceUid=study_uid,
            modelName=model_name,
            inferenceId=str(cmb_response.inference_id),
        )

        # 發送 HTTP 請求
        response = _send_request(url, request)
        if response is None:
            logger.error(f"Failed to send inference success request: {request.model_dump()}")
            return None

        logger.info(f"Inference success uploaded: {request.model_dump()}")
        return request

    except FileNotFoundError:
        logger.error(f"Output JSON file not found: {output_json_path}")
        return None
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON file: {output_json_path}, error: {e}")
        return None
    except Exception as e:
        logger.exception(f"Failed to upload inference success: {e}")
        return None


def upload_inference_failed(
    url: str,
    study_instance_uid: str,
    model_name: ModelName = "cmb_model",
) -> Optional[InferenceFailedRequest]:
    """
    上傳推論失敗通知。

    Args:
        url: API endpoint URL
        study_instance_uid: DICOM Study Instance UID
        model_name: 模型名稱 ("cmb_model" 或 "aneurysm_model")

    Returns:
        InferenceFailedRequest if successful, None if failed
    """
    try:
        request = InferenceFailedRequest(
            studyInstanceUid=study_instance_uid,
            modelName=model_name,
        )

        # 發送 HTTP 請求
        response = _send_request(url, request)
        if response is None:
            logger.error(f"Failed to send inference failed request: {request.model_dump()}")
            return None

        logger.info(f"Inference failed uploaded: {request.model_dump()}")
        return request

    except Exception as e:
        logger.exception(f"Failed to upload inference failed notification: {e}")
        return None


def upload_inference_complete(
    url: str,
    success: bool,
    output_json_path: Optional[str] = None,
    study_instance_uid: Optional[str] = None,
    model_name: ModelName = "cmb_model",
) -> Optional[Union[InferenceSuccessRequest, InferenceFailedRequest]]:
    """
    上傳推論完成通知的統一入口。

    Pure function: 根據 success 參數決定上傳成功或失敗通知。

    Args:
        url: API endpoint URL
        success: 推論是否成功
        output_json_path: 推論結果 JSON 檔案路徑 (成功時必填)
        study_instance_uid: DICOM Study Instance UID (失敗時必填)
        model_name: 模型名稱 ("cmb_model" 或 "aneurysm_model")

    Returns:
        InferenceCompleteRequest (Success or Failed) if sent, None if failed

    Examples:
        # 成功情況
        upload_inference_complete(
            url="http://api/inference/complete",
            success=True,
            output_json_path="/path/to/rdx_cmb_pred_json.json",
            model_name="cmb_model"
        )

        # 失敗情況
        upload_inference_complete(
            url="http://api/inference/complete",
            success=False,
            study_instance_uid="1.2.3.4.5.6.7.8.9",
            model_name="cmb_model"
        )
    """
    if not url:
        logger.warning("URL is empty, skipping upload_inference_complete")
        return None

    if success:
        if not output_json_path:
            logger.error("output_json_path is required for success notification")
            return None
        return upload_inference_success(url, output_json_path, model_name)
    else:
        if not study_instance_uid:
            logger.error("study_instance_uid is required for failed notification")
            return None
        return upload_inference_failed(url, study_instance_uid, model_name)


def _send_request(
    url: str,
    request: Union[InferenceSuccessRequest, InferenceFailedRequest],
    timeout: float = 30.0,
) -> Optional[dict]:
    """
    發送 HTTP POST 請求到 API。

    Args:
        url: API endpoint URL
        request: 請求物件
        timeout: 請求超時時間 (秒)

    Returns:
        API response as dict if successful, None if failed
    """
    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.post(
                url,
                json=request.model_dump(),
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()
            logger.info(f"API response: status={response.status_code}")
            return response.json() if response.text else {}

    except httpx.TimeoutException:
        logger.error(f"Request timeout: {url}")
        return None
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error: {e.response.status_code} - {e.response.text}")
        return None
    except Exception as e:
        logger.exception(f"Failed to send request to {url}: {e}")
        return None
