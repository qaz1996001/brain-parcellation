import pytest

from backend.app.sync.schemas import OrthancID, PostStudyRequest


def test_post_study_request_resolves_ids_list():
    sample_id = OrthancID("ee5f44b1-e1f0dc1c-8825e04b-d5fb7bae-0373ba30")
    request = PostStudyRequest(ids=[sample_id])

    assert request.resolved_ids() == [sample_id]


def test_post_study_request_resolves_single_study_uid():
    sample_id = OrthancID("13fb1be1-71d25700-b131126f-c73708af-42d28093")
    request = PostStudyRequest(study_uid=sample_id)

    assert request.resolved_ids() == [sample_id]


def test_post_study_request_normalizes_prev_study_uid():
    sample_id = OrthancID("ee5f44b1-e1f0dc1c-8825e04b-d5fb7bae-0373ba30")
    request = PostStudyRequest(ids=[sample_id], prev_study_uid="")

    assert request.prev_study_uid is None


def test_post_study_request_rejects_empty_payload():
    with pytest.raises(ValueError):
        PostStudyRequest()

