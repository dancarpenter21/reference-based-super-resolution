from io import BytesIO


def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_create_requires_two_supported_videos(client):
    response = client.post(
        "/api/v1/jobs",
        files={
            "low_video": ("low.txt", BytesIO(b"bad"), "text/plain"),
            "reference_video": ("reference.mp4", BytesIO(b"bad"), "video/mp4"),
        },
    )
    assert response.status_code == 400


def test_unknown_job_is_404(client):
    assert client.get("/api/v1/jobs/not-real").status_code == 404
