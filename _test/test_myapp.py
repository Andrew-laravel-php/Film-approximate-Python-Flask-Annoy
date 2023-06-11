def test_home_page():
    response = app.test_client().get('/')
    assert response.status_code == 200
    assert b"Welcome to my Flask app" in response.data
