import sys
import types
import importlib
import pytest


def import_main(monkeypatch):
    # create dummy flask module
    flask = types.ModuleType('flask')

    class DummyFlask:
        def __init__(self, *args, **kwargs):
            pass

        def route(self, *args, **kwargs):
            def decorator(fn):
                return fn
            return decorator

    flask.Flask = DummyFlask
    flask.request = None
    flask.jsonify = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, 'flask', flask)

    # dummy PIL.Image
    pil = types.ModuleType('PIL')
    image = types.ModuleType('Image')
    image.open = lambda *args, **kwargs: None
    pil.Image = image
    monkeypatch.setitem(sys.modules, 'PIL', pil)
    monkeypatch.setitem(sys.modules, 'PIL.Image', image)

    # dummy ultralytics YOLO
    ultra = types.ModuleType('ultralytics')
    ultra.YOLO = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, 'ultralytics', ultra)

    # dummy firebase_admin and submodules
    fb = types.ModuleType('firebase_admin')
    fb.messaging = types.ModuleType('messaging')
    fb.credentials = types.ModuleType('credentials')
    fb.firestore = types.ModuleType('firestore')
    fb.initialize_app = lambda *args, **kwargs: None
    fb.credentials.Certificate = lambda *args, **kwargs: None
    fb.firestore.client = lambda *args, **kwargs: types.SimpleNamespace()
    monkeypatch.setitem(sys.modules, 'firebase_admin', fb)
    monkeypatch.setitem(sys.modules, 'firebase_admin.messaging', fb.messaging)
    monkeypatch.setitem(sys.modules, 'firebase_admin.credentials', fb.credentials)
    monkeypatch.setitem(sys.modules, 'firebase_admin.firestore', fb.firestore)

    sys.modules.pop('main', None)
    # load module from file location to avoid missing dependencies
    import importlib.util, os
    spec = importlib.util.spec_from_file_location('main', os.path.join(os.path.dirname(__file__), '..', 'main.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules['main'] = module
    return module


def test_environment_factor_q10(monkeypatch):
    main = import_main(monkeypatch)
    assert main.environment_factor_q10(4.0, 85.0) == pytest.approx(1.0)


def test_estimate_rsl(monkeypatch):
    main = import_main(monkeypatch)
    now = 100000
    timestamps = [now - 3600 * 10, now - 3600 * 20, now - 3600 * 70]
    result = main.estimate_rsl('banana', timestamps, now, 4.0, 85.0)
    expected = [50.0, 40.0, 0.0]
    assert result == pytest.approx(expected)
