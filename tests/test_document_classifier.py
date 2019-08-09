from document_classifier import __version__
from document_classifier import CNN

def test_version():
    assert __version__ == '0.1.0'

def test_cnn_instance():
    cnn = CNN(3)
    assert not cnn.is_binary
