def test_inference_package_imports():
    import inference
    from inference import GenerationConfig, NumiInferenceEngine

    assert GenerationConfig().max_new_tokens == 512
    assert NumiInferenceEngine is not None


def test_qwen_loader_disabled():
    import pytest
    from model.loading.qwen_loader import load_qwen

    with pytest.raises(RuntimeError, match="disabled"):
        load_qwen("anything")
