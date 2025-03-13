import capybara as cb
from fire import Fire


def main(path):
    cb.ONNXEngine(
        path,
        backend="cuda",
        session_option={"log_severity_level": 1},
        provider_option={"enable_cuda_graph": True},
    )


Fire(main)
