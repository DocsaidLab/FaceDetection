from .scrfd import SCRFD, SCRFDOrigin

ONNXs = {
    "SCRFD": SCRFD,
    "SCRFDOrigin": SCRFDOrigin,
}


def build_onnx_pipline(
    name: str,
    **kwargs: dict,
):
    detector_cls = ONNXs.get(name, None)

    if detector_cls is None:
        raise ValueError(f"Given name = {name} is unsupported.")

    onnx_pipline = detector_cls(**kwargs)

    return onnx_pipline
