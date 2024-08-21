import cv2

import onnxruntime as ort

if __name__ == "__main__":
    # Start Inference Session in your target language and run your data through
    session = ort.InferenceSession('../../models/model.with_pre_post_processing.onnx')
    image = cv2.imread("../../imgs/2.jpg", cv2.IMREAD_GRAYSCALE)
    image = image[:, :, None]
    input_name = session.get_inputs()[0].name
    result = session.run(None, {input_name: image})
    print(result)
