import cv2
import onnx
import torch

import onnxruntime as ort
import torch.nn as nn

from onnxruntime_extensions.tools.pre_post_processing import (CenterCrop,
                                                              ChannelsLastToChannelsFirst,
                                                              PrePostProcessor,
                                                              create_named_value,
                                                              ImageBytesToFloat,
                                                              Unsqueeze,
                                                              Softmax)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layer = nn.Sequential(nn.Conv2d(1, 16, 4, 2, 2),
                                        nn.ReLU(),
                                        nn.Conv2d(16, 32, 4, 2, 2),
                                        nn.ReLU(),
                                        nn.Conv2d(32, 64, 4, 2, 2),
                                        nn.ReLU())
        self.linear_layer = nn.Sequential(nn.Linear(64 * 5 * 5, 128),
                                          nn.ReLU(),
                                          nn.Linear(128, 10))

    def forward(self, x):
        x = self.conv_layer(x)
        x = x.view(x.shape[0], -1)
        x = self.linear_layer(x)
        return x


if __name__ == "__main__":
    # Basic pytorch
    torch_input = torch.randn(1, 1, 28, 28)
    model = Model()
    torch_output = model(torch_input)

    # Export pytorch model to onnx model
    onnx_model = torch.onnx.export(model,
                                   torch_input,
                                   "models/model.onnx", export_params=True,
                                   opset_version=16,
                                   do_constant_folding=True,
                                   input_names=['input'],
                                   output_names=['output'],
                                   dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
                                   )

    # Specify onnx input
    new_input = create_named_value('image', onnx.TensorProto.UINT8, ['h', "w", "c"])

    # Set up pipeline
    pipeline = PrePostProcessor([new_input], onnx_opset=16)

    # Add preprocessing
    pipeline.add_pre_processing(
        [
            CenterCrop(28, 28),

            # ONNX models are typically channels first. output shape is {channels, 28, 28}
            ChannelsLastToChannelsFirst(),

            # Convert uint8 values in range 0..255 to float values in range 0..1
            ImageBytesToFloat(),

            # add batch dim so shape is {1, channels, 28, 28}. we now match the original model input
            Unsqueeze(axes=[0]),
        ]
    )

    # Add postprocessing
    pipeline.add_post_processing(
        [
            Softmax()
        ]
    )

    # Load model and wrap it in the created pipeline
    model = onnx.load('models/model.onnx')
    new_model = pipeline.run(model)
    onnx.save_model(new_model, 'models/model.with_pre_post_processing.onnx')
