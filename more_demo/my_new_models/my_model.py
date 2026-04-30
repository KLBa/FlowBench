"""
Custom model adapted from [ptlflow's docs on "Adding a new model"](https://ptlflow.readthedocs.io/en/latest/custom/new_model.html).
"""


from typing import Dict, Optional, Sequence

import torch
from ptlflow.models.base_model.base_model import BaseModel
from ptlflow.utils.registry import register_model

from loguru import logger

# 1. Optionally, define a loss function, if you want to be able to train the model.
# from .loss import my_model_loss


# 2. Inherit from Base Model.
class MyModel(BaseModel):
    # 3. Optionally, add a class variable called "pretrained_checkpoints" with the pretrained weights.
    pretrained_checkpoints = {
        'ckpt_id1': '/path/to/local/checkpoint/or/https://online/file',
        'ckpt_id2': '/path/to/another/local/checkpoint/or/https://online/file',
    }

    def __init__(
        self,
        my_arg1: int = 0,  # an example of an int argument
        my_arg2: Sequence[float] = (0.0, 1.0, 2.0),  # an example of a list of floats
        my_arg3: Optional[int] = None,  # an example of an optional int argument
        **kwargs,  # Add this to receive the other arguments from BaseModel
    ) -> None:
        # 5. Call the parent constructor.
        super(MyModel, self).__init__(
            # loss_fn=my_model_loss,  # Can be None, if there is no loss function
            output_stride=64,  # Or another value, depends on the stride of your model
            **kwargs,
        )
        
        # Define your model here
        d = {
            "my_arg1": my_arg1,  # an example of an int argument
            "my_arg2": my_arg2,  # an example of a list of floats
            "my_arg3": my_arg3,  # an example of an optional int argument
            **kwargs
        }
        logger.info(f"Initialized MyModel with parameters {d}")

    # 7. Define the forward function
    def forward(
        self,
        inputs: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        # inputs may have many keys, but the main one is 'images'.
        # inputs['images'] will be a 5D tensor with shape BNCHW, where
        # B=batch size, N=number of inputs, C=channels (usually 3 for images).
        # The normal usage for optical flow is model.forward({"images": torch.stack([image1, image2], dim=1)})

        # Preprocess the input images. Check BaseModel.preprocess_images() for more information.
        images, image_resizer = self.preprocess_images(
            inputs["images"],
            bgr_add=-0.5,
            bgr_mult=2.0,
            bgr_to_rgb=True,
            resize_mode="pad",
            pad_mode="replicate",
            pad_two_side=True,
        )

        # Get the pair of images.
        image1 = images[:, 0]
        image2 = images[:, 1]

        # Define your forward here
        b, c, h, w = image1.shape
        # return zero flow in this example
        my_5D_flow_predictions = torch.zeros(b, 1, 2, h, w).to(self.device)

        # Postprocess the predictions. It mostly removes the additional padding and rescales the flow prediction accordingly.
        my_5D_flow_predictions = self.postprocess_predictions(my_5D_flow_predictions, image_resizer, is_flow=True)

        # Pack the model estimations into a dict and return.
        # It must have at least an entry 'flows', which is a 5D tensor BNCHW, typically N=1.
        preds = {
            'flows': my_5D_flow_predictions
        }
        return preds

    # 8. BaseModel already define optimizers, dataloaders, training steps, etc.
    # However, if you want to use different ones, you should create methods overriding those steps.
    # Check the PyTorch Lightning documentation for more details about which methods are required:
    # https://lightning.ai/docs/pytorch/stable/starter/introduction.html


# Create a lower caps name for your model and register it by decorating it with @register_model
@register_model
class my_model(MyModel):
    pass
