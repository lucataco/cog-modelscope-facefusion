# modelscope-facefusion Cog model

This is an implementation of the [modelscope-facefusion](https://www.modelscope.cn/models/damo/cv_unet_face_fusion_torch/summary) as a Cog model. [Cog packages machine learning models as standard containers.](https://github.com/replicate/cog)

## Basic usage

You can then run the image with:

    cog predict -i template_image=@template.jpg -i user_image=@user.jpg

## Example:

Output Face fused:

![alt text](output.png)
