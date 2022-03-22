import segmentation_models_pytorch as smp


def get_model_dic():
    dic = {}
    dic["model_1"] = smp.FPN(
        encoder_name="resnet101",
        encoder_depth=5,
        encoder_weights=None,
        decoder_pyramid_channels=256,
        decoder_segmentation_channels=128,
        decoder_merge_policy="add",
        decoder_dropout=0.2,
        in_channels=3,
        classes=1,
        activation=None,
        upsampling=4,
        aux_params=None,
    )
    dic["model_2"] = smp.DeepLabV3Plus(
        encoder_name="resnet101",
        encoder_depth=5,
        encoder_weights=None,
        encoder_output_stride=16,
        decoder_channels=256,
        decoder_atrous_rates=(12, 24, 36),
        in_channels=3,
        classes=1,
        activation=None,
        upsampling=4,
        aux_params=None,
    )
    return dic
