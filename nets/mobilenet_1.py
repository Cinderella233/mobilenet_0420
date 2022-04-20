# from torch import nn
# import torch
# from torch.hub import load_state_dict_from_url
# import torch.nn.functional as F
#
# __all__ = ['MobileNetV2', 'mobilenet_v2']
#
#
# model_urls = {
#     'mobilenet_v2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
# }
#
#
# def _make_divisible(v, divisor, min_value=None):
#     if min_value is None:
#         min_value = divisor
#     new_v = max(min_value int(v + divisor / 2) // divisor * divisor)
#     if new_v < 0.9 * v:
#         new_v += divisor
#     return new_v
#
# class ConvBNReLU(nn.Sequential):
#     def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
#         padding = (kernel_size - 1) // 2
#         super(ConvBNReLU, self).__init__(
#             Conv2dFix(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
#             nn.BatchNorm2d(out_planes),
#             nn.ReLU6(inplace=True)
#         )
#
# class InvertedResidual(nn.Module):
#     def __init__(self, inp, oup, stride, expand_ratio):
#         super(InvertedResidual, self).__init__()
#         self.stride = stride
#         assert stride in [1, 2]
#
#         hidden_dim = int(round(inp * expand_ratio))
#         self.use_res_connect = self.stride == 1 and inp == oup
#
#         layers = []
#         if expand_ratio != 1:
#             layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
#         layers.extend([
#             ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
#             Conv2dFix(hidden_dim, oup, 1, 1, 0, bias=False),
#             nn.BatchNorm2d(oup),
#         ])
#         self.conv = nn.Sequential(*layers)
#
#     def forward(self, x):
#         if self.use_res_connect:
#             return x + self.conv(x)
#         else:
#             return self.conv(x)
#
#
# class MobileNetV2(nn.Module):
#     def __init__(self, num_classes=1000, width_mult=1.0, inverted_residual_setting=None, round_nearest=8):
#         super(MobileNetV2, self).__init__()
#         block = InvertedResidual
#         input_channel = 32
#         last_channel = 1280
#
#         if inverted_residual_setting is None:
#             inverted_residual_setting = [
#                 # t, c, n, s
#                 # 112, 112, 32 -> 112, 112, 16
#                 [1, 16, 1, 1],
#                 # 112, 112, 16 -> 56, 56, 24
#                 [6, 24, 2, 2],
#                 # 56, 56, 24 -> 28, 28, 32
#                 [6, 32, 3, 2],
#                 # 28, 28, 32 -> 14, 14, 64
#                 [6, 64, 4, 2],
#                 # 14, 14, 64 -> 14, 14, 96
#                 [6, 96, 3, 1],
#                 # 14, 14, 96 -> 7, 7, 160
#                 [6, 160, 3, 2],
#                 # 7, 7, 160 -> 7, 7, 320
#                 [6, 320, 1, 1],
#             ]
#
#         if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
#             raise ValueError("inverted_residual_setting should be non-empty "
#                              "or a 4-element list, got {}".format(inverted_residual_setting))
#
#         input_channel = _make_divisible(input_channel * width_mult, round_nearest)
#         self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
#
#         # 224, 224, 3 -> 112, 112, 32
#         features = [ConvBNReLU(3, input_channel, stride=2)]
#
#         for t, c, n, s in inverted_residual_setting:
#             output_channel = _make_divisible(c * width_mult, round_nearest)
#             for i in range(n):
#                 stride = s if i == 0 else 1
#                 features.append(block(input_channel, output_channel, stride, expand_ratio=t))
#                 input_channel = output_channel
#
#         # 7, 7, 320 -> 7,7,1280
#         features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1))
#         self.features = nn.Sequential(*features)
#
#         t = self.features[18]
#         print(self.features[18])
#
#         self.classifier = nn.Sequential(
#             nn.Dropout(0.2),
#             nn.Linear(self.last_channel, num_classes),
#         )
#
#         for m in self.modules():
#             if isinstance(m, Conv2dFix):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out')
#                 if m.bias is not None:
#                     nn.init.zeros_(m.bias)
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.ones_(m.weight)
#                 nn.init.zeros_(m.bias)
#             elif isinstance(m, nn.Linear):
#                 nn.init.normal_(m.weight, 0, 0.01)
#                 nn.init.zeros_(m.bias)
#
#     def forward(self, x):
#         x = self.features(x)
#         # 1280
#         x = x.mean([2, 3])
#         x = self.classifier(x)
#         return x
#
#     def freeze_backbone(self):
#         for param in self.features.parameters():
#             param.requires_grad = False
#
#     def Unfreeze_backbone(self):
#         for param in self.features.parameters():
#             param.requires_grad = True
#
# class Conv2dFix(torch.nn.Conv2d):
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1,
#                  padding=0, dilation=1, groups=1,
#                  bias=True, padding_mode='zeros', weight_width=8, bias_width=8, activation_width=8):
#         super(Conv2dFix, self).__init__(
#             in_channels, out_channels, kernel_size, stride,
#             padding, dilation, groups,
#             bias, padding_mode)
#
#         self.quant_fn = quantize.apply
#         self.weight_width = weight_width
#         self.bias_width = bias_width
#         self.is_bias = bias
#
#         self.activation_width = activation_width
#         self.register_buffer('max_value_in', torch.tensor(0).float())
#         self.register_buffer('max_value_out', torch.tensor(0).float())
#         self.alpha = 0.95
#         self.record = False
#
#     def _conv_forward(self, conv_input, weight, bias):
#         if self.padding_mode != 'zeros':
#             return F.conv2d(F.pad(conv_input, self._padding_repeated_twice, mode=self.padding_mode),
#                             weight, bias, self.stride, self.padding, self.dilation, self.groups)
#         return F.conv2d(conv_input, weight, bias, self.stride,
#                         self.padding, self.dilation, self.groups)
#
#     def forward(self, input_f):
#         # quant input
#         if self.record and self.training:
#             # EMA
#             with torch.no_grad():
#                 if self.max_value_in == 0:
#                     self.register_buffer('max_value_in', torch.max(torch.abs(input_f)))
#                 else:
#                     self.register_buffer('max_value_in',
#                                          self.max_value_in * self.alpha + torch.max(torch.abs(input_f)) * (
#                                                      1 - self.alpha))
#             input_f = self.quant_fn(input_f, self.activation_width, self.max_value_in)
#         else:
#             if self.max_value_in == 0:
#                 input_f = self.quant_fn(input_f, self.activation_width, torch.max(torch.abs(input_f)))
#             else:
#                 input_f = self.quant_fn(input_f, self.activation_width, self.max_value_in)
#
#         # quant weight and bias
#         weight = self.quant_fn(self.weight, self.weight_width)
#
#         if self.is_bias:
#             bias = self.quant_fn(self.bias, self.bias_width)
#         else:
#             bias = None
#         out = self._conv_forward(input_f, weight, bias)
#
#         # quant output
#         if self.record and self.training:
#             # EMA
#             with torch.no_grad():
#                 if self.max_value_out == 0:
#                     self.register_buffer('max_value_out', torch.max(torch.abs(out)))
#                 else:
#                     self.register_buffer('max_value_out',
#                                          self.max_value_out * self.alpha + torch.max(torch.abs(out)) * (1 - self.alpha))
#             out = self.quant_fn(out, self.activation_width, self.max_value_out)
#         else:
#             if self.max_value_out == 0:
#                 out = self.quant_fn(out, self.activation_width, torch.max(torch.abs(out)))
#             else:
#                 out = self.quant_fn(out, self.activation_width, self.max_value_out)
#         return out
#
#
# class quantize(torch.autograd.Function):
#
#     # Note that both forward and backward are @staticmethods
#     @staticmethod
#     def forward(ctx, input_f, bitwidth=4, max_value=None):
#         ctx.save_for_backward(input_f)
#         bitwidth_effective = bitwidth - 1  # one sign bit
#         if max_value is None:
#             max_value = torch.max(torch.abs(input_f))
#         msb = torch.ceil(torch.log2(max_value))
#         lsb = msb - bitwidth_effective
#         interval = torch.pow(2, lsb)
#         input_f = torch.clamp(input_f, min=-(2 ** bitwidth_effective) * interval,
#                               max=(2 ** bitwidth_effective - 1) * interval)
#         output = torch.round(input_f / interval) * interval
#         return output
#
#     # This function has only a single output, so it gets only one gradient
#     @staticmethod
#     def backward(ctx, grad_output):
#         grad_input = grad_output.clone()
#         return grad_input, None, None
#
# def mobilenet_v2(pretrained=False, progress=True, num_classes=1000):
#     model = MobileNetV2()
#     if pretrained:
#         state_dict = load_state_dict_from_url(model_urls['mobilenet_v2'], model_dir="model_data",
#                                               progress=progress)
#         model.load_state_dict(state_dict, False)
#
#     if num_classes!=1000:
#         model.classifier = nn.Sequential(
#                 nn.Dropout(0.2),
#                 nn.Linear(model.last_channel, num_classes),
#             )
#     return model
