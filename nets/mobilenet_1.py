from torch import nn
import torch
from torch.hub import load_state_dict_from_url
import torch.nn.functional as F

__all__ = ['MobileNetV2', 'mobilenet_v2']


# model_urls = {
#     'mobilenet_v2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
# }


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True)
        )

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, hidden_dim = None):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        if hidden_dim is None:
            hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend([
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=1),
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class InvertedResidualquartern(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, hidden_dim = None):
        super(InvertedResidualquartern, self).__init__()
        assert stride in [1, 2]

        if hidden_dim is None:
            hidden_dim = int(round(inp * expand_ratio))
        self.identity = stride == 1 and inp == oup
        self.expand_ratio = expand_ratio

        self.conv1 = nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False)
        self.bn1   = nn.BatchNorm2d(hidden_dim)
        self.r1    = nn.ReLU6(inplace=True)

        self.conv2_h = nn.Conv2d(hidden_dim//4, hidden_dim//4, (1,3), stride, (0,1), groups=1, bias=False)
        self.bn2_h   = nn.BatchNorm2d(hidden_dim//4)
        self.conv2_v = nn.Conv2d(hidden_dim//4, hidden_dim//4, (3,1), stride, (1,0), groups=1, bias=False)
        self.bn2_v   = nn.BatchNorm2d(hidden_dim//4)
        self.conv2_h1 = nn.Conv2d(hidden_dim//4, hidden_dim//4, (1,3), stride, (0,1), groups=1, bias=False)
        self.bn2_h1   = nn.BatchNorm2d(hidden_dim//4)
        self.conv2_v1 = nn.Conv2d(hidden_dim//4, hidden_dim//4, (3,1), stride, (1,0), groups=1, bias=False)
        self.bn2_v1   = nn.BatchNorm2d(hidden_dim//4)
        self.r2    = nn.ReLU6(inplace=True)

        self.conv3 = nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False)
        self.bn3   = nn.BatchNorm2d(oup)

    def forward(self, x):
        if self.expand_ratio == 1:
            out = x
        else: 
            out = self.r1(self.bn1(self.conv1(x)))

        out1, out2, out3, out4  = out.chunk(4,1)
        out1 = self.r2(self.bn2_h(self.conv2_h(out1)))
        out2 = self.r2(self.bn2_v(self.conv2_v(out2)))
        out3 = self.r2(self.bn2_h1(self.conv2_h1(out3)))
        out4 = self.r2(self.bn2_v1(self.conv2_v1(out4)))  
        out = torch.cat([out1, out2,out3, out4], 1)

        out = self.bn3(self.conv3(out))
        if self.identity:
            return x + out
        else:
            return out


class InvertedResidualHalf(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, hidden_dim = None):
        super(InvertedResidualHalf, self).__init__()
        assert stride in [1, 2]

        if hidden_dim is None:
            hidden_dim = int(round(inp * expand_ratio))
        self.identity = stride == 1 and inp == oup
        self.expand_ratio = expand_ratio

        self.conv1 = nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False)
        self.bn1   = nn.BatchNorm2d(hidden_dim)
        self.r1    = nn.ReLU6(inplace=True)

        self.conv2_h = nn.Conv2d(hidden_dim//2, hidden_dim//2, (1,3), stride, (0,1), groups=1, bias=False)
        self.bn2_h   = nn.BatchNorm2d(hidden_dim//2)
        self.conv2_v = nn.Conv2d(hidden_dim//2, hidden_dim//2, (3,1), stride, (1,0), groups=1, bias=False)
        self.bn2_v   = nn.BatchNorm2d(hidden_dim//2)
        self.r2    = nn.ReLU6(inplace=True)

        self.conv3 = nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False)
        self.bn3   = nn.BatchNorm2d(oup)

    def forward(self, x):
        if self.expand_ratio == 1:
            out = x
        else: 
            out = self.r1(self.bn1(self.conv1(x)))

        out1, out2 = out.chunk(2,1)
        out1 = self.r2(self.bn2_h(self.conv2_h(out1)))
        out2 = self.r2(self.bn2_v(self.conv2_v(out2)))
        out = torch.cat([out1, out2], 1)

        out = self.bn3(self.conv3(out))

        if self.identity:
            return x + out
        else:
            return out

class InvertedResidualFull(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, hidden_dim = None):
        super(InvertedResidualFull, self).__init__()
        assert stride in [1, 2]

        if hidden_dim is None:
            hidden_dim = int(round(inp * expand_ratio))
        self.identity = stride == 1 and inp == oup
        self.expand_ratio = expand_ratio

        self.conv1 = nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False)
        self.bn1   = nn.BatchNorm2d(hidden_dim)
        self.r1    = nn.ReLU6(inplace=True)

        self.conv2_h = nn.Conv2d(hidden_dim, hidden_dim, (1,3), stride, (0,1), groups=1, bias=False)
        self.bn2_h   = nn.BatchNorm2d(hidden_dim)
        self.conv2_v = nn.Conv2d(hidden_dim, hidden_dim, (3,1), stride, (1,0), groups=1, bias=False)
        self.bn2_v   = nn.BatchNorm2d(hidden_dim)
        self.r2    = nn.ReLU6(inplace=True)

        self.conv3 = nn.Conv2d(2*hidden_dim, oup, 1, 1, 0, bias=False)
        self.bn3   = nn.BatchNorm2d(oup)

    def forward(self, x):
        if self.expand_ratio == 1:
            out = x
        else: 
            out = self.r1(self.bn1(self.conv1(x)))

        # out1, out2 = out.chunk(2,1)
        out1 = self.r2(self.bn2_h(self.conv2_h(out)))
        out2 = self.r2(self.bn2_v(self.conv2_v(out)))
        out = torch.cat([out1, out2], 1)

        out = self.bn3(self.conv3(out))

        if self.identity:
            return x + out
        else:
            return out


class SkipBlock(nn.Module):
    """
    Skip Block: simple module designed to connect together the blocks with the different spatial sizes
    """
    def __init__(self, inp, hidden_dim, out, kernel_size, stride, size):
        super(SkipBlock, self).__init__()
        assert stride in [1, 2]
        self.size = size
        self.identity = stride == 1 and inp == out

        self.core_block = nn.Sequential(
            # pw
            nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            # dw
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2, groups=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),

            # pw-linear
            nn.Conv2d(hidden_dim, out, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out),
        )
    

    def forward(self, x):
        x = nn.functional.adaptive_avg_pool2d(x, self.size)
        if self.identity:
            return x + self.core_block(x)
        else:
            return self.core_block(x)  

class SkipBlockquartern(nn.Module):
    """
    Skip Block: simple module designed to connect together the blocks with the different spatial sizes
    """
    def __init__(self, inp, hidden_dim, out, kernel_size, stride, size):
        super(SkipBlockquartern, self).__init__()
        assert stride in [1, 2]
        self.size = size
        self.identity = stride == 1 and inp == out

        self.conv1 = nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False)
        self.bn1   = nn.BatchNorm2d(hidden_dim)
        self.r1    = nn.ReLU6(inplace=True)

        self.conv2_h = nn.Conv2d(hidden_dim//4, hidden_dim//4, (1,3), stride, (0,1), 1, bias=False)
        self.bn2_h   = nn.BatchNorm2d(hidden_dim//4)
        self.conv2_v = nn.Conv2d(hidden_dim//4, hidden_dim//4, (3,1), stride, (1,0), 1, bias=False)
        self.bn2_v   = nn.BatchNorm2d(hidden_dim//4)
        self.conv2_h1 = nn.Conv2d(hidden_dim//4, hidden_dim//4, (1,3), stride, (0,1), 1, bias=False)
        self.bn2_h1   = nn.BatchNorm2d(hidden_dim//4)
        self.conv2_v1 = nn.Conv2d(hidden_dim//4, hidden_dim//4, (3,1), stride, (1,0), 1, bias=False)
        self.bn2_v1   = nn.BatchNorm2d(hidden_dim//4)
        self.r2    = nn.ReLU6(inplace=True)

        self.conv3 = nn.Conv2d(hidden_dim, out, 1, 1, 0, bias=False)
        self.bn3   = nn.BatchNorm2d(out)

    def forward(self, x):
        x = nn.functional.adaptive_avg_pool2d(x, self.size)
        
        out = self.r1(self.bn1(self.conv1(x)))

        out1, out2, out3, out4  = out.chunk(4,1)
        out1 = self.r2(self.bn2_h(self.conv2_h(out1)))
        out2 = self.r2(self.bn2_v(self.conv2_v(out2)))
        out3 = self.r2(self.bn2_h1(self.conv2_h1(out3)))
        out4 = self.r2(self.bn2_v1(self.conv2_v1(out4)))  
        out = torch.cat([out1, out2,out3, out4], 1)

        out = self.bn3(self.conv3(out))

        if self.identity:
            return x + out
        else:
            return out


class SkipBlockHalf(nn.Module):
    """
    Skip Block: simple module designed to connect together the blocks with the different spatial sizes
    """
    def __init__(self, inp, hidden_dim, out, kernel_size, stride, size):
        super(SkipBlockHalf, self).__init__()
        assert stride in [1, 2]
        self.size = size
        self.identity = stride == 1 and inp == out

        self.conv1 = nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False)
        self.bn1   = nn.BatchNorm2d(hidden_dim)
        self.r1    = nn.ReLU6(inplace=True)

        self.conv2_h = nn.Conv2d(hidden_dim//2, hidden_dim//2, (1,3), stride, (0,1), 1, bias=False)
        self.bn2_h   = nn.BatchNorm2d(hidden_dim//2)
        self.conv2_v = nn.Conv2d(hidden_dim//2, hidden_dim//2, (3,1), stride, (1,0), 1, bias=False)
        self.bn2_v   = nn.BatchNorm2d(hidden_dim//2)
        self.r2    = nn.ReLU6(inplace=True)

        self.conv3 = nn.Conv2d(hidden_dim, out, 1, 1, 0, bias=False)
        self.bn3   = nn.BatchNorm2d(out)

    def forward(self, x):
        x = nn.functional.adaptive_avg_pool2d(x, self.size)
        
        out = self.r1(self.bn1(self.conv1(x)))

        out1, out2 = out.chunk(2,1)
        out1 = self.r2(self.bn2_h(self.conv2_h(out1)))
        out2 = self.r2(self.bn2_v(self.conv2_v(out2)))
        out = torch.cat([out1, out2], 1)

        out = self.bn3(self.conv3(out))

        if self.identity:
            return x + out
        else:
            return out

class SkipBlockFull(nn.Module):
    """
    Skip Block: simple module designed to connect together the blocks with the different spatial sizes
    """
    def __init__(self, inp, hidden_dim, out, kernel_size, stride, size):
        super(SkipBlockFull, self).__init__()
        assert stride in [1, 2]
        self.size = size
        self.identity = stride == 1 and inp == out

        self.conv1 = nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False)
        self.bn1   = nn.BatchNorm2d(hidden_dim)
        self.r1    = nn.ReLU6(inplace=True)

        self.conv2_h = nn.Conv2d(hidden_dim, hidden_dim, (1,3), stride, (0,1), groups=1, bias=False)
        self.bn2_h   = nn.BatchNorm2d(hidden_dim)
        self.conv2_v = nn.Conv2d(hidden_dim, hidden_dim, (3,1), stride, (1,0), groups=1, bias=False)
        self.bn2_v   = nn.BatchNorm2d(hidden_dim)
        self.r2    = nn.ReLU6(inplace=True)

        self.conv3 = nn.Conv2d(hidden_dim*2, out, 1, 1, 0, bias=False)
        self.bn3   = nn.BatchNorm2d(out)


    def forward(self, x):
        x = nn.functional.adaptive_avg_pool2d(x, self.size)
        
        out = self.r1(self.bn1(self.conv1(x)))

        #out1, out2 = out.chunk(2,1)
        out1 = self.r2(self.bn2_h(self.conv2_h(out)))
        out2 = self.r2(self.bn2_v(self.conv2_v(out)))
        out = torch.cat([out1, out2], 1)

        out = self.bn3(self.conv3(out))

        if self.identity:
            return x + out
        else:
            return out

class Net(nn.Module):
    def __init__(self,cfg=None,num_classes=1000, width_mult=1.0, inverted_residual_setting=None, round_nearest=8):
        super(Net, self).__init__()
        block = InvertedResidualHalf
        input_channel = 32
        last_channel = 1280


        self.blocks = nn.ModuleList([])
        # building inverted residual and skip blocks
        block_skip = SkipBlock
        cfg = [ 32, 32, 16, 96, 24, 144, 32, 192, 64 , 384, 96,576, 160,960,320] 
  
  
        cfgs_skip = [
            # input_channel, exp_size, out_channel, kernel, stride, size
            [cfg[2], 6, cfg[6], 3, 1, 28],
            [cfg[2], 6, cfg[10], 3, 1, 14],
        ]
        self.cfgs_skipblocks = cfgs_skip
            
        self.model1 = nn.Sequential(
            ConvBNReLU(3, cfg[0], stride=2),
            block(cfg[0], cfg[2], stride=1, expand_ratio=6,hidden_dim=cfg[1])
        )

        self.model2 = nn.Sequential(
            block(cfg[2], cfg[4], stride=2, expand_ratio=6,hidden_dim=cfg[3]),
            block(cfg[4], cfg[6], stride=2, expand_ratio=6,hidden_dim=cfg[5])
        )

        self.model3 = nn.Sequential(
            block(cfg[6], cfg[8], stride=2, expand_ratio=6,hidden_dim=cfg[7]),
            block(cfg[8], cfg[10], stride=1, expand_ratio=6,hidden_dim=cfg[9])
        )

        self.model4 = nn.Sequential(
            block(cfg[10], cfg[12], stride=2, expand_ratio=6,hidden_dim=cfg[11]),
            block(cfg[12], cfg[14], stride=1, expand_ratio=6,hidden_dim=cfg[13]),
            ConvBNReLU(cfg[14], 1280, kernel_size=1)
        )

        # skip blocks
        skip1_cfg = cfgs_skip[0]
        exp_size_int = _make_divisible(skip1_cfg[0] * skip1_cfg[1], 8)
        output_channel = _make_divisible(skip1_cfg[2] * width_mult, 8)
        self.skip1 = block_skip(inp=skip1_cfg[0], hidden_dim=exp_size_int, out=output_channel, kernel_size=skip1_cfg[3], stride=skip1_cfg[4], size=skip1_cfg[5])

        skip2_cfg = cfgs_skip[1]
        exp_size_int = _make_divisible(skip2_cfg[0] * skip2_cfg[1], 8)
        output_channel = _make_divisible(skip2_cfg[2] * width_mult, 8)
        self.skip2 = block_skip(inp=skip2_cfg[0], hidden_dim=exp_size_int, out=output_channel, kernel_size=skip2_cfg[3],
                                stride=skip2_cfg[4], size=skip2_cfg[5])



        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1280, num_classes),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):

    
        x_base = self.model1(x)
        x_skip1 = self.skip1(x_base)
        x_skip2 = self.skip2(x_base)
        x = self.model2(x_base)
        x = self.model3(x + x_skip1)
        x = self.model4(x + x_skip2)
        
        
        # 1280
        x = x.mean([2, 3])
        x = self.classifier(x)
        return x

    def freeze_backbone(self):
        for param in self.blocks.parameters():
            param.requires_grad = False

    def Unfreeze_backbone(self):
        for param in self.blocks.parameters():
            param.requires_grad = True



class Conv2dFix(torch.nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros', weight_width=8, bias_width=8, activation_width=8):
        super(Conv2dFix, self).__init__(
            in_channels, out_channels, kernel_size, stride,
            padding, dilation, groups,
            bias, padding_mode)

        self.quant_fn = quantize.apply
        self.weight_width = weight_width
        self.bias_width = bias_width
        self.is_bias = bias

        self.activation_width = activation_width
        self.register_buffer('max_value_in', torch.tensor(0).float())
        self.register_buffer('max_value_out', torch.tensor(0).float())
        self.alpha = 0.95
        self.record = True
        self.flag = 0
        self.flag_out = 0

    def _conv_forward(self, conv_input, weight, bias):
        if self.padding_mode != 'zeros':
            return F.conv2d(F.pad(conv_input, self._padding_repeated_twice, mode=self.padding_mode),
                            weight, bias, self.stride, self.padding, self.dilation, self.groups)
        return F.conv2d(conv_input, weight, bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def forward(self, input_f):
        # quant input


        if self.record:
            # EMA
            with torch.no_grad():
                if self.flag == 0:
                    self.register_buffer('max_value_in', torch.max(torch.abs(input_f)))
                    self.flag = 1
                else:
                    self.register_buffer('max_value_in',
                                         self.max_value_in * self.alpha + torch.max(torch.abs(input_f)) * (
                                                     1 - self.alpha))
            input_f = self.quant_fn(input_f, self.activation_width, self.max_value_in)
        else:
            if self.flag == 0:
                input_f = self.quant_fn(input_f, self.activation_width, torch.max(torch.abs(input_f)))
                self.flag = 1
            else:
                input_f = self.quant_fn(input_f, self.activation_width, self.max_value_in)
        # quant weight and bias
        weight = self.quant_fn(self.weight, self.weight_width)
        if self.is_bias:
            bias = self.quant_fn(self.bias, self.bias_width)
        else:
            bias = None
        out = self._conv_forward(input_f, weight, bias)
        # quant output
        if self.record:
            # EMA
            with torch.no_grad():
                if self.flag_out == 0:
                    self.register_buffer('max_value_out', torch.max(torch.abs(out)))
                    self.flag_out = 1
                else:
                    self.register_buffer('max_value_out',
                                         self.max_value_out * self.alpha + torch.max(torch.abs(out)) * (1 - self.alpha))
            out = self.quant_fn(out, self.activation_width, self.max_value_out)
        else:
            if self.flag_out == 0:
                out = self.quant_fn(out, self.activation_width, torch.max(torch.abs(out)))
                self.flag_out = 1
            else:
                out = self.quant_fn(out, self.activation_width, self.max_value_out)
        return out


class quantize(torch.autograd.Function):

    # Note that both forward and backward are @staticmethods
    @staticmethod
    def forward(ctx, input_f, bitwidth=8, max_value=None):
        ctx.save_for_backward(input_f)
        bitwidth_effective = bitwidth - 1  # one sign bit
        if max_value is None:
            max_value = torch.max(torch.abs(input_f))
        msb = torch.ceil(torch.log2(max_value))
        lsb = msb - bitwidth_effective
        interval = torch.pow(2, lsb)
        input_f = torch.clamp(input_f, min=-(2 ** bitwidth_effective) * interval,
                              max=(2 ** bitwidth_effective - 1) * interval)
        output = torch.round(input_f / interval) * interval
        return output

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input, None, None



def mobilenet_v2(pretrained=False, progress=True, num_classes=1000):
    model = Net()
    # if pretrained:
    #     state_dict = load_state_dict_from_url(model_urls['mobilenet_v2'], model_dir="model_data",
    #                                           progress=progress)
    #     model.load_state_dict(state_dict, False)

    if num_classes!=1000:
        model.classifier = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(1280, num_classes),
            )
    return model
