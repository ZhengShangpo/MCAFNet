from models.common import *

class SimAM(torch.nn.Module):
    def __init__(self, channels=None, out_channels=None, e_lambda=1e-4):
        super(SimAM, self).__init__()

        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)
        return s

    @staticmethod
    def get_module_name():
        return "simam"

    def forward(self, x):
        b, c, h, w = x.size()

        n = w * h - 1

        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5

        return x * self.activaton(y)


class feature_fusion_at(nn.Module):
    def __init__(self, inchannel):
        super(feature_fusion_at, self).__init__()
        self.fusion_1 = Conv(inchannel, inchannel, 1)
        self.fusion_2 = Conv(inchannel, inchannel, 1)
        self.fusion_3 = Conv(inchannel, inchannel, 1)
        self.fusion_4 = Conv(inchannel * 3, 3, 1)

    def forward(self, x1, x2, x3):
        fusion = torch.softmax(
            self.fusion_4(torch.cat([self.fusion_1(x1), self.fusion_2(x2), self.fusion_3(x3)], dim=1)), dim=1)
        x1_weight, x2_weight, x3_weight = torch.split(fusion, [1, 1, 1], dim=1)
        return x1 * x1_weight + x2 * x2_weight + x3 * x3_weight


class feature_fusion(nn.Module):
    def __init__(self, inchannel):
        super(feature_fusion, self).__init__()
        self.fusion_1 = Conv(inchannel, inchannel, 1)
        self.fusion_2 = Conv(inchannel, inchannel, 1)
        self.fusion_3 = Conv(inchannel * 2, 2, 1)

    def forward(self, x1, x2):
        fusion = torch.softmax(self.fusion_3(torch.cat([self.fusion_1(x1), self.fusion_1(x2)], dim=1)), dim=1)
        x1_weight, x2_weight = torch.split(fusion, [1, 1], dim=1)
        return x1 * x1_weight + x2 * x2_weight


class BasicConv(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        if bn:
            self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                                  dilation=dilation, groups=groups, bias=False)
            self.bn = nn.BatchNorm2d(out_planes)
            self.relu = nn.SiLU() if relu else None
        else:
            self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                                  dilation=dilation, groups=groups, bias=True)
            self.bn = None
            self.relu = nn.SiLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class NonLocalBlockND(nn.Module):
    """
    调用过程
    NONLocalBlock2D(in_channels=32),
    super(NONLocalBlock2D, self).__init__(in_channels,
            inter_channels=inter_channels,
            dimension=2, sub_sample=sub_sample,
            bn_layer=bn_layer)
    """

    def __init__(self,
                 in_channels,
                 inter_channels=None,
                 dimension=2,
                 sub_sample=False,
                 bn_layer=True):
        super(NonLocalBlockND, self).__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample
        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            # 进行压缩得到channel个数
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        self.g = conv_nd(in_channels=self.in_channels,
                         out_channels=self.inter_channels,
                         kernel_size=1,
                         stride=1,
                         padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels,
                        out_channels=self.in_channels,
                        kernel_size=1,
                        stride=1,
                        padding=0), bn(self.in_channels))
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels,
                             out_channels=self.in_channels,
                             kernel_size=1,
                             stride=1,
                             padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels,
                             out_channels=self.inter_channels,
                             kernel_size=1,
                             stride=1,
                             padding=0)
        self.phi = conv_nd(in_channels=self.in_channels,
                           out_channels=self.inter_channels,
                           kernel_size=1,
                           stride=1,
                           padding=0)

        self.D = conv_nd(in_channels=self.in_channels,
                         out_channels=self.inter_channels,
                         kernel_size=1,
                         stride=1,
                         padding=0)
        self.S = conv_nd(in_channels=self.in_channels,
                         out_channels=self.inter_channels,
                         kernel_size=1,
                         stride=1,
                         padding=0)
        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)
            self.C = nn.Sequential(self.D, max_pool_layer)
            self.S = nn.Sequential(self.S, max_pool_layer)
        self.CAM = ChannelAttention(self.inter_channels)
        self.SP = SpatialAttention(self.inter_channels)

    def forward(self, x):
        '''
        :param x: (b, c,  h, w)
        :return:
        '''
        batch_size = x.size(0)
        s_x = self.D(x)
        s_x = s_x * self.CAM(s_x)
        s_x = s_x.view(batch_size, self.inter_channels, -1)
        s_x = s_x.permute(0, 2, 1)
        c_x = self.S(x)
        c_x = c_x * self.SP(c_x)
        c_x = c_x.view(batch_size, self.inter_channels, -1)
        c_x = c_x.permute(0, 2, 1)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)
        s_x = torch.matmul(f_div_C, s_x + c_x)
        y = torch.matmul(f_div_C, g_x)
        y = y + s_x
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z


class Conv_NonLocal(nn.Module):
    # Standard convolution·
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        # Standard convolution
        super(Conv_NonLocal, self).__init__()
        self.NonLocalBlockND = NonLocalBlockND(c1, c2)
        # self.NonLocal = NonLocalBlockND(in_channels=c1)
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.Hardswish() if act else nn.Identity()

    def forward(self, x):
        x = self.act(self.bn(self.conv(x)))
        return x


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.f1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu = nn.ReLU()
        self.f2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.f2(self.relu(self.f1(self.avg_pool(x))))
        max_out = self.f2(self.relu(self.f1(self.max_pool(x))))
        out = self.sigmoid(avg_out + max_out)
        return out


class SpatialAttention(nn.Module):
    def __init__(self, c1, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        # (特征图的大小-算子的size+2*padding)/步长+1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.conv2 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x1, x2):
        # 1*h*w
        avg_out1 = torch.mean(x1, dim=1, keepdim=True)
        max_out1, _ = torch.max(x1, dim=1, keepdim=True)
        avg_out2 = torch.mean(x2, dim=1, keepdim=True)
        max_out2, _ = torch.max(x2, dim=1, keepdim=True)
        x1 = torch.cat([avg_out1, max_out1], dim=1)
        x2 = torch.cat([avg_out2, max_out2], dim=1)
        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        weight1 = self.sigmoid(x1)

        weight2 = self.sigmoid(x2)
        return weight1,weight2


class Add_SN(nn.Module):
    #  Add two tensorss
    def __init__(self, arg):
        super(Add_SN, self).__init__()
        self.SimAM = SimAM()
        self.CAM_SAM = CAM_SAM(arg)

    def forward(self, x):
        x1 = x[0]
        x2 = x[1]
        x1 = self.SimAM(x1)
        x2 = self.SimAM(x2)
        x = self.CAM_SAM(x1, x2)
        return x


class CAM_SAM(nn.Module):
    def __init__(self, c1):
        super().__init__()
        self.CAM = ChannelAttention(c1)
        self.SAM = SpatialAttention(c1)

    def forward(self, x1, x2):
        x1 = x1 * self.CAM(x1)
        x2 = x2 * self.CAM(x2)
        weight1,weight2 = self.SAM(x1, x2)
        x =  weight1*x1+weight2*x2
        return x
#
#
# class Add2_SN(nn.Module):
#     def __init__(self, c1, index):
#         super().__init__()
#         self.index = index
#         self.SimAM = SimAM()
#         self.CAM_SAM = CAM_SAM(c1)
#         self.feature_fusion = feature_fusion(c1)
#
#     def forward(self, x):
#         if self.index == 0:
#             x1 = x[0]
#             x2 = x[1][0]
#             x1 = self.SimAM(x1)
#             x = self.CAM_SAM(x1, x2)
#             return x
#         elif self.index == 1:
#             x1 = x[0]
#             x2 = x[1][1]
#             x1 = self.SimAM(x1)
#             x = self.CAM_SAM(x1, x2)
#             return x