class FuseBlock(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = conv_bn(self.in_channels, self.out_channels, conv_sampler(kernel_size=1))
        self.conv2 = conv_bn(self.in_channels, self.out_channels, conv_sampler(kernel_size=3, stride=1))

    def forward(self, inputs):
        custom_print(inputs, 'FuseBlockStart')
        a = self.conv1(inputs)
        b = self.conv2(inputs)

        c = a + b
        custom_print(c, 'FuseBlockEnd')
        return c
