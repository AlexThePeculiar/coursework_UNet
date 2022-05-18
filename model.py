import torch
import torch.nn as nn
import torchvision.transforms.functional as TF


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNET(nn.Module):
    def __init__(
            self, in_channels=3, out_channels=1, features=[64,128,256,512]
    ):
        super(UNET, self).__init__()
        self.down_steps = nn.ModuleList()
        self.up_steps_scaling = nn.ModuleList()
        self.up_steps_conv = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        for feature in features: # going down
            self.down_steps.append(DoubleConv(in_channels, feature))
            in_channels = feature

        for feature in reversed(features): # going up
            self.up_steps_scaling.append(
                nn.ConvTranspose2d(2*feature, feature, kernel_size=2, stride=2)
            )
            self.up_steps_conv.append(DoubleConv(2 * feature, feature))

        self.bottom = DoubleConv(features[-1], 2*features[-1])
        self.final = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        # going down each step, adding to skip connections and pooling
        for down_step in self.down_steps:
            x = down_step(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottom(x) # bottom layers

        for up_scale, skip_conn, up_conv in\
                zip(self.up_steps_scaling, reversed(skip_connections), self.up_steps_conv):
            x = up_scale(x)
            if x.shape != skip_conn.shape:
                x = TF.resize(x, size=skip_conn.shape[2:])

            concat_skip = torch.cat((skip_conn, x), dim=1)
            x = up_conv(concat_skip)

        return self.final(x)


def test():
    x = torch.randn((1, 1, 161, 161))
    model = UNET(in_channels=1, out_channels=1)
    y = model(x)
    print("x shape:", x.shape)
    print("y shape:", y.shape)



if __name__ == "__main__":
    test()

