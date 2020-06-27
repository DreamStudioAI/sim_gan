import torch.nn as nn


class DCGenerator(nn.Module):
    def __init__(self, ngpu):
        super(DCGenerator, self).__init__()
        self.ngpu = ngpu
        ngf = 64
        self.main = nn.Sequential(
            # shape in = [N, 50, 1]
            nn.ConvTranspose1d(100, ngf * 32, 4, 1, 0, bias=True),
            nn.BatchNorm1d(ngf * 32),
            nn.ReLU(True),
            # shape in = [N, 64*4, 4]
            nn.ConvTranspose1d(ngf * 32, ngf * 16, 4, 1, 0, bias=True),
            nn.BatchNorm1d(ngf * 16),
            nn.ReLU(True),
            # shape in = [N, 64*2, 7]
            nn.ConvTranspose1d(ngf * 16, ngf * 8, 4, 2, 1, bias=True),
            nn.BatchNorm1d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose1d(ngf * 8, ngf * 4, 3, 2, 1, bias=True),
            nn.BatchNorm1d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose1d(ngf * 4, ngf * 2, 4, 2, 1, bias=True),
            nn.BatchNorm1d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose1d(ngf * 2, ngf, 4, 2, 1, bias=True),
            nn.BatchNorm1d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose1d(ngf, 1, 4, 2, 1, bias=True),
        )

    def forward(self, x):
        x = x.view(-1, 100, 1)
        x = self.main(x)
        x = x.view(-1, 216)
        return x


class DCDiscriminator(nn.Module):
    def __init__(self, ngpu):
        super(DCDiscriminator, self).__init__()
        self.ngpu = ngpu
        ndf = 64
        self.main = nn.Sequential(
        # input is (nc) x 64 x 64
        nn.Conv1d(in_channels=1, out_channels=ndf, kernel_size=4, stride=2, padding=1, bias=False),
        nn.LeakyReLU(0.2, inplace=True),
        # state size. (ndf) x 32 x 32
        nn.Conv1d(ndf, ndf * 2, 4, 2, 1, bias=False),
        nn.BatchNorm1d(ndf * 2),
        nn.LeakyReLU(0.2, inplace=True),
        # state size. (ndf*2) x 16 x 16
        nn.Conv1d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
        nn.BatchNorm1d(ndf * 4),
        nn.LeakyReLU(0.2, inplace=True),
        # state size. (ndf*4) x 8 x 8
        nn.Conv1d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
        nn.BatchNorm1d(ndf * 8),
        nn.LeakyReLU(0.2, inplace=True),

        nn.Conv1d(ndf * 8, ndf * 16, 4, 2, 1, bias=False),
        nn.BatchNorm1d(ndf * 16),
        nn.LeakyReLU(0.2, inplace=True),

        # state size. (ndf*8) x 4 x 4
        nn.Conv1d(ndf * 16, 1, 5, 2, 0, bias=False),
        nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(-1, 1, 216)
        return self.main(x)
