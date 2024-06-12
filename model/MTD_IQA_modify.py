from thop import profile, clever_format
from CLIP import clip
import torch.nn as nn
import torch

class MLP(nn.Module):
    def __init__(self, ch_in, ch_mid, ch_out, is_relu=True):
        super(MLP, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(ch_in, ch_mid),
            nn.ReLU()
        )
        if is_relu:
            self.fc2 = nn.Sequential(
                nn.Linear(ch_mid, ch_out),
                nn.ReLU()
            )
        else:
            self.fc2 = nn.Sequential(
                nn.Linear(ch_mid, ch_out)
            )

    def forward(self, v):
        v = self.fc1(v)
        v = self.fc2(v)
        return v



class MSAM(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.att = nn.Sequential(nn.Linear(in_dim, int(in_dim/2)),
                                 nn.ReLU(),
                                 nn.Linear(int(in_dim/2), in_dim))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, l, m, s, return_feats=False):
        """l,m,s表示大中小三个尺度"""
        feat = torch.stack([l, m, s], dim=1)
        attn = self.att(feat)
        attn = self.softmax(attn)
        attn_l, attn_m, attn_s = attn.chunk(3, dim=1)
        attn_l, attn_m, attn_s = attn_l.squeeze(dim=1), attn_m.squeeze(dim=1), attn_s.squeeze(dim=1)
        lms = attn_l * l + attn_m * m + attn_s * s
        if return_feats:
            return lms, dict(attn_l=attn_l, attn_m=attn_m, attn_s=attn_s, l=l, m=m, s=s)
        return lms


class MTD_IQA(nn.Module):
    def __init__(self, device, clip_net='RN50', in_size=1024):
        super(MTD_IQA, self).__init__()

        self.in_size = in_size
        self.base, _ = clip.load(clip_net, device=device)
        self.logit_scale = self.base.logit_scale
        self.encode_image = self.base.encode_image
        self.encode_text = self.base.encode_text
        self.MSAMLayer = MSAM(in_dim=in_size)
        self.MLP_q = MLP(ch_in=in_size, ch_mid=256, ch_out=1)
        self.MLP_a = MLP(ch_in=in_size, ch_mid=256, ch_out=1)

    def forward(self, img_l, img_m, img_s, tokens_c):
        img_l_feature = self.encode_image(img_l).to(torch.float32)
        img_m_feature = self.encode_image(img_m).to(torch.float32)
        img_s_feature = self.encode_image(img_s).to(torch.float32)

        img_feature = self.MSAMLayer(img_l_feature, img_m_feature, img_s_feature)
        con_feature = self.encode_text(tokens_c).to(torch.float32)

        logits_per_aes = self.MLP_a(img_feature)
        logits_per_qua = self.MLP_q(img_feature)

        logit_scale = self.logit_scale.exp()
        img_feature_n = img_feature / img_feature.norm(dim=1, keepdim=True)
        con_feature_n = con_feature / con_feature.norm(dim=1, keepdim=True)
        con_feature_n = con_feature_n.view(-1, 1, self.in_size)
        img_feature_n = img_feature_n.view(-1, 1, self.in_size)
        logits_per_con = [logit_scale * img_feature_n[i] @ con_feature_n[i].t() for i in range(con_feature_n.shape[0])]
        logits_per_con = torch.cat(logits_per_con, dim=0)

        return logits_per_qua, logits_per_con, logits_per_aes


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch_size = 1
    text_c = ["a beautiful women"]*batch_size
    token_c = torch.stack([clip.tokenize(prompt) for prompt in text_c]).to(device)
    x_m = torch.randn(batch_size, 3, 224, 224).to(device)
    x_l = torch.randn(batch_size, 3, 336, 336).to(device)
    x_s = torch.randn(batch_size, 3, 112, 112).to(device)

    model = MTD_IQA(device=device).to(device)
    pred_c = []
    pred_q = []
    pred_a = []
    pred_nss = []

    input_token_c = token_c.view(-1, 77)

    for param in model.parameters():
        param.requires_grad = True
    flops, params = profile(model, inputs=(x_l, x_m, x_s, input_token_c))
    print('Flops: % .4fG' % (2 * flops / 1000000000))
    print('params: % .4fM' % (params / 1000000))