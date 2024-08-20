from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import torch
from torch import nn

from modules.until_module import PreTrainedModel, AllGather, CrossEn
from modules.module_cross import CrossModel, CrossConfig, Transformer as TransformerClip

from modules.module_clip import CLIP, convert_weights
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import math
from torch import Tensor
from mamba_ssm import Mamba
from einops import rearrange
from functools import partial
import numpy as np
from flash_attn import flash_attn_qkvpacked_func, flash_attn_func
from typing import Tuple, Union, Optional
import torch.nn.functional as F

logger = logging.getLogger(__name__)
allgather = AllGather.apply

class Mamba_Out(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        conv_bias=True,
        bias=False,
        use_fast_path=True,  # Fused kernel options
        layer_idx=None,
        device=None,
        dtype=None,
        bimamba_type="none",
        if_devide_out=False,
        init_layer_scale=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx
        self.bimamba_type = bimamba_type
        self.if_devide_out = if_devide_out

        self.init_layer_scale = init_layer_scale
        if init_layer_scale is not None:
            self.gamma = nn.Parameter(init_layer_scale * torch.ones((d_model)), requires_grad=True)

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.activation = "silu"
        self.act = nn.SiLU()
        self.conv1d_b = nn.Conv1d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                groups=self.d_inner,
                padding=d_conv - 1,
                **factory_kwargs,
            )
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

    def forward(self, hidden_states, inference_params=None):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        batch, seqlen, dim = hidden_states.shape
        # We do matmul and transpose BLH -> HBL at the same time
        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")
        
        if self.use_fast_path and inference_params is None:  # Doesn't support outputting the states
            x, z = xz.chunk(2, dim=1)
            out = self.conv1d(x)
            x, z = xz.flip([-1]).chunk(2, dim=1)
            out_b = self.conv1d_b(x)
            if not self.if_devide_out:
                out = F.linear(rearrange(out + out_b.flip([-1]), "b d l -> b l d"), self.out_proj.weight, self.out_proj.bias)
              
        if self.init_layer_scale is not None:
            out = out * self.gamma    
        return out


class MultiheadAttention_flash(nn.MultiheadAttention):
    
    def forward(self, query: Tensor, key: Tensor, value: Tensor, key_padding_mask: Optional[Tensor] = None,
                need_weights: bool = True, attn_mask: Optional[Tensor] = None):

        return flash_attn_func(
                q=query, k=key, v=value, dropout_p=0.0, softmax_scale=None, causal=False,
                window_size=(-1, -1), alibi_slopes=None, deterministic=False)


class LayerNorm_conv(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""
    def __init__(self, normalized_shape):
        super().__init__(normalized_shape=normalized_shape)

    def forward(self, x: torch.Tensor):
        x = x.permute(0,2,3,1)
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))# add ssf
        return ret.type(orig_type).permute(0,3,1,2)

class CLIP4ClipPreTrainedModel(PreTrainedModel, nn.Module):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """
    def __init__(self, cross_config, *inputs, **kwargs):
        super(CLIP4ClipPreTrainedModel, self).__init__(cross_config)
        self.cross_config = cross_config
        self.clip = None
        self.cross = None
 

    @classmethod
    def from_pretrained(cls, cross_model_name, state_dict=None, cache_dir=None, type_vocab_size=2, *inputs, **kwargs):

        task_config = None
        if "task_config" in kwargs.keys():
            task_config = kwargs["task_config"]
            if not hasattr(task_config, "local_rank"):
                task_config.__dict__["local_rank"] = 0
            elif task_config.local_rank == -1:
                task_config.local_rank = 0

        if state_dict is None: state_dict = {}
        pretrained_clip_name = "ViT-B/32"
        if hasattr(task_config, 'pretrained_clip_name'):
            pretrained_clip_name = task_config.pretrained_clip_name
        clip_state_dict = CLIP.get_config(pretrained_clip_name=pretrained_clip_name)
        for key, val in clip_state_dict.items():
            new_key = "clip." + key
            if new_key not in state_dict:
                state_dict[new_key] = val.clone()

        cross_config, _ = CrossConfig.get_config(cross_model_name, cache_dir, type_vocab_size, state_dict=None, task_config=task_config)

        model = cls(cross_config, clip_state_dict, *inputs, **kwargs)

        ## ===> Initialization trick [HARD CODE]
        if model.linear_patch == "3d":
            contain_conv2 = False
            for key in state_dict.keys():
                if key.find("visual.conv2.weight") > -1:
                    contain_conv2 = True
                    break
            if contain_conv2 is False and hasattr(model.clip.visual, "conv2"):
                cp_weight = state_dict["clip.visual.conv1.weight"].clone()
                kernel_size = model.clip.visual.conv2.weight.size(2)
                conv2_size = model.clip.visual.conv2.weight.size()
                conv2_size = list(conv2_size)

                left_conv2_size = conv2_size.copy()
                right_conv2_size = conv2_size.copy()
                left_conv2_size[2] = (kernel_size - 1) // 2
                right_conv2_size[2] = kernel_size - 1 - left_conv2_size[2]

                left_zeros, right_zeros = None, None
                if left_conv2_size[2] > 0:
                    left_zeros = torch.zeros(*tuple(left_conv2_size), dtype=cp_weight.dtype, device=cp_weight.device)
                if right_conv2_size[2] > 0:
                    right_zeros = torch.zeros(*tuple(right_conv2_size), dtype=cp_weight.dtype, device=cp_weight.device)

                cat_list = []
                if left_zeros != None: cat_list.append(left_zeros)
                cat_list.append(cp_weight.unsqueeze(2))
                if right_zeros != None: cat_list.append(right_zeros)
                cp_weight = torch.cat(cat_list, dim=2)

                state_dict["clip.visual.conv2.weight"] = cp_weight

        if model.sim_header == 'tightTransf':
            contain_cross = False
            for key in state_dict.keys():
                if key.find("cross.transformer") > -1:
                    contain_cross = True
                    break
            if contain_cross is False:
                for key, val in clip_state_dict.items():
                    if key == "positional_embedding":
                        state_dict["cross.embeddings.position_embeddings.weight"] = val.clone()
                        continue
                    if key.find("transformer.resblocks") == 0:
                        num_layer = int(key.split(".")[2])

                        # cut from beginning
                        if num_layer < task_config.cross_num_hidden_layers:
                            state_dict["cross."+key] = val.clone()
                            continue

        if model.sim_header == "seqLSTM" or model.sim_header == "seqTransf":
            contain_frame_position = False
            for key in state_dict.keys():
                if key.find("frame_position_embeddings") > -1:
                    contain_frame_position = True
                    break
            if contain_frame_position is False:
                for key, val in clip_state_dict.items():
                    if key == "positional_embedding":
                        state_dict["frame_position_embeddings.weight"] = val.clone()
                        continue
                    if (model.sim_header == "seqTransf" ) and key.find("transformer.resblocks") == 0:
                        num_layer = int(key.split(".")[2])
                        # cut from beginning
                        if num_layer < task_config.cross_num_hidden_layers:
                            state_dict[key.replace("transformer.", "transformerClip.")] = val.clone()
                            continue
        ## <=== End of initialization trick

        if state_dict is not None:
            model = cls.init_preweight(model, state_dict, task_config=task_config)

        return model

def show_log(task_config, info):
    if task_config is None or task_config.local_rank == 0:
        logger.warning(info)

def update_attr(target_name, target_config, target_attr_name, source_config, source_attr_name, default_value=None):
    if hasattr(source_config, source_attr_name):
        if default_value is None or getattr(source_config, source_attr_name) != default_value:
            setattr(target_config, target_attr_name, getattr(source_config, source_attr_name))
            show_log(source_config, "Set {}.{}: {}.".format(target_name,
                                                            target_attr_name, getattr(target_config, target_attr_name)))
    return target_config

def check_attr(target_name, task_config):
    return hasattr(task_config, target_name) and task_config.__dict__[target_name]


class Mamba_head(nn.Module):
    def __init__(self, embed_dim, layer_num=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.mamba = Mamba(self.embed_dim, d_conv=4, bimamba_type='v2', use_fast_path=True, expand=1)
        # self.mamba_out = Mamba_Out(self.embed_dim, d_conv=1, bimamba_type='v2', use_fast_path=True, expand=1)
        # self.transformer = nn.MultiheadAttention(self.embed_dim, self.embed_dim // 64)
        # self.flash_attn = MultiheadAttention_flash(self.embed_dim, self.embed_dim // 64)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim)
        # self.layer_norm1 = RMSNorm(hidden_size=self.embed_dim)

        self.proj_drop = nn.Dropout(layer_num)
        self.temporal_fc = nn.Linear(self.embed_dim, self.embed_dim)
        nn.init.constant_(self.temporal_fc.weight, 0.)
        nn.init.constant_(self.temporal_fc.bias, 0.)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask=None,
        causal_attention_mask=None,
    ):
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        # hidden_states = self.transformer((hidden_states, None))[0] [L,B,D]
        hidden_states = self.mamba(hidden_states)
        # hidden_states = self.mamba_out(hidden_states)
        # hidden_states = self.flash_attn(hidden_states, hidden_states, hidden_states, need_weights=False, attn_mask=None)
        res_temporal = self.proj_drop(hidden_states.contiguous())
        
        res_temporal = self.temporal_fc(res_temporal)
        hidden_states = residual + res_temporal
        outputs = hidden_states

        return outputs

class CLIP4Clip(CLIP4ClipPreTrainedModel):
    def __init__(self, cross_config, clip_state_dict, task_config):
        super(CLIP4Clip, self).__init__(cross_config)
        self.task_config = task_config
        self.ignore_video_index = -1

        assert self.task_config.max_words + self.task_config.max_frames <= cross_config.max_position_embeddings

        self._stage_one = True
        self._stage_two = False

        show_log(task_config, "Stage-One:{}, Stage-Two:{}".format(self._stage_one, self._stage_two))

        self.loose_type = False
        if self._stage_one and check_attr('loose_type', self.task_config):
            self.loose_type = True
            show_log(task_config, "Test retrieval by loose type.")

        # CLIP Encoders: From OpenAI: CLIP [https://github.com/openai/CLIP] ===>
        vit = "visual.proj" in clip_state_dict
        assert vit
        if vit:
            vision_width = clip_state_dict["visual.conv1.weight"].shape[0]
            vision_layers = len(
                [k for k in clip_state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
            vision_patch_size = clip_state_dict["visual.conv1.weight"].shape[-1]
            grid_size = round((clip_state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
            image_resolution = vision_patch_size * grid_size
        else:
            counts: list = [len(set(k.split(".")[2] for k in clip_state_dict if k.startswith(f"visual.layer{b}"))) for b in
                            [1, 2, 3, 4]]
            vision_layers = tuple(counts)
            vision_width = clip_state_dict["visual.layer1.0.conv1.weight"].shape[0]
            output_width = round((clip_state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
            vision_patch_size = None
            assert output_width ** 2 + 1 == clip_state_dict["visual.attnpool.positional_embedding"].shape[0]
            image_resolution = output_width * 32

        embed_dim = clip_state_dict["text_projection"].shape[1]
        context_length = clip_state_dict["positional_embedding"].shape[0]
        vocab_size = clip_state_dict["token_embedding.weight"].shape[0]
        transformer_width = clip_state_dict["ln_final.weight"].shape[0]
        transformer_heads = transformer_width // 64
        transformer_layers = len(set(k.split(".")[2] for k in clip_state_dict if k.startswith(f"transformer.resblocks")))

        show_log(task_config, "\t embed_dim: {}".format(embed_dim))
        show_log(task_config, "\t image_resolution: {}".format(image_resolution))
        show_log(task_config, "\t vision_layers: {}".format(vision_layers))
        show_log(task_config, "\t vision_width: {}".format(vision_width))
        show_log(task_config, "\t vision_patch_size: {}".format(vision_patch_size))
        show_log(task_config, "\t context_length: {}".format(context_length))
        show_log(task_config, "\t vocab_size: {}".format(vocab_size))
        show_log(task_config, "\t transformer_width: {}".format(transformer_width))
        show_log(task_config, "\t transformer_heads: {}".format(transformer_heads))
        show_log(task_config, "\t transformer_layers: {}".format(transformer_layers))

        self.linear_patch = '2d'
        if hasattr(task_config, "linear_patch"):
            self.linear_patch = task_config.linear_patch
            show_log(task_config, "\t\t linear_patch: {}".format(self.linear_patch))

        # use .float() to avoid overflow/underflow from fp16 weight. https://github.com/openai/CLIP/issues/40
        cut_top_layer = 0
        show_log(task_config, "\t cut_top_layer: {}".format(cut_top_layer))
        self.clip = CLIP(
            embed_dim,
            image_resolution, vision_layers-cut_top_layer, vision_width, vision_patch_size,
            context_length, vocab_size, transformer_width, transformer_heads, transformer_layers-cut_top_layer,
            linear_patch=self.linear_patch
        ).float()

        for key in ["input_resolution", "context_length", "vocab_size"]:
            if key in clip_state_dict:
                del clip_state_dict[key]

        convert_weights(self.clip)
        # <=== End of CLIP Encoders

        self.sim_header = 'meanP'
        if hasattr(task_config, "sim_header"):
            self.sim_header = task_config.sim_header
            show_log(task_config, "\t sim_header: {}".format(self.sim_header))
        if self.sim_header == "tightTransf": assert self.loose_type is False

        cross_config.max_position_embeddings = context_length
        if self.loose_type is False:
            # Cross Encoder ===>
            cross_config = update_attr("cross_config", cross_config, "num_hidden_layers", self.task_config, "cross_num_hidden_layers")
            self.cross = CrossModel(cross_config)
            # <=== End of Cross Encoder
            self.similarity_dense = nn.Linear(cross_config.hidden_size, 1)

        if self.sim_header == "seqLSTM" or self.sim_header == "seqTransf":
            self.frame_position_embeddings = nn.Embedding(cross_config.max_position_embeddings, cross_config.hidden_size)
        if self.sim_header == "seqTransf":
            self.transformerClip = TransformerClip(width=transformer_width, layers=self.task_config.cross_num_hidden_layers,
                                                   heads=transformer_heads, )
            num_words = self.task_config.max_words
            num_frames = self.task_config.max_frames

            self.global_mat_weight = nn.parameter.Parameter(torch.eye(embed_dim), requires_grad=True)

            # for cross-grained constrast weights
            self.word_logit_weight = nn.parameter.Parameter(torch.eye(num_words), requires_grad=True)
            self.frame_logit_weight = nn.parameter.Parameter(torch.eye(num_frames), requires_grad=True)
            self.loss_fct = CrossEn()
        
            self.local_mat_weight = nn.parameter.Parameter(torch.eye(embed_dim), requires_grad=True)
            self.frame_mat_weight = nn.parameter.Parameter(torch.eye(num_frames), requires_grad=True)
            self.word_mat_weight = nn.parameter.Parameter(torch.eye(num_words), requires_grad=True)
            self.frame_mat_weight2 = nn.parameter.Parameter(torch.eye(num_frames), requires_grad=True)
            self.word_mat_weight2 = nn.parameter.Parameter(torch.eye(num_words), requires_grad=True)
        if self.sim_header == "seqLSTM":
            self.lstm_visual = nn.LSTM(input_size=cross_config.hidden_size, hidden_size=cross_config.hidden_size,
                                       batch_first=True, bidirectional=False, num_layers=1)
       
        if self.sim_header == "MUSE":
            self.mamba_stages = []
            scale_factors = [0.5, 1.0, 2.0] # feature scales used
            dim = transformer_width
            for idx, scale in enumerate(scale_factors):
                out_dim = dim
                out_channels = dim
                if scale == 4.0:
                    layers = [
                        nn.ConvTranspose2d(dim, dim // 2, kernel_size=2, stride=2),
                        LayerNorm_conv(dim // 2),
                        nn.GELU(),
                        nn.ConvTranspose2d(dim // 2, dim // 4, kernel_size=2, stride=2),
                    ]
                    out_dim = dim // 4
                elif scale == 2.0:
                    layers = [nn.ConvTranspose2d(dim, dim // 2, kernel_size=2, stride=2)]
                    out_dim = dim // 2
                elif scale == 1.0:
                    layers = []
                elif scale == 0.5:
                    layers = [nn.MaxPool2d(kernel_size=2, stride=2)]
                elif scale == 0.25:
                    layers = [nn.MaxPool2d(kernel_size=4, stride=4)]
                else:
                    raise NotImplementedError(f"scale_factor={scale} is not supported yet.")

                layers.extend(
                    [
                        nn.Conv2d(
                            out_dim,
                            out_channels,
                            kernel_size=1,
                        ),
                        LayerNorm_conv(out_channels),
                        nn.GELU(),
                        nn.Conv2d(
                            out_channels,
                            out_channels,
                            kernel_size=3,
                            padding=1,
                        ),
                        LayerNorm_conv(out_channels)
                    ]
                )
                layers = nn.Sequential(*layers).cuda()
                self.mamba_stages.append(layers)
            depth = 4
            dpr = np.linspace(0, 0.1, depth)
            self.MS_mamba = nn.ModuleList([Mamba_head(transformer_width, dpr[i]) for i in range(depth)]).cuda()
          
            ### Add for using X-CLIP
            # num_words = self.task_config.max_words
            # num_frames = self.task_config.max_frames

            # self.global_mat_weight = nn.parameter.Parameter(torch.eye(embed_dim), requires_grad=True)

            # # for cross-grained constrast weights
            # self.word_logit_weight = nn.parameter.Parameter(torch.eye(num_words), requires_grad=True)
            # self.frame_logit_weight = nn.parameter.Parameter(torch.eye(num_frames), requires_grad=True)
           
            # self.local_mat_weight = nn.parameter.Parameter(torch.eye(embed_dim), requires_grad=True)
            # self.frame_mat_weight = nn.parameter.Parameter(torch.eye(num_frames), requires_grad=True)
            # self.word_mat_weight = nn.parameter.Parameter(torch.eye(num_words), requires_grad=True)
            # self.frame_mat_weight2 = nn.parameter.Parameter(torch.eye(num_frames), requires_grad=True)
            # self.word_mat_weight2 = nn.parameter.Parameter(torch.eye(num_words), requires_grad=True)
            self.loss_fct = CrossEn()

        self.apply(self.init_weights)

    def forward(self, input_ids, token_type_ids, attention_mask, video, video_mask=None):
        input_ids = input_ids.view(-1, input_ids.shape[-1])
        token_type_ids = token_type_ids.view(-1, token_type_ids.shape[-1])
        attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
        video_mask = video_mask.view(-1, video_mask.shape[-1])

        # T x 3 x H x W
        video = torch.as_tensor(video).float()
        b, pair, bs, ts, channel, h, w = video.shape
        video = video.view(b * pair * bs * ts, channel, h, w)
        video_frame = bs * ts

        (sequence_output, seq_features), visual_output = self.get_sequence_visual_output(input_ids, token_type_ids, attention_mask,
                                                                         video, video_mask, shaped=True, video_frame=video_frame)

        if self.training:
            loss = 0.
            sim_matrix, *_tmp = self.get_similarity_logits(sequence_output, seq_features, visual_output, attention_mask, video_mask,
                                                    shaped=True, loose_type=self.loose_type)
            sim_loss1 = self.loss_fct(sim_matrix)
            sim_loss2 = self.loss_fct(sim_matrix.T)
            sim_loss = (sim_loss1 + sim_loss2) / 2
            loss += sim_loss

            return loss
        else:
            return None

    def get_sequence_output(self, input_ids, token_type_ids, attention_mask, shaped=False):
        if shaped is False:
            input_ids = input_ids.view(-1, input_ids.shape[-1])
            token_type_ids = token_type_ids.view(-1, token_type_ids.shape[-1])
            attention_mask = attention_mask.view(-1, attention_mask.shape[-1])

        bs_pair = input_ids.size(0)
        sequence_hidden, seq_features = self.clip.encode_text(input_ids, return_hidden=True)
        sequence_hidden, seq_features = sequence_hidden.float(), seq_features.float()
        sequence_hidden = sequence_hidden.view(bs_pair, -1, sequence_hidden.size(-1))

        return sequence_hidden, seq_features

    def get_visual_output(self, video, video_mask, shaped=False, video_frame=-1):
        if shaped is False:
            video_mask = video_mask.view(-1, video_mask.shape[-1])
            video = torch.as_tensor(video).float()
            b, pair, bs, ts, channel, h, w = video.shape
            video = video.view(b * pair * bs * ts, channel, h, w)
            video_frame = bs * ts

        bs_pair = video_mask.size(0)
        visual_hidden = self.clip.encode_image(video, video_frame=video_frame).float()
        visual_hidden = visual_hidden.view(bs_pair, -1, visual_hidden.size(-1))

        return visual_hidden

    def get_sequence_visual_output(self, input_ids, token_type_ids, attention_mask, video, video_mask, shaped=False, video_frame=-1):
        if shaped is False:
            input_ids = input_ids.view(-1, input_ids.shape[-1])
            token_type_ids = token_type_ids.view(-1, token_type_ids.shape[-1])
            attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
            video_mask = video_mask.view(-1, video_mask.shape[-1])

            video = torch.as_tensor(video).float()
            b, pair, bs, ts, channel, h, w = video.shape
            video = video.view(b * pair * bs * ts, channel, h, w)
            video_frame = bs * ts

        sequence_output, seq_features = self.get_sequence_output(input_ids, token_type_ids, attention_mask, shaped=True)
        visual_output = self.get_visual_output(video, video_mask, shaped=True, video_frame=video_frame)

        return (sequence_output, seq_features), visual_output

    def _get_cross_output(self, sequence_output, visual_output, attention_mask, video_mask):

        concat_features = torch.cat((sequence_output, visual_output), dim=1)  # concatnate tokens and frames
        concat_mask = torch.cat((attention_mask, video_mask), dim=1)
        text_type_ = torch.zeros_like(attention_mask)
        video_type_ = torch.ones_like(video_mask)
        concat_type = torch.cat((text_type_, video_type_), dim=1)

        cross_layers, pooled_output = self.cross(concat_features, concat_type, concat_mask, output_all_encoded_layers=True)
        cross_output = cross_layers[-1]

        return cross_output, pooled_output, concat_mask

    def _mean_pooling_for_similarity_sequence(self, sequence_output, attention_mask):
        attention_mask_un = attention_mask.to(dtype=torch.float).unsqueeze(-1)
        attention_mask_un[:, 0, :] = 0.
        sequence_output = sequence_output * attention_mask_un
        text_out = torch.sum(sequence_output, dim=1) / torch.sum(attention_mask_un, dim=1, dtype=torch.float)
        return text_out

    def _mean_pooling_for_similarity_visual(self, visual_output, video_mask,):
        video_mask_un = video_mask.to(dtype=torch.float).unsqueeze(-1)
        visual_output = visual_output * video_mask_un
        video_mask_un_sum = torch.sum(video_mask_un, dim=1, dtype=torch.float)
        video_mask_un_sum[video_mask_un_sum == 0.] = 1.
        video_out = torch.sum(visual_output, dim=1) / video_mask_un_sum
        return video_out

    def _mean_pooling_for_similarity(self, sequence_output, visual_output, attention_mask, video_mask,):
        text_out = self._mean_pooling_for_similarity_sequence(sequence_output, attention_mask)
        video_out = self._mean_pooling_for_similarity_visual(visual_output, video_mask)

        return text_out, video_out


    def _loose_similarity(self, sequence_output, seq_features, visual_output, attention_mask, video_mask, sim_header="meanP"):
        sequence_output, visual_output = sequence_output.contiguous(), visual_output.contiguous()

        if sim_header == "meanP":
            # Default: Parameter-free type
            B, TL, C = visual_output.shape
            T = 12
            L = TL // T
            visual_output = visual_output.view(B, T, L, C)
            visual_output = visual_output[:,:,0,:].contiguous()
        elif sim_header == "seqLSTM":
            # Sequential type: LSTM
            visual_output_original = visual_output
            visual_output = pack_padded_sequence(visual_output, torch.sum(video_mask, dim=-1).cpu(),
                                                 batch_first=True, enforce_sorted=False)
            visual_output, _ = self.lstm_visual(visual_output)
            if self.training: self.lstm_visual.flatten_parameters()
            visual_output, _ = pad_packed_sequence(visual_output, batch_first=True)
            visual_output = torch.cat((visual_output, visual_output_original[:, visual_output.size(1):, ...].contiguous()), dim=1)
            visual_output = visual_output + visual_output_original
        elif sim_header == "seqTransf":
            # Sequential type: Transformer Encoder
            B, TL, C = visual_output.shape
            T = 12
            L = TL // T
            visual_output = visual_output.view(B, T, L, C)
            visual_output = visual_output[:,:,0,:]
            visual_output_original = visual_output #[32, 12, 512]
            
            seq_length = visual_output.size(1)
            position_ids = torch.arange(seq_length, dtype=torch.long, device=visual_output.device)
            position_ids = position_ids.unsqueeze(0).expand(visual_output.size(0), -1)
            frame_position_embeddings = self.frame_position_embeddings(position_ids)
            visual_output = visual_output + frame_position_embeddings

            extended_video_mask = (1.0 - video_mask.unsqueeze(1)) * -1000000.0
            extended_video_mask = extended_video_mask.expand(-1, video_mask.size(1), -1)
            visual_output = visual_output.permute(1, 0, 2)  # NLD -> LND
            visual_output = self.transformerClip(visual_output, extended_video_mask)
            visual_output = visual_output.permute(1, 0, 2)  # LND -> NLD
            visual_output = visual_output + visual_output_original
        elif sim_header == "MUSE":
            B, TL, C = visual_output.shape
            L = 50
            T = TL // L
            H = W = int(math.sqrt(L-1))

            visual_output = visual_output.view(B, T, L, C)        
            visual_output_original = visual_output[:,:,0,:]
            visual_mamba = visual_output[:,:,1:,:].reshape(B*T, H, W, C).permute(0,3,1,2)

            agg_mode = "scale_wise"
            assert agg_mode in ["scale_wise", "spatial_wise", "frame_wise"] 
            visual_mamba_ms = []
            for stage in self.mamba_stages:
                visual_mamba_ms.append(stage(visual_mamba).view(B, T, C, -1).permute(0,1,3,2))
            if agg_mode == "scale_wise":
                visual_mamba_ms = [vi.view(B, -1, C) for vi in visual_mamba_ms]
                visual_mamba_st = torch.cat(visual_mamba_ms, dim=1)
                visual_mamba_output = torch.cat((visual_output_original, visual_mamba_st), dim=1)
                # visual_mamba_output = visual_output_original
            elif agg_mode == "spatial_wise":
                visual_mamba_ms = [vi.mean(dim=1).squeeze() for vi in visual_mamba_ms]
                visual_mamba_st = torch.cat(visual_mamba_ms, dim=1)
                visual_mamba_output = torch.cat((visual_output_original, visual_mamba_st), dim=1)
            elif agg_mode == "frame_wise":
                visual_mamba_output = []
                for t in range(T):
                    visual_mamba_output.append(visual_output_original[:,t,:].unsqueeze(1))
                    for vi in visual_mamba_ms:
                        visual_mamba_output.append(vi[:,t,:])
                visual_mamba_output = torch.cat(visual_mamba_output, dim=1)

            for layer in range(len(self.MS_mamba)):
                visual_mamba_output = self.MS_mamba[layer](visual_mamba_output)
            # visual_mamba_output = self.transformerClip(visual_mamba_output, None)
            if agg_mode == "frame_wise":
                visual_output = visual_mamba_output[:, ::visual_mamba_output.shape[1] // T, :].contiguous()
            else:
                visual_output = visual_mamba_output[:, :T, :].contiguous() #+ visual_output_original 
        # ## Using X-CLIP
        
        # video_output = visual_output / visual_output.norm(dim=-1, keepdim=True)
        # video_output = self._mean_pooling_for_similarity_visual(video_output, video_mask)
        # video_output = video_output / video_output.norm(dim=-1, keepdim=True) 
        # frame_features = visual_output / visual_output.norm(dim=-1, keepdim=True)

        #  # sentence-level textual feature
        # sentence_output = sequence_output.squeeze(1)
        # sentence_output  = sentence_output / sentence_output.norm(dim=-1, keepdim=True)          # [bs, dim]
        
        # # word-level textual features
        # word_features = seq_features / seq_features.norm(dim=-1, keepdim=True)                   # [bs, num_words, dim]

        # logit_scale = self.clip.logit_scale.exp()
        # if self.training:
        #     video_output = allgather(video_output, self.task_config)
        #     frame_features = allgather(frame_features, self.task_config)
        #     sentence_output = allgather(sentence_output, self.task_config)
        #     word_features = allgather(word_features, self.task_config)         
        #     # video_mask = allgather(video_mask, self.task_config)
        #     # visual_output = allgather(visual_output, self.task_config)
        #     # sequence_output = allgather(sequence_output, self.task_config)
        #     torch.distributed.barrier()

        # # retrieve_logits = logit_scale * torch.einsum('ad,abd->ab', [t_feat, v_feat])

        # # video-sentence score 
        # # video_sentence_logits = logit_scale * torch.einsum('ad,abd->ab', [t_feat, v_feat])
        # video_sentence_logits = logit_scale * torch.matmul(torch.matmul(sentence_output, self.global_mat_weight), video_output.t())
        # # video-word score
        # video_word_logits = logit_scale * torch.sum(torch.matmul(word_features, video_output.t()) \
        #     * torch.matmul(torch.softmax(torch.matmul(word_features, video_output.t()) / 1e-2, dim=1).permute(0,2,1), self.word_logit_weight).permute(0,2,1), dim=1)

        # # sentence-frame score 
        # sentence_frame_logits = logit_scale * torch.sum(torch.matmul(sentence_output, frame_features.permute(0, 2, 1)) \
        #     * torch.matmul(torch.softmax(torch.matmul(sentence_output, frame_features.permute(0, 2, 1)) / 1e-2, dim=-1), self.frame_logit_weight), dim=-1).t()

        # # frame-word score
        # frame_word_logits = logit_scale * self._attenion_over_fine_grained_sim_matrix(word_features, frame_features)

        # logits = (video_sentence_logits + video_word_logits + sentence_frame_logits + frame_word_logits) / 4
        # # print(logits.shape)
        # return logits
            

        if self.training:
            visual_output = allgather(visual_output, self.task_config)
            video_mask = allgather(video_mask, self.task_config)
            sequence_output = allgather(sequence_output, self.task_config)
            torch.distributed.barrier()
            
        # self.curv.data = torch.clamp(self.curv.data, **self._curv_minmax)
        # _curv = self.curv.exp()

        # Clamp scaling factors such that they do not up-scale the feature norms.
        # Once `exp(scale) = 1`, they can simply be removed during inference.
        # self.visual_alpha.data = torch.clamp(self.visual_alpha.data, max=0.0)
        # self.textual_alpha.data = torch.clamp(self.textual_alpha.data, max=0.0)

        visual_output = visual_output / visual_output.norm(dim=-1, keepdim=True)
        # visual_mamba_output = visual_mamba_output / visual_mamba_output.norm(dim=-1, keepdim=True)
        #[128,12,512]
      
        visual_output = self._mean_pooling_for_similarity_visual(visual_output, video_mask)
        visual_output = visual_output / visual_output.norm(dim=-1, keepdim=True)

        sequence_output = sequence_output.squeeze(1)
        sequence_output = sequence_output / sequence_output.norm(dim=-1, keepdim=True)
        # [128,512]

        ### Using EMCL
        # text_feat = sequence_output.squeeze()
        # video_feat = visual_output
        # temp = 5
        # v_weight = torch.einsum('ad,bvd->abv', [text_feat, video_feat])
        # v_weight = torch.softmax(v_weight /temp, dim=-1)
        # v_weight = torch.einsum('abv,bv->abv', [v_weight, video_mask])
        # video_feat = torch.einsum('abv,bvd->abd', [v_weight, video_feat])
            
        # t_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
        # v_feat = video_feat / video_feat.norm(dim=-1, keepdim=True)
        # logit_scale = self.clip.logit_scale.exp()
        # retrieve_logits = logit_scale * torch.einsum('ad,abd->ab', [t_feat, v_feat])
        
        logit_scale = self.clip.logit_scale.exp()
        retrieve_logits = logit_scale * torch.matmul(sequence_output, visual_output.t())
        return retrieve_logits
    def _attenion_over_fine_grained_sim_matrix(self, word_features, frame_features):
        bs_video, num_frames, dim_video = frame_features.shape
        bs_text, num_words, dim_text = word_features.shape
        fine_grained_sim_scores = torch.matmul(torch.matmul(word_features.view(-1, dim_text), self.local_mat_weight), frame_features.view(-1, dim_video).t()).view(bs_text, num_words, bs_video, num_frames)  # [bs_text, num_words, bs_video, num_frames]

        word_level_logit = torch.sum(torch.matmul(torch.softmax(fine_grained_sim_scores/1e-2, dim=1).permute(0,2,3,1), self.word_mat_weight).permute(0,3,1,2) * fine_grained_sim_scores, dim=1)               # [bs_text, bs_video, num_frames]
        frame_level_logit = torch.sum(torch.matmul(torch.softmax(fine_grained_sim_scores/1e-2, dim=-1), self.frame_mat_weight) * fine_grained_sim_scores, dim=-1)                                             # [bs_text, num_words, bs_video]

        sent2frame_logits = torch.sum(torch.matmul(torch.softmax(word_level_logit/1e-2, dim=-1),self.frame_mat_weight2) * word_level_logit, dim=-1)                                # [bs_text, bs_video]
        video2word_logits = torch.sum(torch.matmul(torch.softmax(frame_level_logit/1e-2, dim=1).permute(0,2,1), self.word_mat_weight2).permute(0,2,1) * frame_level_logit, dim=1)  # [bs_text, bs_video]

        return (sent2frame_logits + video2word_logits) / 2

    def _cross_similarity(self, sequence_output, visual_output, attention_mask, video_mask):
        sequence_output, visual_output = sequence_output.contiguous(), visual_output.contiguous()

        b_text, s_text, h_text = sequence_output.size()
        b_visual, s_visual, h_visual = visual_output.size()

        retrieve_logits_list = []

        step_size = b_text      # set smaller to reduce memory cost
        split_size = [step_size] * (b_text // step_size)
        release_size = b_text - sum(split_size)
        if release_size > 0:
            split_size += [release_size]

        # due to clip text branch retrun the last hidden
        attention_mask = torch.ones(sequence_output.size(0), 1)\
            .to(device=attention_mask.device, dtype=attention_mask.dtype)

        sequence_output_splits = torch.split(sequence_output, split_size, dim=0)
        attention_mask_splits = torch.split(attention_mask, split_size, dim=0)
        for i in range(len(split_size)):
            sequence_output_row = sequence_output_splits[i]
            attention_mask_row = attention_mask_splits[i]
            sequence_output_l = sequence_output_row.unsqueeze(1).repeat(1, b_visual, 1, 1)
            sequence_output_l = sequence_output_l.view(-1, s_text, h_text)
            attention_mask_l = attention_mask_row.unsqueeze(1).repeat(1, b_visual, 1)
            attention_mask_l = attention_mask_l.view(-1, s_text)

            step_truth = sequence_output_row.size(0)
            visual_output_r = visual_output.unsqueeze(0).repeat(step_truth, 1, 1, 1)
            visual_output_r = visual_output_r.view(-1, s_visual, h_visual)
            video_mask_r = video_mask.unsqueeze(0).repeat(step_truth, 1, 1)
            video_mask_r = video_mask_r.view(-1, s_visual)

            cross_output, pooled_output, concat_mask = \
                self._get_cross_output(sequence_output_l, visual_output_r, attention_mask_l, video_mask_r)
            retrieve_logits_row = self.similarity_dense(pooled_output).squeeze(-1).view(step_truth, b_visual)

            retrieve_logits_list.append(retrieve_logits_row)

        retrieve_logits = torch.cat(retrieve_logits_list, dim=0)
        return retrieve_logits

    def get_similarity_logits(self, sequence_output, seq_features, visual_output, attention_mask, video_mask, shaped=False, loose_type=False):
        if shaped is False:
            attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
            video_mask = video_mask.view(-1, video_mask.shape[-1])

        contrastive_direction = ()
        if loose_type:
            assert self.sim_header in ["meanP", "seqLSTM", "seqTransf", "MUSE"]
            retrieve_logits = self._loose_similarity(sequence_output, seq_features, visual_output, attention_mask, video_mask, sim_header=self.sim_header)
        else:
            assert self.sim_header in ["tightTransf"]
            retrieve_logits = self._cross_similarity(sequence_output, visual_output, attention_mask, video_mask, )

        return retrieve_logits, contrastive_direction
