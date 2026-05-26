import torch
import torch.nn.functional as nnf
import abc

from .text_utils import get_word_inds, get_time_words_attention_alpha
from . import seq_aligner

MAX_NUM_WORDS = 77
# 64->32, 32->16
LATENT_SIZE = (32, 32)
LOW_RESOURCE = False 

def register_attention_control_controlnet(model, controller):
    def ca_forward(self, place_in_unet):
        to_out = self.to_out
        if type(to_out) is torch.nn.modules.container.ModuleList:
            to_out = self.to_out[0]
        else:
            to_out = self.to_out
        
        def forward(hidden_states, encoder_hidden_states=None, attention_mask=None,temb=None,):
            is_cross = encoder_hidden_states is not None
            
            residual = hidden_states

            if self.spatial_norm is not None:
                hidden_states = self.spatial_norm(hidden_states, temb)

            input_ndim = hidden_states.ndim

            if input_ndim == 4:
                batch_size, channel, height, width = hidden_states.shape
                hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

            batch_size, sequence_length, _ = (
                hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
            )
            attention_mask = self.prepare_attention_mask(attention_mask, sequence_length, batch_size)

            if self.group_norm is not None:
                hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

            query = self.to_q(hidden_states)

            if encoder_hidden_states is None:
                encoder_hidden_states = hidden_states
            elif self.norm_cross:
                encoder_hidden_states = self.norm_encoder_hidden_states(encoder_hidden_states)

            key = self.to_k(encoder_hidden_states)
            value = self.to_v(encoder_hidden_states)

            query = self.head_to_batch_dim(query)
            key = self.head_to_batch_dim(key)
            value = self.head_to_batch_dim(value)

            attention_probs = self.get_attention_scores(query, key, attention_mask)
            attention_probs = controller(attention_probs, is_cross, place_in_unet)

            hidden_states = torch.bmm(attention_probs, value)
            hidden_states = self.batch_to_head_dim(hidden_states)

            # linear proj
            hidden_states = to_out(hidden_states)

            if input_ndim == 4:
                hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

            if self.residual_connection:
                hidden_states = hidden_states + residual

            hidden_states = hidden_states / self.rescale_output_factor

            return hidden_states
        return forward

    class DummyController:

        def __call__(self, *args):
            return args[0]

        def __init__(self):
            self.num_att_layers = 0

    if controller is None:
        controller = DummyController()

    def register_recr(net_, count, place_in_unet):
        
        if net_.__class__.__name__ == 'Attention':
        ## new diffusers use Attention class
        #if net_.__class__.__name__ == 'CrossAttention':
            net_.forward = ca_forward(net_, place_in_unet)
            return count + 1
        elif hasattr(net_, 'children'):
            for net__ in net_.children():
                count = register_recr(net__, count, place_in_unet)
        return count

    cross_att_count = 0
    sub_nets = model.controlnet.named_children()
    # for controlnet append the down and mid blocks
    for net in sub_nets:
        if "down" in net[0]:
            cross_att_count += register_recr(net[1], 0, "down")
        elif "mid" in net[0]:
            cross_att_count += register_recr(net[1], 0, "mid")
    sub_nets = model.unet.named_children()
    # for unet append the up blocks here
    for net in sub_nets:
        # if "up" in net[0]:
        #     cross_att_count += register_recr(net[1], 0, "up")
        if "down" in net[0]:
            cross_att_count += register_recr(net[1], 0, "down")
        elif "up" in net[0]:
            cross_att_count += register_recr(net[1], 0, "up")
        elif "mid" in net[0]:
            cross_att_count += register_recr(net[1], 0, "mid")

    controller.num_att_layers = cross_att_count


def get_equalizer(text, word_select, values, tokenizer=None):
    if type(word_select) is int or type(word_select) is str:
        word_select = (word_select,)
    equalizer = torch.ones(1, 77)
    
    for word, val in zip(word_select, values):
        inds = get_word_inds(text, word, tokenizer)
        equalizer[:, inds] = val
    return equalizer


class LocalBlend:

    def get_mask(self, maps, alpha, use_pool):
        k = 1
        maps = (maps * alpha).sum(-1).mean(1)
        if use_pool:
            maps = nnf.max_pool2d(maps, (k * 2 + 1, k * 2 +1), (1, 1), padding=(k, k))
        mask = nnf.interpolate(maps, size=LATENT_SIZE)
        mask = mask / mask.max(2, keepdims=True)[0].max(3, keepdims=True)[0]
        mask = mask.gt(self.th[1-int(use_pool)])

        B = maps.shape[0]//2
        mask[B:] = mask[B:] + mask[:B]
        #mask = mask[:B]+mask
        #mask = mask[:1] + mask
        return mask
        # select chosen words and aggregate across layers and heads
        # (maps * alpha) sums over last dim (words), then mean over layers:
        # -> [B, 1, 16, 16]
        # maps = (maps * alpha).sum(-1).mean(1)

        # if use_pool:
        #     k = 1
        #     maps = nnf.max_pool2d(maps, (k * 2 + 1, k * 2 + 1), (1, 1), padding=(k, k))

        # # upsample to latent size and normalize per-sample
        # mask = nnf.interpolate(maps, size=LATENT_SIZE, mode="bilinear", align_corners=False)
        # eps = 1e-6
        # max_hw = mask.amax(dim=(2, 3), keepdim=True).clamp_min(eps)
        # mask = mask / max_hw

        # # threshold; use higher threshold when pooling off (index 1), else index 0
        # thr = self.th[1 - int(use_pool)]
        # mask = (mask > thr).float()  # [B, 1, H, W]
        # return mask
    
    def __call__(self, x_t, attention_store):
        self.counter += 1
        if self.counter > self.start_blend:
            #original LDM 
            #maps = attention_store["down_cross"][:2] + attention_store["up_cross"][3:6]
            maps = attention_store["down_cross"][2:4] + attention_store["up_cross"][:3]
            maps = [item.reshape(self.alpha_layers.shape[0], -1, 1, 16, 16, MAX_NUM_WORDS) for item in maps]
            maps = torch.cat(maps, dim=1)
            mask = self.get_mask(maps, self.alpha_layers, True)
            if self.substruct_layers is not None:
                maps_sub = ~self.get_mask(maps, self.substruct_layers, False)
                mask = mask * maps_sub
            mask = mask.float()
            B = self.alpha_layers.shape[0]//2

            #x_t = x_t[:B] + mask * (x_t - x_t[:B])
            base = x_t[:B]        # [B, C, H, W]
            rep  = x_t[B:]     # [B, C, H, W]
            # --- Normalize mask to shape [B, 1 or C, H, W] ---
            if mask.dim() == 0:  # scalar
                mask_rep = mask.view(1, 1, 1, 1).expand(B, 1, base.shape[-2], base.shape[-1])
            elif mask.shape[0] == 2 * B:            # given per-sample for all 2B -> use the second half
                mask_rep = mask[B:]
            elif mask.shape[0] == B:                # already per-sample for the rep half
                mask_rep = mask
            elif mask.shape[0] == 1:                # broadcast a single mask to all rep samples
                mask_rep = mask.expand(B, *mask.shape[1:])
            else:
                raise ValueError(f"Unexpected mask.shape[0]={mask.shape[0]} for batch 2B={2*B}")

            # If mask channel is 1 but x_t has C>1, broadcast over channels
            if mask_rep.shape[1] == 1 and base.shape[1] > 1:
                mask_rep = mask_rep.expand(B, base.shape[1], base.shape[-2], base.shape[-1])

            # --- Blend only the second half, keep base unchanged ---
            rep_blended = base + mask_rep * (rep - base)   # [B, C, H, W]
            x_t = torch.cat([base, rep_blended], dim=0)  
        return x_t

    def __init__(self,update_token_index , substruct_token_index=None, start_blend=0.2, th=(.3, .3),
                 tokenizer=None, device="cuda",num_ddim_steps=50):
        alpha_layers = torch.zeros(len(update_token_index),  1, 1, 1, 1, MAX_NUM_WORDS)
        for i, token_indexes in enumerate(update_token_index):
            for ind in token_indexes:    
                alpha_layers[i, :, :, :, :, ind] = 1
        alpha_layers = alpha_layers.repeat(2, 1, 1, 1, 1, 1)
        if substruct_token_index is not None:
            substruct_layers = torch.zeros(len(substruct_token_index),  1, 1, 1, 1, MAX_NUM_WORDS)
            for i, token_indexes in enumerate(substruct_token_index):
                for ind in token_indexes:    
                    substruct_layers[i, :, :, :, :, ind] = 1
            substruct_layers = substruct_layers.repeat(2, 1, 1, 1, 1, 1)
            self.substruct_layers = substruct_layers.to(device)
        else:
            self.substruct_layers = None
        self.alpha_layers = alpha_layers.to(device)
        self.start_blend = int(start_blend * num_ddim_steps)
        self.counter = 0 
        self.th=th
# import torch
# import torch.nn.functional as F

# class LocalBlend:
#     def __init__(self, update_token_index, substruct_words=None, start_blend=0.2,
#                  th=(.3, .3), tokenizer=None, device="cuda", num_ddim_steps=50):
#         B = len(update_token_index)
#         self.device = device
#         self.th = th
#         self.start_blend = int(start_blend * num_ddim_steps)
#         self.counter = 0

#         # alpha_layers: [2B, 1, 1, 1, 1, MAX_NUM_WORDS]
#         alpha_layers = torch.zeros(B, 1, 1, 1, 1, MAX_NUM_WORDS)
#         for i, token_indexes in enumerate(update_token_index):
#             for ind in token_indexes:
#                 alpha_layers[i, :, :, :, :, ind] = 1
#         alpha_layers = alpha_layers.repeat(2, 1, 1, 1, 1, 1).to(device)
#         self.alpha_layers = alpha_layers

#         # Optional substructure layers (build only if you pass prompts+words explicitly)
#         self.substruct_layers = None
#         if substruct_words is not None:
#             # You likely want per-sample word indices. Ensure you pass `prompts`.
#             raise ValueError("substruct_words path requires prompts; pass prompts and implement get_word_inds.")

#     def get_mask(self, maps, alpha, use_pool):
#         """
#         maps: list/stacked tensor of attention maps shaped like
#               [2B, L, 1, 16, 16, MAX_NUM_WORDS] after your reshape+cat
#         alpha: same last-dim alignment as maps (..., MAX_NUM_WORDS)
#         returns: boolean mask [B, 1, H_lat, W_lat]
#         """
#         # Aggregate over words then layers: -> [2B, 1, 16, 16]
#         attn = (maps * alpha).sum(dim=-1).mean(dim=1)  # float

#         if use_pool:
#             k = 1
#             attn = F.max_pool2d(attn, kernel_size=(2*k+1, 2*k+1), stride=1, padding=k)

#         # Upsample to latent size
#         mask = F.interpolate(attn, size=LATENT_SIZE, mode="bilinear", align_corners=False)

#         # Per-sample normalization
#         max_hw = mask.amax(dim=(2, 3), keepdim=True).clamp_min(1e-6)
#         mask = mask / max_hw  # [2B, 1, H, W]

#         # Threshold selection: th[0] if pooled else th[1]
#         thr = self.th[1 - int(use_pool)]
#         mask = (mask > thr)  # boolean, [2B, 1, H, W]

#         # Union the pairwise masks: first half vs second half
#         B2 = mask.shape[0]
#         assert B2 % 2 == 0, "Batch must be even (pairs)."
#         B = B2 // 2
#         mask_pair = mask[:B] | mask[B:]  # [B, 1, H, W]
#         return mask_pair

#     def __call__(self, x_t, attention_store):
#         """
#         x_t: [2B, C, H, W] where first B are bases, next B are edits
#         attention_store: dict with lists of cross-attn tensors per block
#         """
#         self.counter += 1
#         if self.counter <= self.start_blend:
#             return x_t

#         # Collect cross-attn maps (choose a stable subset of blocks)
#         # Example: use up to 5 blocks total combining down/up (tweak to taste)
#         maps_list = attention_store["down_cross"][2:4] + attention_store["up_cross"][:3]
#         # Reshape each to [2B, L, 1, 16, 16, MAX_NUM_WORDS] and concat on layer dim
#         maps_list = [m.reshape(self.alpha_layers.shape[0], -1, 1, 16, 16, MAX_NUM_WORDS) for m in maps_list]
#         maps = torch.cat(maps_list, dim=1)

#         # Primary mask
#         mask = self.get_mask(maps, self.alpha_layers, use_pool=True)  # [B,1,H,W]
#         if self.substruct_layers is not None:
#             # Exclude substructure region if provided
#             maps_sub = ~self.get_mask(maps, self.substruct_layers, use_pool=False)
#             mask = mask & maps_sub

#         # Blend only the second half with the first half as base
#         B2, C, H, W = x_t.shape
#         assert B2 % 2 == 0, "x_t must have even batch for pairing."
#         B = B2 // 2
#         base = x_t[:B]      # [B, C, H, W]
#         rep  = x_t[B:]      # [B, C, H, W]

#         # Broadcast mask to channels
#         mask_bc = mask.float().expand(-1, C, -1, -1)  # [B, C, H, W]

#         rep_blended = base + mask_bc * (rep - base)
#         x_t = torch.cat([base, rep_blended], dim=0)
#         return x_t

        
class EmptyControl:

    def step_callback(self, x_t):
        return x_t
    
    def between_steps(self):
        return
    
    def __call__(self, attn, is_cross, place_in_unet):
        return attn

    
class AttentionControl(abc.ABC):
    
    def step_callback(self, x_t):
        return x_t
    
    def between_steps(self):
        return
    
    @property
    def num_uncond_att_layers(self):
        return self.num_att_layers if LOW_RESOURCE else 0
    
    @abc.abstractmethod
    def forward (self, attn, is_cross, place_in_unet):
        raise NotImplementedError

    def __call__(self, attn, is_cross, place_in_unet):
        if self.cur_att_layer >= self.num_uncond_att_layers:
            if LOW_RESOURCE:
                attn = self.forward(attn, is_cross, place_in_unet)
            else:
                h = attn.shape[0]
                attn[h // 2:] = self.forward(attn[h // 2:], is_cross, place_in_unet)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()
        return attn
        
    def remove(self):
        # remove hooks
        for h in getattr(self, "handles", []):
            try: h.remove()
            except Exception: pass
        self.handles = []

        # clear any stored attention (if subclass added it)
        if hasattr(self, "step_store"): self.step_store = AttentionStore.get_empty_store()
        if hasattr(self, "attention_store"): self.attention_store = {}

        # break potential reference cycles
        if hasattr(self, "prev_controller"): self.prev_controller = None
        if hasattr(self, "local_blend"): self.local_blend = None
    
    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0

class SpatialReplace(EmptyControl):

    def step_callback(self, x_t):
        if self.cur_step < self.stop_inject:
            b = x_t.shape[0]
            x_t = x_t[:1].expand(b, *x_t.shape[1:])
        return x_t

    def __init__(self, stop_inject,num_ddim_steps=50):
        super(SpatialReplace, self).__init__()
        self.stop_inject = int((1 - stop_inject) * num_ddim_steps)
        

class AttentionStore(AttentionControl):

    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [],  "mid_self": [],  "up_self": []}

    def forward(self, attn, is_cross, place_in_unet):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        #if attn.shape[1] <= 16 ** 2:  # avoid memory overhead
        if attn.shape[1] <= 32 ** 2:  # avoid memory overhead
            self.step_store[key].append(attn)
        return attn

    def between_steps(self):
        if len(self.attention_store) == 0:
            self.attention_store = self.step_store
        else:
            for key in self.attention_store:
                for i in range(len(self.attention_store[key])):
                    self.attention_store[key][i] += self.step_store[key][i]
        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        average_attention = {key: [item / self.cur_step for item in self.attention_store[key]] for key in self.attention_store}
        return average_attention

    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    def __init__(self):
        super(AttentionStore, self).__init__()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

        
class AttentionControlEdit(AttentionStore, abc.ABC):
    
    def step_callback(self, x_t):
        if self.local_blend is not None:
            x_t = self.local_blend(x_t, self.attention_store)
        return x_t
        
    def replace_self_attention(self, attn_base, att_replace, place_in_unet):
        if att_replace.shape[2] <= 16 ** 2:
        #if att_replace.shape[2] <= 32 ** 2:
            #attn_base = attn_base.unsqueeze(0).expand(att_replace.shape[0], *attn_base.shape)
            #return attn_base.expand(att_replace.shape[0], *attn_base.shape)
            return attn_base
        else:
            return att_replace
    
    @abc.abstractmethod
    def replace_cross_attention(self, attn_base, att_replace):
        raise NotImplementedError
    
    def forward(self, attn, is_cross, place_in_unet):
        super(AttentionControlEdit, self).forward(attn, is_cross, place_in_unet)
        if is_cross or (self.num_self_replace[0] <= self.cur_step < self.num_self_replace[1]):
            # h = attn.shape[0] // (self.batch_size)
            # attn = attn.reshape(self.batch_size, h, *attn.shape[1:])
            # attn_base, attn_repalce = attn[0], attn[1:]
            # if is_cross:
            #     alpha_words = self.cross_replace_alpha[self.cur_step]
            #     attn_repalce_new = self.replace_cross_attention(attn_base, attn_repalce) * alpha_words + (1 - alpha_words) * attn_repalce
            #     attn[1:] = attn_repalce_new
            # else:
            #     attn[1:] = self.replace_self_attention(attn_base, attn_repalce, place_in_unet)
            # attn = attn.reshape(self.batch_size * h, *attn.shape[2:])
            h = attn.shape[0] // (self.batch_size*2)
            attn = attn.reshape(self.batch_size*2, h, *attn.shape[1:])
            attn_base, attn_repalce = attn[:self.batch_size], attn[self.batch_size:]
            if is_cross:
                alpha_words = self.cross_replace_alpha[self.cur_step]
                attn_repalce_new = self.replace_cross_attention(attn_base, attn_repalce) * alpha_words + (1 - alpha_words) * attn_repalce
                attn[self.batch_size:] = attn_repalce_new
            else:
                attn[self.batch_size:] = self.replace_self_attention(attn_base, attn_repalce, place_in_unet)
            attn = attn.reshape(self.batch_size * h*2, *attn.shape[2:])
        return attn
    
    def __init__(self, 
                 update_token_index, 
                 num_steps,
                 cross_replace_steps,
                 self_replace_steps,
                 local_blend, 
                 tokenizer=None,
                 device="cuda"):
        super(AttentionControlEdit, self).__init__()
        self.batch_size = len(update_token_index)
        self.cross_replace_alpha = get_time_words_attention_alpha(update_token_index, num_steps, cross_replace_steps, tokenizer).to(device)
        if type(self_replace_steps) is float:
            self_replace_steps = 0, self_replace_steps
        self.num_self_replace = int(num_steps * self_replace_steps[0]), int(num_steps * self_replace_steps[1])
        self.local_blend = local_blend


class AttentionReplace(AttentionControlEdit):

    def replace_cross_attention(self, attn_base, att_replace):
        # einsum over (H,Q,K) × (K,N) per batch → (B,H,Q,N)
        return torch.einsum('bhqk,bkn->bhqn', attn_base, self.mapper)
        #return torch.einsum('hpw,bwn->bhpn', attn_base, self.mapper)
      
    def __init__(self, update_token_index, num_steps, cross_replace_steps, self_replace_steps,
                 local_blend = None, tokenizer=None,device="cuda"):
        super(AttentionReplace, self).__init__(update_token_index=update_token_index, 
                                              num_steps=num_steps, 
                                              cross_replace_steps=cross_replace_steps, 
                                              self_replace_steps=self_replace_steps, 
                                              local_blend=local_blend,
                                              device=device)
        self.mapper = seq_aligner.get_replacement_mapper(update_token_index, tokenizer).to(device)


class AttentionRefine(AttentionControlEdit):

    def replace_cross_attention(self, attn_base, att_replace):
        attn_base_replace = attn_base[:, :, self.mapper].permute(2, 0, 1, 3)
        attn_replace = attn_base_replace * self.alphas + att_replace * (1 - self.alphas)
        # attn_replace = attn_replace / attn_replace.sum(-1, keepdims=True)
        return attn_replace

    def __init__(self, prompts, num_steps, cross_replace_steps, self_replace_steps,
                 local_blend = None, tokenizer=None,device="cuda"):
        super(AttentionRefine, self).__init__(prompts=prompts, 
                                              num_steps=num_steps, 
                                              cross_replace_steps=cross_replace_steps, 
                                              self_replace_steps=self_replace_steps, 
                                              local_blend=local_blend,
                                              device=device)
        self.mapper, alphas = seq_aligner.get_refinement_mapper(prompts, tokenizer)
        self.mapper, alphas = self.mapper.to(device), alphas.to(device)
        self.alphas = alphas.reshape(alphas.shape[0], 1, 1, alphas.shape[1])


class AttentionReweight(AttentionControlEdit):

    def replace_cross_attention(self, attn_base, att_replace):
        if self.prev_controller is not None:
            attn_base = self.prev_controller.replace_cross_attention(attn_base, att_replace)
        attn_replace = attn_base[None, :, :, :] * self.equalizer[:, None, None, :]
        # attn_replace = attn_replace / attn_replace.sum(-1, keepdims=True)
        return attn_replace

    def __init__(self, 
                 prompts, 
                 num_steps, 
                 cross_replace_steps, 
                 self_replace_steps, 
                 equalizer,
                 local_blend = None, 
                 controller = None,
                 device="cuda"):
        super(AttentionReweight, self).__init__(prompts=prompts, 
                                                num_steps=num_steps, 
                                                cross_replace_steps=cross_replace_steps, 
                                                self_replace_steps=self_replace_steps, 
                                                local_blend=local_blend,
                                                device=device)
        self.equalizer = equalizer.to(device)
        self.prev_controller = controller


def make_controller(pipeline, 
                    update_token_index, 
                    is_replace_controller, 
                    cross_replace_steps, 
                    self_replace_steps, 
                    blend_words=None, 
                    equilizer_params=None, 
                    num_ddim_steps=50,
                    blend_params ={'start_blend':0.2,'th':(0.3,0.3)},
                    substruct_token_index=None,
                    device="cuda") -> AttentionControlEdit:
    # if blend_words is None:
    #     lb = None
    # else:
    if blend_words:
        lb = LocalBlend(update_token_index,substruct_token_index=substruct_token_index,start_blend =blend_params['start_blend'] ,th = blend_params['th'],
                    tokenizer=pipeline.tokenizer, device=device,num_ddim_steps=num_ddim_steps)
    else:
        lb=None
    if is_replace_controller:
        controller = AttentionReplace(update_token_index, 
                                      num_ddim_steps, 
                                      cross_replace_steps=cross_replace_steps, 
                                      self_replace_steps=self_replace_steps, 
                                      local_blend=lb,
                                      tokenizer=pipeline.tokenizer)
    else:
        controller = AttentionRefine(prompts, 
                                     num_ddim_steps, 
                                     cross_replace_steps=cross_replace_steps, 
                                     self_replace_steps=self_replace_steps, 
                                     local_blend=lb,
                                     tokenizer=pipeline.tokenizer)
    # if equilizer_params is not None:
    #     eq = get_equalizer(prompts[1], 
    #                        equilizer_params["words"], 
    #                        equilizer_params["values"], 
    #                        tokenizer=pipeline.tokenizer)
    #     controller = AttentionReweight(prompts, 
    #                                    num_ddim_steps,
    #                                    cross_replace_steps=cross_replace_steps,
    #                                    self_replace_steps=self_replace_steps, 
    #                                    equalizer=eq, 
    #                                    local_blend=lb, 
    #                                    controller=controller)
    return controller
