import torch
import numpy as np
from PIL import Image
import torch.nn.functional as nnf
from torch.optim.adam import Adam

from .attention_control import register_attention_control_controlnet

#from utils.utils import slerp_tensor, image2latent, latent2image

# class NegativePromptInversion:
    
#     def prev_step(self, model_output, timestep, sample):
#         prev_timestep = timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
#         alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
#         alpha_prod_t_prev = self.scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.scheduler.final_alpha_cumprod
#         beta_prod_t = 1 - alpha_prod_t
#         pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
#         pred_sample_direction = (1 - alpha_prod_t_prev) ** 0.5 * model_output
#         prev_sample = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction
#         return prev_sample
    
#     def next_step(self, model_output, timestep, sample):
#         timestep, next_timestep = min(timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps, 999), timestep
#         alpha_prod_t = self.scheduler.alphas_cumprod[timestep] if timestep >= 0 else self.scheduler.final_alpha_cumprod
#         alpha_prod_t_next = self.scheduler.alphas_cumprod[next_timestep]
#         beta_prod_t = 1 - alpha_prod_t
#         next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
#         next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
#         next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
#         return next_sample
    
#     def get_noise_pred_single(self, latents, t, context):
#         noise_pred = self.model.unet(latents, t, encoder_hidden_states=context)["sample"]
#         return noise_pred

#     @torch.no_grad()
#     def init_prompt(self, prompt):
#         uncond_input = self.model.tokenizer(
#             [""], padding="max_length", max_length=self.model.tokenizer.model_max_length,
#             return_tensors="pt"
#         )
#         uncond_embeddings = self.model.text_encoder(uncond_input.input_ids.to(self.model.device))[0]
#         text_input = self.model.tokenizer(
#             [prompt],
#             padding="max_length",
#             max_length=self.model.tokenizer.model_max_length,
#             truncation=True,
#             return_tensors="pt",
#         )
#         text_embeddings = self.model.text_encoder(text_input.input_ids.to(self.model.device))[0]
#         self.context = torch.cat([uncond_embeddings, text_embeddings])
#         self.prompt = prompt

#     @torch.no_grad()
#     def ddim_loop(self, latent):
#         uncond_embeddings, cond_embeddings = self.context.chunk(2)
#         all_latent = [latent]
#         latent = latent.clone().detach()
#         print("DDIM Inversion ...")
#         for i in range(self.num_ddim_steps):
#             t = self.model.scheduler.timesteps[len(self.model.scheduler.timesteps) - i - 1]
#             noise_pred = self.get_noise_pred_single(latent, t, cond_embeddings)
#             latent = self.next_step(noise_pred, t, latent)
#             all_latent.append(latent)
#         return all_latent

#     @property
#     def scheduler(self):
#         return self.model.scheduler

#     @torch.no_grad()
#     def ddim_inversion(self, image):
#         latent = image2latent(self.model.vae, image)
#         image_rec = latent2image(self.model.vae, latent)[0]
#         ddim_latents = self.ddim_loop(latent)
#         return image_rec, ddim_latents, latent

#     def invert(self, image_gt, prompt, npi_interp=0.0):
#         """
#         Get DDIM Inversion of the image
        
#         Parameters:
#         image_gt - the gt image with a size of [512,512,3], the channel follows the rgb of PIL.Image. i.e. RGB.
#         prompt - this is the prompt used for DDIM Inversion
#         npi_interp - the interpolation ratio among conditional embedding and unconditional embedding
#         num_ddim_steps - the number of ddim steps
        
#         Returns:
#             image_rec - the image reconstructed by VAE decoder with a size of [512,512,3], the channel follows the rgb of PIL.Image. i.e. RGB.
#             image_rec_latent - the image latent with a size of [64,64,4]
#             ddim_latents - the ddim inversion latents 50*[64,4,4], the first latent is the image_rec_latent, the last latent is noise (but in fact not pure noise)
#             uncond_embeddings - the fake uncond_embeddings, in fact is cond_embedding or a interpolation among cond_embedding and uncond_embedding
#         """
#         self.init_prompt(prompt)
#         register_attention_control(self.model, None)
#         image_rec, ddim_latents, image_rec_latent = self.ddim_inversion(image_gt)
#         uncond_embeddings, cond_embeddings = self.context.chunk(2)
#         if npi_interp > 0.0: # do vector interpolation among cond_embedding and uncond_embedding
#             cond_embeddings = slerp_tensor(npi_interp, cond_embeddings, uncond_embeddings)
#         uncond_embeddings = [cond_embeddings] * self.num_ddim_steps
#         return image_rec, image_rec_latent, ddim_latents, uncond_embeddings

#     def __init__(self, model,num_ddim_steps):
#         self.model = model
#         self.tokenizer = self.model.tokenizer
#         self.prompt = None
#         self.context = None
#         self.num_ddim_steps=num_ddim_steps




# class NullInversion:
    
#     def prev_step(self, model_output, timestep: int, sample):
#         prev_timestep = timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
#         alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
#         alpha_prod_t_prev = self.scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.scheduler.final_alpha_cumprod
#         beta_prod_t = 1 - alpha_prod_t
#         pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
#         pred_sample_direction = (1 - alpha_prod_t_prev) ** 0.5 * model_output
#         prev_sample = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction
#         return prev_sample
    
#     def next_step(self, model_output, timestep: int, sample):
#         timestep, next_timestep = min(timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps, 999), timestep
#         alpha_prod_t = self.scheduler.alphas_cumprod[timestep] if timestep >= 0 else self.scheduler.final_alpha_cumprod
#         alpha_prod_t_next = self.scheduler.alphas_cumprod[next_timestep]
#         beta_prod_t = 1 - alpha_prod_t
#         next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
#         next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
#         next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
#         return next_sample
    
#     def get_noise_pred_single(self, latents, t, context):
#         noise_pred = self.model.unet(latents, t, encoder_hidden_states=context)["sample"]
#         return noise_pred

#     def get_noise_pred(self, latents, t, guidance_scale, is_forward=True, context=None):
#         latents_input = torch.cat([latents] * 2)
#         if context is None:
#             context = self.context
#         guidance_scale = 1 if is_forward else guidance_scale
#         noise_pred = self.model.unet(latents_input, t, encoder_hidden_states=context)["sample"]
#         noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
#         noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
#         if is_forward:
#             latents = self.next_step(noise_pred, t, latents)
#         else:
#             latents = self.prev_step(noise_pred, t, latents)
#         return latents

#     @torch.no_grad()
#     def init_prompt(self, prompt: str):
#         uncond_input = self.model.tokenizer(
#             [""], 
#             padding="max_length", 
#             max_length=self.model.tokenizer.model_max_length,
#             return_tensors="pt"
#         )
#         uncond_embeddings = self.model.text_encoder(uncond_input.input_ids.to(self.model.device))[0]
#         text_input = self.model.tokenizer(
#             [prompt],
#             padding="max_length",
#             max_length=self.model.tokenizer.model_max_length,
#             truncation=True,
#             return_tensors="pt",
#         )
#         text_embeddings = self.model.text_encoder(text_input.input_ids.to(self.model.device))[0]
#         self.context = torch.cat([uncond_embeddings, text_embeddings])
#         self.prompt = prompt

#     @torch.no_grad()
#     def ddim_loop(self, latent):
#         uncond_embeddings, cond_embeddings = self.context.chunk(2)
#         all_latent = [latent]
#         latent = latent.clone().detach()
#         for i in range(self.num_ddim_steps):
#             t = self.model.scheduler.timesteps[len(self.model.scheduler.timesteps) - i - 1]
#             noise_pred = self.get_noise_pred_single(latent, t, cond_embeddings)
#             latent = self.next_step(noise_pred, t, latent)
#             all_latent.append(latent)
#         return all_latent

#     @property
#     def scheduler(self):
#         return self.model.scheduler

#     @torch.no_grad()
#     def ddim_inversion(self, image):
#         latent = image2latent(self.model.vae, image)
#         image_rec = latent2image(self.model.vae, latent)[0]
#         ddim_latents = self.ddim_loop(latent)
#         return image_rec, ddim_latents

#     def null_optimization(self, latents, num_inner_steps, epsilon, guidance_scale):
#         uncond_embeddings, cond_embeddings = self.context.chunk(2)
#         uncond_embeddings_list = []
#         latent_cur = latents[-1]
#         for i in range(self.num_ddim_steps):
#             uncond_embeddings = uncond_embeddings.clone().detach()
#             t = self.model.scheduler.timesteps[i]
#             if num_inner_steps!=0:
#                 uncond_embeddings.requires_grad = True
#                 optimizer = Adam([uncond_embeddings], lr=1e-2 * (1. - i / 100.))
#                 latent_prev = latents[len(latents) - i - 2]
#                 with torch.no_grad():
#                     noise_pred_cond = self.get_noise_pred_single(latent_cur, t, cond_embeddings)
#                 for j in range(num_inner_steps):
#                     noise_pred_uncond = self.get_noise_pred_single(latent_cur, t, uncond_embeddings)
#                     noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
#                     latents_prev_rec = self.prev_step(noise_pred, t, latent_cur)
#                     loss = nnf.mse_loss(latents_prev_rec, latent_prev)
#                     optimizer.zero_grad()
#                     loss.backward()
#                     optimizer.step()
#                     loss_item = loss.item()
#                     if loss_item < epsilon + i * 2e-5:
#                         break
                
#             uncond_embeddings_list.append(uncond_embeddings[:1].detach())
#             with torch.no_grad():
#                 context = torch.cat([uncond_embeddings, cond_embeddings])
#                 latent_cur = self.get_noise_pred(latent_cur, t, guidance_scale, False, context)
#         return uncond_embeddings_list
    
#     def invert(self, image_gt, prompt, guidance_scale, num_inner_steps=10, early_stop_epsilon=1e-5):
#         self.init_prompt(prompt)
#         register_attention_control(self.model, None)
        
#         image_rec, ddim_latents = self.ddim_inversion(image_gt)
        
#         uncond_embeddings = self.null_optimization(ddim_latents, num_inner_steps, early_stop_epsilon,guidance_scale)
#         return image_gt, image_rec, ddim_latents, uncond_embeddings
    
#     def __init__(self, model,num_ddim_steps):
#         self.model = model
#         self.tokenizer = self.model.tokenizer
#         self.prompt = None
#         self.context = None
#         self.num_ddim_steps=num_ddim_steps

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import numpy as np
import PIL.Image as Image
import torch

def slerp(val, low, high):
    """ 
    taken from https://discuss.pytorch.org/t/help-regarding-slerp-function-for-generative-model-sampling/32475/4
    """
    low_norm = low/torch.norm(low, dim=1, keepdim=True)
    high_norm = high/torch.norm(high, dim=1, keepdim=True)
    omega = torch.acos((low_norm*high_norm).sum(1))
    so = torch.sin(omega)
    res = (torch.sin((1.0-val)*omega)/so).unsqueeze(1)*low + (torch.sin(val*omega)/so).unsqueeze(1) * high
    return res


def slerp_tensor(val, low, high):
    """ 
    used in negtive prompt inversion
    """
    shape = low.shape
    res = slerp(val, low.flatten(1), high.flatten(1))
    return res.reshape(shape)

def load_512(image_path, left=0, right=0, top=0, bottom=0):
    if type(image_path) is str:
        image = np.array(Image.open(image_path))[:, :, :3]
    else:
        image = image_path
    h, w, c = image.shape
    left = min(left, w-1)
    right = min(right, w - left - 1)
    top = min(top, h - left - 1)
    bottom = min(bottom, h - top - 1)
    image = image[top:h-bottom, left:w-right]
    h, w, c = image.shape
    if h < w:
        offset = (w - h) // 2
        image = image[:, offset:offset + h]
    elif w < h:
        offset = (h - w) // 2
        image = image[offset:offset + w]
    image = np.array(Image.fromarray(image).resize((512, 512)))
    return image

def init_latent(latent, model, height, width, generator, batch_size):
    if latent is None:
        latent = torch.randn(
            (1, model.unet.in_channels, height // 8, width // 8),
            generator=generator,
        )
    latents = latent.expand(batch_size,  model.unet.in_channels, height // 8, width // 8).to(model.device)
    return latent, latents


@torch.no_grad()
def latent2image(model, latents, return_type='np'):
    latents = 1 / 0.18215 * latents.detach()
    image = model.decode(latents)['sample']
    if return_type == 'np':
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        image = (image * 255).astype(np.uint8)
    return image

@torch.no_grad()
def image2latent(model, image):
    with torch.no_grad():
        if type(image) is Image:
            image = np.array(image)
        if type(image) is torch.Tensor and image.dim() == 4:
            latents = image
        else:
            image = torch.from_numpy(image).float() / 127.5 - 1
            image = image.permute(2, 0, 1).unsqueeze(0).to(model.device)
            latents = model.encode(image)['latent_dist'].mean
            latents = latents * 0.18215
    return latents



def get_word_inds(text: str, word_place: int, tokenizer):
    split_text = text.split(" ")
    if type(word_place) is str:
        word_place = [i for i, word in enumerate(split_text) if word_place == word]
    elif type(word_place) is int:
        word_place = [word_place]
    out = []
    if len(word_place) > 0:
        words_encode = [tokenizer.decode([item]).strip("#") for item in tokenizer.encode(text)][1:-1]
        cur_len, ptr = 0, 0

        for i in range(len(words_encode)):
            cur_len += len(words_encode[i])
            if ptr in word_place:
                out.append(i + 1)
            if cur_len >= len(split_text[ptr]):
                ptr += 1
                cur_len = 0
    return np.array(out)

def update_alpha_time_word(alpha, bounds, prompt_ind,
                           word_inds=None):
    if type(bounds) is float:
        bounds = 0, bounds
    start, end = int(bounds[0] * alpha.shape[0]), int(bounds[1] * alpha.shape[0])
    if word_inds is None:
        word_inds = torch.arange(alpha.shape[2])
    alpha[: start, prompt_ind, word_inds] = 0
    alpha[start: end, prompt_ind, word_inds] = 1
    alpha[end:, prompt_ind, word_inds] = 0
    return alpha


def get_time_words_attention_alpha(prompts, num_steps,
                                   cross_replace_steps,
                                   tokenizer, max_num_words=77):
    if type(cross_replace_steps) is not dict:
        cross_replace_steps = {"default_": cross_replace_steps}
    if "default_" not in cross_replace_steps:
        cross_replace_steps["default_"] = (0., 1.)
    alpha_time_words = torch.zeros(num_steps + 1, len(prompts) - 1, max_num_words)
    for i in range(len(prompts) - 1):
        alpha_time_words = update_alpha_time_word(alpha_time_words, cross_replace_steps["default_"],
                                                  i)
    for key, item in cross_replace_steps.items():
        if key != "default_":
             inds = [get_word_inds(prompts[i], key, tokenizer) for i in range(1, len(prompts))]
             for i, ind in enumerate(inds):
                 if len(ind) > 0:
                    alpha_time_words = update_alpha_time_word(alpha_time_words, item, i, ind)
    alpha_time_words = alpha_time_words.reshape(num_steps + 1, len(prompts) - 1, 1, 1, max_num_words)
    return alpha_time_words

def txt_draw(text,
                target_size=[512,512]):
    plt.figure(dpi=300,figsize=(1,1))
    plt.text(-0.1, 1.1, text,fontsize=3.5, wrap=True,verticalalignment="top",horizontalalignment="left")
    plt.axis('off')
    
    canvas = FigureCanvasAgg(plt.gcf())
    canvas.draw()
    w, h = canvas.get_width_height()
    buf = np.fromstring(canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)
    buf = np.roll(buf, 3, axis=2)
    image = Image.frombytes("RGBA", (w, h), buf.tostring())
    image = image.resize(target_size,Image.ANTIALIAS)
    image = np.asarray(image)[:,:,:3]
    
    plt.close('all')
    
    return image


class DirectInversion:
    
    def prev_step(self, model_output, timestep: int, sample):
        prev_timestep = timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.scheduler.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        pred_sample_direction = (1 - alpha_prod_t_prev) ** 0.5 * model_output
        prev_sample = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction
        
        difference_scale_pred_original_sample= - beta_prod_t ** 0.5  / alpha_prod_t ** 0.5
        difference_scale_pred_sample_direction = (1 - alpha_prod_t_prev) ** 0.5 
        difference_scale = alpha_prod_t_prev ** 0.5 * difference_scale_pred_original_sample + difference_scale_pred_sample_direction
        
        return prev_sample,difference_scale
    
    def next_step(self, model_output, timestep: int, sample):
        timestep, next_timestep = min(timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps, 999), timestep
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep] if timestep >= 0 else self.scheduler.final_alpha_cumprod
        alpha_prod_t_next = self.scheduler.alphas_cumprod[next_timestep]
        beta_prod_t = 1 - alpha_prod_t
        next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
        next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
        return next_sample
    
    def get_noise_pred_single(self, latents, t, context):
        bs = len(context)//len(self.label)
        (down_block_res_samples, mid_block_res_sample),_,_,_ = self.pipe.controlnet(
                    latents,
                    t,
                    encoder_hidden_states=context,
                    controlnet_cond=None,
                    return_dict=False,
                    training=False,
                    label = self.label.repeat(bs,1,1),
                    sampling=True,
        )
        noise_pred = self.pipe.unet(latents, t, encoder_hidden_states=context,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                    return_dict=False)[0]
        return noise_pred

    def get_noise_pred(self, latents, t, guidance_scale, is_forward=True, context=None):
        latents_input = torch.cat([latents] * 2)
        if context is None:
            context = self.context
        guidance_scale = 1 if is_forward else guidance_scale
        noise_pred = self.pipe.unet(latents_input, t, encoder_hidden_states=context)["sample"]
        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
        if is_forward:
            latents = self.next_step(noise_pred, t, latents)
        else:
            latents = self.prev_step(noise_pred, t, latents)
        return latents

    @torch.no_grad()
    def init_prompt(self, prompt: str,negative_prompt="",batch_size=1):
        text_input = self.pipe.tokenizer(prompt, padding='max_length', max_length=self.pipe.tokenizer.model_max_length,
                                    truncation=True, return_tensors='pt')
        text_embeddings = self.pipe.text_encoder(text_input.input_ids.to(self.device))[0]

        # Do the same for unconditional embeddings
        uncond_input = self.pipe.tokenizer(negative_prompt, padding='max_length', max_length=self.pipe.tokenizer.model_max_length,
                                      return_tensors='pt')

        uncond_embeddings = self.pipe.text_encoder(uncond_input.input_ids.to(self.device))[0]

        # Cat for final embeddings
        text_embeddings = torch.cat([uncond_embeddings] * batch_size + [text_embeddings] * batch_size)
        self.context = text_embeddings
        self.prompt = prompt

    @torch.no_grad()
    def ddim_loop(self, latent):
        uncond_embeddings,all_cond_embeddings = self.context.chunk(2)
        source_cond_embeddings,target_cond_embeddings=all_cond_embeddings.chunk(2)
        cond_embeddings=source_cond_embeddings
        all_latent = [latent]
        latent = latent.clone().detach()
        for i in range(self.num_ddim_steps):
            t = self.pipe.scheduler.timesteps[len(self.pipe.scheduler.timesteps) - i - 1]
            noise_pred = self.get_noise_pred_single(latent, t, cond_embeddings)
            latent = self.next_step(noise_pred, t, latent)
            all_latent.append(latent)
        return torch.stack(all_latent, dim=0)
        #return all_latent
    
    @torch.no_grad()
    def ddim_null_loop(self, latent):
        uncond_embeddings, cond_embeddings = self.context.chunk(2)
        uncond_embeddings=uncond_embeddings[[0]]
        all_latent = [latent]
        latent = latent.clone().detach()
        for i in range(self.num_ddim_steps):
            t = self.pipe.scheduler.timesteps[len(self.pipe.scheduler.timesteps) - i - 1]
            noise_pred = self.get_noise_pred_single(latent, t, uncond_embeddings)
            latent = self.next_step(noise_pred, t, latent)
            all_latent.append(latent)
        return all_latent
    
    @torch.no_grad()
    def ddim_with_guidance_scale_loop(self, latent,guidance_scale):
        uncond_embeddings, cond_embeddings = self.context.chunk(2)
        uncond_embeddings=uncond_embeddings[[0]]
        cond_embeddings=cond_embeddings[[0]]
        all_latent = [latent]
        latent = latent.clone().detach()
        for i in range(self.num_ddim_steps):
            t = self.pipe.scheduler.timesteps[len(self.pipe.scheduler.timesteps) - i - 1]
            uncond_noise_pred = self.get_noise_pred_single(latent, t, uncond_embeddings)
            cond_noise_pred = self.get_noise_pred_single(latent, t, cond_embeddings)
            noise_pred = uncond_noise_pred + guidance_scale * (cond_noise_pred - uncond_noise_pred)
            latent = self.next_step(noise_pred, t, latent)
            all_latent.append(latent)
        return all_latent

    @property
    def scheduler(self):
        return self.pipe.scheduler

    @torch.no_grad()
    def ddim_inversion(self, img_latent):
        latent = img_latent
        image_rec = latent2image(self.pipe.vae, latent)[0]
        ddim_latents = self.ddim_loop(latent)
        return image_rec, ddim_latents
    
    @torch.no_grad()
    def ddim_null_inversion(self, image):
        latent = image2latent(self.pipe.vae, image)
        image_rec = latent2image(self.pipe.vae, latent)[0]
        ddim_latents = self.ddim_null_loop(latent)
        return image_rec, ddim_latents
    
    @torch.no_grad()
    def ddim_with_guidance_scale_inversion(self, image,guidance_scale):
        latent = image2latent(self.pipe.vae, image)
        image_rec = latent2image(self.pipe.vae, latent)[0]
        ddim_latents = self.ddim_with_guidance_scale_loop(latent,guidance_scale)
        return image_rec, ddim_latents

    def offset_calculate(self, latents, num_inner_steps, epsilon, guidance_scale):
        noise_loss_list = []
        # for both of the source and target
        latent_cur = torch.concat([latents[-1]]*2)
        #latent_cur = torch.concat([latents[-1]]*(self.context.shape[0]//2))
        for i in range(self.num_ddim_steps):            
            latent_prev = torch.concat([latents[len(latents) - i - 2]]*2)
            t = self.pipe.scheduler.timesteps[i]
            with torch.no_grad():
                noise_pred = self.get_noise_pred_single(torch.concat([latent_cur]*2), t, self.context)
                noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                noise_pred_w_guidance = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                latents_prev_rec, _ = self.prev_step(noise_pred_w_guidance, t, latent_cur)
                loss = latent_prev - latents_prev_rec
                
            noise_loss_list.append(loss.detach())
            latent_cur = latents_prev_rec + loss
            
        return noise_loss_list
    
    def invert(self, img_latent, guidance_scale, num_inner_steps=10, early_stop_epsilon=1e-5, ):
        #self.init_prompt(prompt)
        register_attention_control_controlnet(self.pipe, None)
        
        image_rec, ddim_latents = self.ddim_inversion(img_latent)
        
        noise_loss_list = self.offset_calculate(ddim_latents, num_inner_steps, early_stop_epsilon,guidance_scale)
        return image_rec, image_rec, ddim_latents, noise_loss_list
    
    def invert_without_attn_controller(self, image_gt, prompt, guidance_scale, num_inner_steps=10, early_stop_epsilon=1e-5):
        self.init_prompt(prompt)
        
        image_rec, ddim_latents = self.ddim_inversion(image_gt)
        
        noise_loss_list = self.offset_calculate(ddim_latents, num_inner_steps, early_stop_epsilon,guidance_scale)
        return image_gt, image_rec, ddim_latents, noise_loss_list
    
    def invert_with_guidance_scale_vary_guidance(self, image_gt, prompt, inverse_guidance_scale, forward_guidance_scale, num_inner_steps=10, early_stop_epsilon=1e-5):
        self.init_prompt(prompt)
        register_attention_control(self.pipe, None)
        
        image_rec, ddim_latents = self.ddim_with_guidance_scale_inversion(image_gt,inverse_guidance_scale)
        
        noise_loss_list = self.offset_calculate(ddim_latents, num_inner_steps, early_stop_epsilon,forward_guidance_scale)
        return image_gt, image_rec, ddim_latents, noise_loss_list

    def null_latent_calculate(self, latents, num_inner_steps, epsilon, guidance_scale):
        noise_loss_list = []
        latent_cur = torch.concat([latents[-1]]*(self.context.shape[0]//2))
        uncond_embeddings, cond_embeddings = self.context.chunk(2)
        for i in range(self.num_ddim_steps):            
            latent_prev = torch.concat([latents[len(latents) - i - 2]]*latent_cur.shape[0])
            t = self.pipe.scheduler.timesteps[i]

            if num_inner_steps!=0:
                uncond_embeddings = uncond_embeddings.clone().detach()
                uncond_embeddings.requires_grad = True
                optimizer = Adam([uncond_embeddings], lr=1e-2 * (1. - i / 100.))
                for j in range(num_inner_steps):
                    latents_input = torch.cat([latent_cur] * 2)
                    noise_pred = self.pipe.unet(latents_input, t, encoder_hidden_states=torch.cat([uncond_embeddings, cond_embeddings]))["sample"]
                    noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
                    
                    latents_prev_rec = self.prev_step(noise_pred, t, latent_cur)[0]
                    
                    loss = nnf.mse_loss(latents_prev_rec[[0]], latent_prev[[0]])
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    loss_item = loss.item()

                    if loss_item < epsilon + i * 2e-5:
                        break
                    
            with torch.no_grad():
                noise_pred = self.get_noise_pred_single(torch.concat([latent_cur]*2), t, self.context)
                noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                noise_pred_w_guidance = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                latents_prev_rec, _ = self.prev_step(noise_pred_w_guidance, t, latent_cur)
                
                latent_cur = self.get_noise_pred(latent_cur, t,guidance_scale, False, torch.cat([uncond_embeddings, cond_embeddings]))[0]
                loss = latent_cur - latents_prev_rec
                
            noise_loss_list.append(loss.detach())
            latent_cur = latents_prev_rec + loss
            
        return noise_loss_list
        
    
    def invert_null_latent(self, image_gt, prompt, guidance_scale, num_inner_steps=10, early_stop_epsilon=1e-5):
        self.init_prompt(prompt)
        register_attention_control(self.pipe, None)
        
        image_rec, ddim_latents = self.ddim_inversion(image_gt)
        
        latent_list = self.null_latent_calculate(ddim_latents, num_inner_steps, early_stop_epsilon,guidance_scale)
        return image_gt, image_rec, ddim_latents, latent_list
    
    def offset_calculate_not_full(self, latents, num_inner_steps, epsilon, guidance_scale,scale):
        noise_loss_list = []
        latent_cur = torch.concat([latents[-1]]*(self.context.shape[0]//2))
        for i in range(self.num_ddim_steps):            
            latent_prev = torch.concat([latents[len(latents) - i - 2]]*latent_cur.shape[0])
            t = self.pipe.scheduler.timesteps[i]
            with torch.no_grad():
                noise_pred = self.get_noise_pred_single(torch.concat([latent_cur]*2), t, self.context)
                noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                noise_pred_w_guidance = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                latents_prev_rec, _ = self.prev_step(noise_pred_w_guidance, t, latent_cur)
                loss = latent_prev - latents_prev_rec
                loss=loss*scale
                
            noise_loss_list.append(loss.detach())
            latent_cur = latents_prev_rec + loss
            
        return noise_loss_list
        
    def invert_not_full(self, image_gt, prompt, guidance_scale, num_inner_steps=10, early_stop_epsilon=1e-5,scale=1.):
        self.init_prompt(prompt)
        register_attention_control(self.pipe, None)
        
        image_rec, ddim_latents = self.ddim_inversion(image_gt)
        
        noise_loss_list = self.offset_calculate_not_full(ddim_latents, num_inner_steps, early_stop_epsilon,guidance_scale,scale)
        return image_gt, image_rec, ddim_latents, noise_loss_list
    
    def offset_calculate_skip_step(self, latents, num_inner_steps, epsilon, guidance_scale,skip_step):
        noise_loss_list = []
        latent_cur = torch.concat([latents[-1]]*(self.context.shape[0]//2))
        for i in range(self.num_ddim_steps):            
            latent_prev = torch.concat([latents[len(latents) - i - 2]]*latent_cur.shape[0])
            t = self.pipe.scheduler.timesteps[i]
            with torch.no_grad():
                noise_pred = self.get_noise_pred_single(torch.concat([latent_cur]*2), t, self.context)
                noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                noise_pred_w_guidance = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                latents_prev_rec, _ = self.prev_step(noise_pred_w_guidance, t, latent_cur)
                if (i%skip_step)==0:
                    loss = latent_prev - latents_prev_rec
                else:
                    loss=torch.zeros_like(latent_prev)
                
            noise_loss_list.append(loss.detach())
            latent_cur = latents_prev_rec + loss
            
        return noise_loss_list
    
    
    def invert_skip_step(self, image_gt, prompt, guidance_scale, skip_step,num_inner_steps=10, early_stop_epsilon=1e-5,scale=1.):
        self.init_prompt(prompt)
        register_attention_control(self.pipe, None)
        
        image_rec, ddim_latents = self.ddim_inversion(image_gt)
        
        noise_loss_list = self.offset_calculate_skip_step(ddim_latents, num_inner_steps, early_stop_epsilon,guidance_scale,skip_step)
        return image_gt, image_rec, ddim_latents, noise_loss_list
    
    
    def __init__(self, pipe,num_ddim_steps,context,label,device):
        self.pipe = pipe
        self.tokenizer = self.pipe.tokenizer
        self.num_ddim_steps=num_ddim_steps
        self.context=context
        self.label =label
        self.device= device
        
        
def prompt_aligned_injection(pipe,inputs_id,controlnet_cond):
    data_type=pipe.dtype
    text_encoder= pipe.text_encoder
    if 'after' in pipe.controlnet.task_cond:
        # insert embedding after transformer
        def get_concept_ids(text_encoder):
            model = text_encoder.module if hasattr(text_encoder, "module") else text_encoder
            return model.text_model.embeddings.embed_control.control_concept_ids

        concept_ids = get_concept_ids(text_encoder)
        input_ids_clone = inputs_id.clone()
        encoder_hidden_states = text_encoder(inputs_id)[0].to(dtype=data_type)
        if pipe.controlnet.dataset == 'ADNI':
            controlnet_cond_clone = controlnet_cond.clone()
            if len(concept_ids)==3:
                #if controlnet_cond_clone.shape[1] == 16:
                # only use brain_v, ven_v and slice 0-9 following benchmark
                controlnet_cond_clone=controlnet_cond_clone[:,4:,:]
            if controlnet_cond_clone.shape[1]==6:
                for i,token_id in enumerate(concept_ids):
                    placeholder_idx = torch.where(input_ids_clone == token_id)
                    encoder_hidden_states[placeholder_idx] = encoder_hidden_states[placeholder_idx]+ controlnet_cond_clone[:,i,:]
            elif controlnet_cond_clone.shape[1] == 12:
                for i,token_id in enumerate(concept_ids):
                    placeholder_idx = torch.where(input_ids_clone == token_id)
                    if i==len(concept_ids)-1:
                        encoder_hidden_states[placeholder_idx] = encoder_hidden_states[placeholder_idx]+ controlnet_cond_clone[:,i:].reshape(-1,1)
                    else:
                        encoder_hidden_states[placeholder_idx] = encoder_hidden_states[placeholder_idx]+ controlnet_cond_clone[:,i,:]
            elif controlnet_cond_clone.shape[1] > 12:
                for i,token_id in enumerate(concept_ids):
                    placeholder_idx = torch.where(input_ids_clone == token_id)
                    if i==0:
                        encoder_hidden_states[placeholder_idx] = encoder_hidden_states[placeholder_idx]+ controlnet_cond_clone[:,:2].reshape(-1,1)
                    elif i==len(concept_ids)-1:
                        encoder_hidden_states[placeholder_idx] = encoder_hidden_states[placeholder_idx]+ controlnet_cond_clone[:,-10:].reshape(-1,1)
                    else:
                        encoder_hidden_states[placeholder_idx] = encoder_hidden_states[placeholder_idx]+ controlnet_cond_clone[:,i+1,:]
        else:
            for i,token_id in enumerate(concept_ids):
                placeholder_idx = torch.where(input_ids_clone == token_id)
                encoder_hidden_states[placeholder_idx] = encoder_hidden_states[placeholder_idx]+ controlnet_cond[:,i,:]
    else:    
        encoder_hidden_states = text_encoder(encoder_hidden_states,attribute_cond = controlnet_cond)[0].to(dtype=data_type)
    
    return encoder_hidden_states