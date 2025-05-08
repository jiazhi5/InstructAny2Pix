import torch
from PIL import Image

from instructany2pix import InstructAny2PixPipeline
ckpt = "instructany2pix_retrained"
llm_folder = 'llm-retrained'
pipe = InstructAny2PixPipeline(ckpt, llm_folder=llm_folder)
pipe.pipe.scheduler = pipe.pipe_inversion.scheduler

x = {"inst": "add <video> to <video>",
 "ans": "an image of an antique shop with a clock ticking",
 "mm_data": [
     {"type": "audio", "fname": "assets/demo/clock ticking.wav", }, 
     {"type": "image", "fname": "assets/demo/an antique shop.jpg", }],
}

inst = x['inst']
mm_data = x['mm_data']

Image.open("assets/demo/an antique shop.jpg")

torch.manual_seed(0)
print(inst)
print(mm_data)
res0,res,_ = pipe(inst,mm_data,alpha = 1.0,h=[0.4,0.6,0.4],norm=20.0,refinement=0.3,llm_only=False,num_inference_steps=50)