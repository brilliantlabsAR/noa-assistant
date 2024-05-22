from .generate_image import GenerateImage
from replicate import async_run as replicate

import requests
from io import BytesIO

import base64

NEGATIVE_PROMPT = 'ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, extra limbs, disfigured, deformed, body out of frame, bad anatomy, watermark, signature, cut off, low contrast, underexposed, overexposed, bad art, beginner, amateur, distorted face'
POSITIVE_PROMPT = 'digital art, hyperrealistic, fantasy, artstation, highly detailed, sharp focus, studio lighting'

class ReplicateGenerateImage(GenerateImage):
    def __init__(self, model: str='asiryan/juggernaut-xl-v7:6a52feace43ce1f6bbc2cdabfc68423cb2319d7444a1a1dae529c5e88b976382'):
        super().__init__()
        self._model = model
    
    async def generate_image(
        self,
        query: str,
        use_image: bool,
        image_bytes: bytes | None,
    ) -> str:
        if use_image:
            if not image_bytes:
                raise ValueError('Image bytes must be provided')
            input_base64_img = base64.b64encode(image_bytes).decode('utf-8')
            response = await replicate(
                self._model,
                input={
                    "width": 512,
                    "height": 512,
                    "prompt": f"{query}, {POSITIVE_PROMPT}",
                    "image": f'data:image/png;base64,{input_base64_img}',
                    "refine": "expert_ensemble_refiner",
                    "scheduler": "K_EULER",
                    "lora_scale": 0.5,
                    "num_outputs": 1,
                    "guidance_scale": 7.5,
                    "apply_watermark": False,
                    "high_noise_frac": 0.7,
                    "negative_prompt": NEGATIVE_PROMPT,
                    "prompt_strength": 0.8,
                    "num_inference_steps": 30
                }
            )
        else:
            raise NotImplementedError('Text generation is not imlemented yet')
        # else:
        #     response = replicate.run(
        #         self._model,
        #         input={
        #             "width": 512,
        #             "height": 512,
        #             "prompt": f"{query}, {POSITIVE_PROMPT}",
        #             "refine": "expert_ensemble_refiner",
        #             "scheduler": "K_EULER",
        #             "lora_scale": 0.6,
        #             "num_outputs": 1,
        #             "guidance_scale": 7.5,
        #             "apply_watermark": False,
        #             "high_noise_frac": 0.8,
        #             "negative_prompt": "",
        #             "prompt_strength": 1,
        #             "num_inference_steps": 25
        #         }
        #     )
            
        #  response is url of image
        # make it base64
        image_url = response[0]
        response = requests.get(image_url)
        if response.status_code != 200:
            return 'Failed to generate image'
        # convert to base64
        base64_img = base64.b64encode(response.content).decode('utf-8')
        return base64_img
        
GenerateImage.register(ReplicateGenerateImage)