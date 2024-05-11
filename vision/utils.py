

import cv2
import numpy as np
from abc import ABC, abstractmethod
from typing import List
import os

def detect_media_type(image_bytes: bytes) -> str:
    if image_bytes is not None:
        if image_bytes[0:4] == b"\x89PNG":
            return "image/png"
        elif b"JFIF" in image_bytes[0:64]:  # probably should do a stricter check here
            return "image/jpeg"
        elif image_bytes[0:4] == b"RIFF" and image_bytes[8:12] == b"WEBP":
            return "image/webp"

    # Unknown: just assume JPEG
    return "image/jpeg"



class BaseFilter(ABC):
    @abstractmethod
    def apply(self, image):
        pass

    @abstractmethod
    def adjust(self):
        pass

class SmoothFilter(BaseFilter):
    def __init__(self, kernel_size=5):
        self.kernel_size = kernel_size

    def apply(self, image):
        print("Smoothing kernel size: ", self.kernel_size)
        # Apply Gaussian blur
        return cv2.GaussianBlur(image, (self.kernel_size, self.kernel_size), 0)
    
    def adjust(self, kernel_size=None):
        self.kernel_size = kernel_size

class SaturationFilter(BaseFilter):
    def __init__(self, saturation=1.0):
        self.saturation = saturation

    def apply(self, image):
        print("Saturation value: ", self.saturation)
        # Convert the image to the HSV color space
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        # Apply saturation to the S channel
        s = cv2.addWeighted(s, self.saturation, np.zeros_like(s), 0, 0)
        
        # Merge the channels back together
        hsv = cv2.merge([h, s, v])
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    def adjust(self, saturation=None, scale=100):
        self.saturation = (saturation / scale)*10 if saturation > 0 else 0.1

class TemperatureFilter(BaseFilter):
    def __init__(self, temperature=0.0):
        self.temperature = temperature

    def apply(self, image):
        print("Temperature value: ", self.temperature)
        # Convert the image to the LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply temperature to the A channel
        a = cv2.addWeighted(a, self.temperature, np.zeros_like(a), 0, 0)
        
        # Merge the channels back together
        lab = cv2.merge([l, a, b])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    def adjust(self, temperature=None, scale=100):
        self.temperature = (temperature / scale)*10 if temperature > 0 else 0.1

class GammaCorrectionFilter(BaseFilter):
    def __init__(self, gamma=1.0):
        self.gamma = gamma

    def apply(self, image):
        print("Gamma value: ", self.gamma)
            # Apply gamma correction using cv2.addWeighted()
        invGamma = 1.0 / self.gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(image, table)
    def adjust(self, gamma=None, scale=100):
        # adjust to 1, 10 scale depending on the slider
        self.gamma = (gamma / scale)*10 if gamma > 0 else 0.1
        self.gamma = gamma

class BoostShadowFilter(BaseFilter):
    def __init__(self, amount=1):
        self.amount = amount

    def apply(self, image):
        gamma_corrected = np.power(image / 255.0, self.amount) * 255
        return np.uint8(gamma_corrected)
    
    def adjust(self, amount=None, scale=100):
        self.amount = (amount / scale)*10 if amount > 0 else 0.1

class SharpeningFilter(BaseFilter):
    def __init__(self, sigma=1.0, strength=1.0):
        self.sigma = sigma
        self.strength = strength

    def apply(self, image):
        # Apply Gaussian blur
        print("Sharpening sigma: ", self.sigma, "Strength: ", self.strength)
        blurred = cv2.GaussianBlur(image, (0, 0), self.sigma)
        
        # Calculate the unsharp mask
        unsharp_mask = cv2.addWeighted(image, 1.0 + self.strength, blurred, -self.strength, 0)
        
        return unsharp_mask
    
    def adjust(self, amount=None, scale=100):
        self.amount = (amount / scale)*10 if amount > 0 else 0.1    

class NoiseReductionFilter(BaseFilter):
    def __init__(self, method='gaussian', kernel_size=5):
        self.method = method
        self.kernel_size = kernel_size

    def apply(self, image):
        print("Noise reduction method: ", self.method, "Kernel size: ", self.kernel_size)
        if self.method == 'gaussian':
            # Apply Gaussian blur
            blurred = cv2.GaussianBlur(image, (self.kernel_size, self.kernel_size), 0)
        elif self.method == 'median':
            # Apply Median blur
            blurred = cv2.medianBlur(image, self.kernel_size)
        else:
            raise ValueError("Unsupported noise reduction method. Use 'gaussian' or 'median'.")

        return blurred
    def adjust(self, method=None, kernel_size=None):
        self.method = method
        self.kernel_size = kernel_size
        
class ContrastFilter(BaseFilter):
    def __init__(self, contrast=1.0):
        self.contrast = contrast

    def apply(self, image):
        print("Contrast value: ", self.contrast)
        # Apply contrast by converting the image to YUV color space
        yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        y, u, v = cv2.split(yuv)
        
        # Apply contrast to the Y channel
        y = cv2.addWeighted(y, self.contrast, np.zeros_like(y), 0, 0)
        
        # Merge the channels back together
        yuv = cv2.merge([y, u, v])
        return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
    
    def adjust(self, contrast=None, scale=100):
        self.contrast = contrast

class BoostResolutionFilter(BaseFilter):
    def __init__(self, factor=2):
        self.factor = factor

    def apply(self, image):
        print("Resolution boost factor: ", self.factor)
        # Upscale the image using bicubic interpolation
        return cv2.resize(image, None, fx=self.factor, fy=self.factor, interpolation=cv2.INTER_CUBIC)
    
    def adjust(self, factor=None, scale=100):
        self.factor = factor

class ApplyBlurFilter(BaseFilter):
    def __init__(self, kernel_size=5):
        self.kernel_size = kernel_size

    def apply(self, image):
        print("Blur kernel size: ", self.kernel_size)
        # Apply Gaussian blur
        return cv2.GaussianBlur(image, (self.kernel_size, self.kernel_size), 0)
    
    def adjust(self, kernel_size=None):
        self.kernel_size = kernel_size

class ReduceResolutionFilter(BaseFilter):
    def __init__(self, factor=2):
        self.factor = factor

    def apply(self, image):
        print("Resolution reduction factor: ", self.factor)
        # Downscale the image using bicubic interpolation
        return cv2.resize(image, None, fx=1.0/self.factor, fy=1.0/self.factor, interpolation=cv2.INTER_CUBIC)
    
    def adjust(self, factor=None, scale=100):
        self.factor = factor

class WhiteBalanceFilter(BaseFilter):
    def __init__(self):
        pass

    def apply(self, image):
        print("White balance")
        # Auto white balance by equalizing the histogram of the LAB L channel
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(4, 4))
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    def adjust(self, b_ratio=None, g_ratio=None, r_ratio=None):
        self.b_ratio = b_ratio
        self.g_ratio = g_ratio
        self.r_ratio = r_ratio

class BrightnessFilter(BaseFilter):
    def __init__(self, brightness=0):
        self.brightness = brightness

    def apply(self, image):
        print("Brightness value: ", self.brightness)
        # Increase the brightness by adding the specified value to each pixel
        return cv2.add(image, np.array([self.brightness]))
    
    def adjust(self, brightness=None, scale=100):
        self.brightness = brightness

class ImageProcessor:
    def __init__(self,path: str|List[str]|bytes, filters: List[BaseFilter]):
        self.path = path
        self.filters = filters
        if isinstance(path, list):
            self.image = [cv2.imread(p) for p in path]
        elif isinstance(path, bytes):
            nparr = np.frombuffer(path, np.uint8)
            self.image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        else:
            self.image = cv2.imread(self.path)
        self.filtered_image = None

    def apply_filters(self, image=None):
        if image is None:
            image = self.image

        if isinstance(image, list):
            self.filtered_image = [None] * len(image)

            for i, img in enumerate(image):
                for f in self.filters:
                    self.filtered_image[i] = f.apply(img)
                    
            print("Filtered image: ", len(self.filtered_image))
        else:
            for f in self.filters:
                image = f.apply(image)
            self.filtered_image = image
        return image
    
    def __call__(self, preview: bool = True):
        self.apply_filters()
        if preview:
            self.show_preview(slider=True)

    def show_preview(self, slider=False):
        if self.filtered_image is not None:
            if isinstance(self.filtered_image, list):
                for i, img in enumerate(self.filtered_image):
                    filters_applied = ", ".join([f.__class__.__name__ for f in self.filters])
                    cv2.imshow('Original', self.image[i])
                    cv2.imshow('Filtered [{}]'.format(filters_applied), img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            else:
                filters_applied = ", ".join([f.__class__.__name__ for f in self.filters])
                cv2.imshow('Original', self.image)
                cv2.imshow('Filtered [{}]'.format(filters_applied), self.filtered_image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
    def save(self, dir_path: str="output"):
        dir_path = os.path.join(os.getcwd(), dir_path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        if self.filtered_image is not None:
            if isinstance(self.filtered_image, list):
                for i, img in enumerate(self.filtered_image):
                    basename = os.path.basename(self.path[i])
                    path = os.path.join(dir_path, basename.replace('.webp', '.jpg'))
                    cv2.imwrite(path, img)
            else:
                if isinstance(self.path, bytes):
                    basename = "test_image.jpg"
                else:
                    basename = os.path.basename(self.path)
                path = os.path.join(dir_path, basename.replace('.webp', '.jpg'))
                print(path)
                cv2.imwrite(path, self.filtered_image)
    def get_bytes(self):
        if self.filtered_image is not None:
            if isinstance(self.filtered_image, list):
                return [cv2.imencode('.jpg', img)[1].tobytes() for img in self.filtered_image]
            else:
                return cv2.imencode('.jpg', self.filtered_image)[1].tobytes()
        return None

def process_image(bytes: bytes)->bytes:
    filters:List[BaseFilter] = [
        # BoostShadowFilter(amount=0.8),
        GammaCorrectionFilter(gamma=1.2),
        # BoostResolutionFilter(factor=),

        SharpeningFilter(sigma=0.5, strength=5.0),
        WhiteBalanceFilter(),

        BoostShadowFilter(amount=1.2),
        ContrastFilter(contrast=1.2),
        SaturationFilter(saturation=1.2),
        TemperatureFilter(temperature=1.02),
        ApplyBlurFilter(kernel_size=3),

        # ReduceResolutionFilter(factor=2),

        # NoiseReductionFilter(method='gaussian', kernel_size=3),
    ]
    image_processor = ImageProcessor(path=bytes, filters=filters)
    image_processor.apply_filters()
    # image_processor.save("output")
    return image_processor.get_bytes()