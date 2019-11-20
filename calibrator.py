import tensorrt as trt
import os
import captcha_generator
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit

gen = captcha_generator.CaptchaGen(80, 32, ['Ubuntu-M.ttf'])

def get_captchar_batch(batch_size):
    return np.asarray([gen.image(captcha_generator.DigitCaptcha.get_rand(3, 4)) for _ in range(batch_size)], np.float32)

class CaptcharEntropyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, cache_file, batch_size=64):
        trt.IInt8EntropyCalibrator2.__init__(self)
        self.cache_file = cache_file
        self.batch_size = batch_size
        self.device_input = cuda.mem_alloc(80 * 32 * 4 * self.batch_size)
        self.count = 0

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        if self.count >= 10:
            return None
        self.count += 1
        batch = get_captchar_batch(self.batch_size)
        cuda.memcpy_htod(self.device_input, batch)
        return [self.device_input]

    def read_calibration_cache(self):
        print('read_calibration_cache')
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)
