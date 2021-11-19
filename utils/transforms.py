from graphs.models.mobileNet import MobileNet
import matplotlib.pyplot as plt 
class SemanticSegmentation(object):
	""" Gets rid of the background and only returns a smaller image containing 
	the bird's pixels. 
	
	Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.
	Converts a PIL Image or numpy.ndarray (H x W x C) in the range
	[0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
	if the PIL Image belongs to one of the modes (L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK, 1)
	or if the numpy.ndarray has dtype = np.uint8
	In the other cases, tensors are returned without scaling.
	"""
	def __init__(self):
		super().__init__()
		self.net = MobileNet()
		self.bird_class = 3


	def __call__(self, pic):
		"""
			Args:
				pic (PIL Image or numpy.ndarray): Image to be converted to tensor.
			Returns:
				Tensor: Converted image.
		"""
		segmented = self.net(pic)
		segmented = segmented[:,self.bird_class]
		segmented = segmented[segmented >= 0.8]
		plt.ioff()
		plt.imshow(segmented)
		plt.close()
		return segmented

	def __repr__(self):
		return self.__class__.__name__ + '()'