import torch
import torch.nn as nn
import torch.nn.functional as F

class TripletMarginLossCosine(nn.Module):
	"""
	Triplet Margin Loss with Cosine Similarity.

	Args:
		margin: A float value representing the margin for the triplet loss.
	"""
	def __init__(self, margin=0.2):
		super(TripletMarginLossCosine, self).__init__()
		self.margin = margin

	def forward(self, anchor, pos, neg):
		"""
		Calculates the Triplet Margin Loss with cosine similarity.

		Args:
			anchor: Tensor representing the anchor embedding.
			pos: Tensor representing the positive embedding.
			neg: Tensor representing the negative embedding.

		Returns:
			A tensor containing the triplet margin loss.
		"""
		pos_sim = F.cosine_similarity(anchor, pos)
		neg_sim = F.cosine_similarity(anchor, neg)
		losses = self.margin - pos_sim + neg_sim
		return F.relu(losses).mean()
