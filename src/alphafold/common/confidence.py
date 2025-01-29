# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modified to add predicted interface TM-score and Interface-score
# Metrics for complex model evaluation
#
# Mu Gao and Davi Nakajima An
# Georgia Institute of Technology
#
"""Functions for processing confidence metrics."""

from typing import Dict, Optional, Tuple, Union, List
import numpy as np
import scipy.special
import networkx as nx
import cupy as cp
import cugraph
import cudf


def compute_plddt(logits: np.ndarray) -> np.ndarray:
  """Computes per-residue pLDDT from logits.

  Args:
    logits: [num_res, num_bins] output from the PredictedLDDTHead.

  Returns:
    plddt: [num_res] per-residue pLDDT.
  """
  # Move computation to GPU
  logits_gpu = cp.asarray(logits)
  num_bins = logits_gpu.shape[-1]
  bin_width = 1.0 / num_bins
  bin_centers = cp.arange(start=0.5 * bin_width, stop=1.0, step=bin_width)
  probs = cp.special.softmax(logits_gpu, axis=-1)
  predicted_lddt_ca = cp.sum(probs * bin_centers[None, :], axis=-1)
  return cp.asnumpy(predicted_lddt_ca * 100)


def _calculate_bin_centers(breaks: np.ndarray):
  """Gets the bin centers from the bin edges.

  Args:
    breaks: [num_bins - 1] the error bin edges.

  Returns:
    bin_centers: [num_bins] the error bin centers.
  """
  breaks_gpu = cp.asarray(breaks)
  step = (breaks_gpu[1] - breaks_gpu[0])

  # Add half-step to get the center
  bin_centers = breaks_gpu + step / 2
  # Add a catch-all bin at the end.
  bin_centers = cp.concatenate([bin_centers, [bin_centers[-1] + step]],
                               axis=0)
  return bin_centers


def _calculate_expected_aligned_error(
    alignment_confidence_breaks: np.ndarray,
    aligned_distance_error_probs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
  """Calculates expected aligned distance errors for every pair of residues.

  Args:
    alignment_confidence_breaks: [num_bins - 1] the error bin edges.
    aligned_distance_error_probs: [num_res, num_res, num_bins] the predicted
      probs for each error bin, for each pair of residues.

  Returns:
    predicted_aligned_error: [num_res, num_res] the expected aligned distance
      error for each pair of residues.
    max_predicted_aligned_error: The maximum predicted error possible.
  """
  breaks_gpu = cp.asarray(alignment_confidence_breaks)
  probs_gpu = cp.asarray(aligned_distance_error_probs)
  
  bin_centers = _calculate_bin_centers(breaks_gpu)
  expected_error = cp.sum(probs_gpu * bin_centers, axis=-1)
  max_error = bin_centers[-1]
  
  return cp.asnumpy(expected_error), cp.asnumpy(max_error)


def compute_predicted_aligned_error(
    logits: np.ndarray,
    breaks: np.ndarray) -> Dict[str, np.ndarray]:
  """Computes aligned confidence metrics from logits.

  Args:
    logits: [num_res, num_res, num_bins] the logits output from
      PredictedAlignedErrorHead.
    breaks: [num_bins - 1] the error bin edges.

  Returns:
    aligned_confidence_probs: [num_res, num_res, num_bins] the predicted
      aligned error probabilities over bins for each residue pair.
    predicted_aligned_error: [num_res, num_res] the expected aligned distance
      error for each pair of residues.
    max_predicted_aligned_error: The maximum predicted error possible.
  """
  logits_gpu = cp.asarray(logits)
  breaks_gpu = cp.asarray(breaks)
  
  aligned_confidence_probs = cp.special.softmax(logits_gpu, axis=-1)
  predicted_aligned_error, max_predicted_aligned_error = (
      _calculate_expected_aligned_error(
          alignment_confidence_breaks=breaks_gpu,
          aligned_distance_error_probs=aligned_confidence_probs))
          
  return {
      'aligned_confidence_probs': cp.asnumpy(aligned_confidence_probs),
      'predicted_aligned_error': predicted_aligned_error,
      'max_predicted_aligned_error': max_predicted_aligned_error,
  }


def predicted_tm_score(
    logits: np.ndarray,
    breaks: np.ndarray,
    residue_weights: Optional[np.ndarray] = None,
    asym_id: Optional[np.ndarray] = None,
    interface: bool = False) -> np.ndarray:
  """Computes predicted TM alignment or predicted interface TM alignment score."""
  # Move computation to GPU
  logits_gpu = cp.asarray(logits)
  
  if residue_weights is None:
    residue_weights = np.ones(logits.shape[0])
  residue_weights_gpu = cp.asarray(residue_weights)

  bin_centers = _calculate_bin_centers(breaks)

  num_res = int(cp.sum(residue_weights_gpu))
  clipped_num_res = max(num_res, 19)

  d0 = 1.24 * (clipped_num_res - 15) ** (1./3) - 1.8

  probs = cp.special.softmax(logits_gpu, axis=-1)

  tm_per_bin = 1. / (1 + cp.square(bin_centers) / cp.square(d0))
  predicted_tm_term = cp.sum(probs * tm_per_bin, axis=-1)

  pair_mask = cp.ones(shape=(num_res, num_res), dtype=bool)
  if interface:
    asym_id_gpu = cp.asarray(asym_id)
    pair_mask *= asym_id_gpu[:, None] != asym_id_gpu[None, :]

  predicted_tm_term *= pair_mask

  pair_residue_weights = pair_mask * (
      residue_weights_gpu[None, :] * residue_weights_gpu[:, None])
  normed_residue_mask = pair_residue_weights / (1e-8 + cp.sum(
      pair_residue_weights, axis=-1, keepdims=True))
  per_alignment = cp.sum(predicted_tm_term * normed_residue_mask, axis=-1)
  return cp.asnumpy(per_alignment[(per_alignment * residue_weights_gpu).argmax()])

def predicted_tm_score_v1(
    logits: np.ndarray,
    breaks: np.ndarray,
    residue_weights: Optional[np.ndarray] = None,
    is_probs: Optional[bool] = False,
    chain_mask: Optional[np.ndarray] = None,
    inter_chain_mask: Optional[np.ndarray] = None) -> np.ndarray:
  """Computes predicted TM alignment score."""
  # Move computation to GPU
  if not is_probs:
    logits_gpu = cp.asarray(logits)
  else:
    logits_gpu = cp.asarray(logits)

  if residue_weights is None:
    residue_weights = np.ones(logits.shape[0])
  residue_weights_gpu = cp.asarray(residue_weights)

  bin_centers = _calculate_bin_centers(breaks)

  num_res = cp.sum(residue_weights_gpu)
  clipped_num_res = max(num_res, 19)

  d0 = 1.24 * (clipped_num_res - 15) ** (1./3) - 1.8

  if d0 < 0.5: 
    d0 = 0.02 * num_res

  if not is_probs:
    probs = cp.special.softmax(logits_gpu, axis=-1)
  else:
    probs = logits_gpu

  tm_per_bin = 1. / (1 + cp.square(bin_centers) / cp.square(d0))
  predicted_tm_term = cp.sum(probs * tm_per_bin, axis=-1)

  if inter_chain_mask is not None:
    inter_chain_mask_gpu = cp.asarray(inter_chain_mask)
    predicted_tm_term = predicted_tm_term * inter_chain_mask_gpu

  normed_residue_mask = residue_weights_gpu / (1e-8 + residue_weights_gpu.sum())

  if chain_mask is not None:
    chain_mask_gpu = cp.asarray(chain_mask)
    per_alignment = cp.sum(predicted_tm_term * normed_residue_mask * chain_mask_gpu, axis=-1)
    return cp.asnumpy(per_alignment[(per_alignment * residue_weights_gpu).argmax()])
  else:
    per_alignment = cp.sum(predicted_tm_term * normed_residue_mask, axis=-1)
    return cp.asnumpy(per_alignment[(per_alignment * residue_weights_gpu).argmax()])


def predicted_interface_tm_score(
    logits: np.ndarray,
    breaks: np.ndarray,
    pos: np.ndarray,
    atom_mask: np.ndarray,
    asym_id: np.ndarray,
    residue_weights: Optional[np.ndarray] = None,
    distance_threshold: Optional[int] = 4.5,
    is_probs: Optional[bool] = False,
    inter_chain_mask: Optional[np.ndarray] = None) -> Dict[str, Union[np.ndarray, int]]:
  """Computes predicted interfacial TM-score using GPU acceleration."""
  
  # Only calculate piTMS if a multi-chain target
  if np.all(asym_id == asym_id[0]):
    return {
      'score': np.asarray(0),
      'num_residues': np.asarray(0, dtype=np.int32),
      'num_contacts': np.asarray(0, dtype=np.int32),
    }

  residue_mask, contact_mask = get_residue_and_contact_masks(
      asym_id, pos, atom_mask, distance_threshold)

  # Return 0.0 if no inter-chain contacts found
  if residue_mask.sum() == 0:
    return {
      'score': np.asarray(0),
      'num_residues': np.asarray(0, dtype=np.int32),
      'num_contacts': np.asarray(0, dtype=np.int32),
    }

  # Select only interfacial residues
  if residue_weights is None:
    residue_weights = np.ones(logits.shape[0])
  residue_weights = residue_weights * residue_mask

  # Return the piTM score and other interface data
  return {
    'score': predicted_tm_score_v1(logits, breaks, residue_weights,
        is_probs=is_probs, inter_chain_mask=inter_chain_mask),
    'num_residues': np.asarray(residue_mask.sum()),
    'num_contacts': np.asarray(contact_mask.sum()),
  }

def make_tm_score_masks(asym_id):
  # create an inter-chain mask, where only residues relevant to tm scoring are ones.
  # note that residue pairs of the same chain subject to tm-scoring are zeros.
  # this force to select the best local frame from residues of other chains.
  full_length = len(asym_id)

  def _make_tm_score_masks(start_a, end_a):
      inter_chain_mask = np.zeros((full_length, full_length))
      chain_mask = np.zeros(full_length, dtype=int)
      chain_mask[start_a:end_a] = 1
      inter_chain_mask[start_a:end_a,:] = 1
      inter_chain_mask[:,start_a:end_a] = 1
      inter_chain_mask[start_a:end_a,start_a:end_a] = 0
      return inter_chain_mask == 1, chain_mask == 1

  tm_masks = []
  for i in range(int(asym_id.max()+1)):
    rangei = np.where(asym_id == i)[0]

    prev_id = np.roll(rangei, 1)
    diff = np.abs(rangei - prev_id)
    non_contiguous = diff > 1
    non_contiguous[0] = False

    if np.any(non_contiguous):
      # in case the range is not contiguous (superchain is across chains not
      # next to each other in the sequence) xor the masks of each contiguous sequence
      ranges = []
      start = 0
      for end in np.where(non_contiguous)[0]:
        if end == 0:
          continue
        ranges.append((rangei[start], rangei[end]))
        start = end
      inter_chain_mask = None
      chain_mask = None
      for starti, termini in ranges:
        _inter_mask, _mask  = _make_tm_score_masks(starti, termini)
        if inter_chain_mask is None:
          inter_chain_mask = _inter_mask
          chain_mask = _mask
        else:
          inter_chain_mask = np.bitwise_xor(inter_chain_mask, _inter_mask)
          chain_mask = np.bitwise_xor(chain_mask, _mask)
        tm_masks.append((inter_chain_mask, chain_mask))

    else:
      starti, termini = rangei.min(), rangei.max() + 1
      tm_masks.append(_make_tm_score_masks(starti, termini))

  return tm_masks

def join_superchains_asym_id(
    asym_id: np.ndarray,
    asym_id_list: Optional[List] = None,) -> np.ndarray:
  """ Returns a new asym_id array where each chain in a superchain have the
    same index. This enables each superchain to be considered as a single
    chain in subsequent calculations.
  """
  if asym_id_list is None:
    return asym_id, None

  full_length = len(asym_id)
  super_id2chain_ids = {}
  # make new asym_id list with one index per superchain
  new_asym_id = np.empty(asym_id.shape)
  for i, id_list in enumerate(asym_id_list):
    chain_mask = np.zeros(full_length, dtype=bool)
    super_id2chain_ids[i] = []
    for idx in id_list:
      i_range = np.where(asym_id == idx)[0]
      start_a, end_a = i_range.min(), i_range.max()+1
      chain_mask[start_a:end_a] = 1
      super_id2chain_ids[i].append(idx)
    new_asym_id[chain_mask] = i
  if len(super_id2chain_ids) == 0:
    return new_asym_id, None

  return new_asym_id, super_id2chain_ids

def interface_score(
    logits: np.ndarray,
    breaks: np.ndarray,
    pos: np.ndarray,
    atom_mask: np.ndarray,
    asym_id: np.ndarray,
    residue_weights: Optional[np.ndarray] = None,
    distance_threshold: Optional[float] = 4.5,
    is_probs: Optional[bool] = False,) -> Dict[str, Union[np.ndarray, int]]:
  """ Returns the interface-score, number of residues in the interface, and
    number of contacts of a complex model. This is a further tweak from piTM by
    looking the best tm-score estimate from other chains

  Args:
    logits: [num_res, num_res, num_bins] the logits output from
      PredictedAlignedErrorHead.
    breaks: [num_bins] the error bins.
    pos: [num_res, atom_type_num, 3] the predicted atom positions
    atom_mask: mask for atoms (each residue type has different number of atoms)
    asym_id: [num_res] a unique integer per chain indicating the chain number.
      The ordering of the input chains is arbitrary (As defined by AF-Multimer
      paper)
    residue_weights: [num_res] the per residue weights to use for the
      expectation.
    distance_threshold: maximum distance between two residue's heavy atoms
      from different chains to be considered in the interface
    is_probs: boolean indicating whether the logits argument are probabilities
  Returns:
    None if target is a single chain, otherwise
    pitm_dict: dict of np.ndarrays containing
      "score" - piTM score for the sequence
      "num_residues" - number of residues in the interface
      "num_contacts" - number of contacts along the interface of protein complex
  """
  # only calculate the score if multi-chain target
  if np.all(asym_id == asym_id[0]):
    return {
      'score': np.asarray(0),
      'num_residues': np.asarray(0, dtype=np.int32),
      'num_contacts': np.asarray(0, dtype=np.int32),
    }

  residue_mask, contact_mask = get_residue_and_contact_masks(
      asym_id, pos, atom_mask, distance_threshold)

  # return 0.0 if no inter-chain contacts found
  if residue_mask.sum() == 0:
    return {
      'score': np.asarray(0),
      'num_residues': np.asarray(0, dtype=np.int32),
      'num_contacts': np.asarray(0, dtype=np.int32),
    }

  score = calculate_interface_score(
    logits, breaks, asym_id, residue_mask, residue_weights, is_probs
  )

  return {
      'score': score,
      'num_residues': np.asarray(residue_mask.sum(), dtype=np.int32),
      'num_contacts': np.asarray(contact_mask.sum(), dtype=np.int32),
  }

def calculate_interface_score(
    logits: np.ndarray,
    breaks: np.ndarray,
    asym_id: np.ndarray,
    residue_mask: np.ndarray,
    residue_weights: Optional[np.ndarray] = None,
    is_probs: Optional[bool] = False,) -> Dict[str, Union[np.ndarray, int]]:
  """Computes the interface-score of a complex model.
    Score defined in the AF2Complex manuscript (2021)

  Args:
    logits: [num_res, num_res, num_bins] the logits output from
      PredictedAlignedErrorHead.
    breaks: [num_bins] the error bins.
    asym_id: [num_res] a unique integer per chain indicating the chain number.
      The ordering of the input chains is arbitrary (As defined by AF-Multimer
      paper)
    residue_mask: [num_res] array indicating if a residue is in the interface
      of the complex.
    residue_weights: [num_res] the per residue weights to use for the
      expectation.
    distance_threshold: maximum distance between two residue's heavy atoms
      from different chains to be considered in the interface
    is_probs: boolean indicating whether the logits argument are probabilities

  Returns:
    score - piTM score for the sequence
  """

  tm_masks= make_tm_score_masks(asym_id)

  # select only interfacial residues
  if residue_weights is None:
    residue_weights = np.ones(logits.shape[0])
  residue_weights = residue_weights * residue_mask

  # calculate the tm-score for interface residues of each chain (or chain set) pair, and sum them.
  score = 0
  for inter_chain_mask, chain_mask in tm_masks:
    sc = predicted_tm_score_v1( logits, breaks, residue_weights,is_probs=is_probs,
           chain_mask=chain_mask, inter_chain_mask=inter_chain_mask )
    score += sc

  # return the  interaction score and other interface data
  return score
  
def get_residue_and_contact_masks(
    asym_id: np.ndarray,
    pos: np.ndarray,
    atom_mask: np.ndarray,
    distance_threshold: Optional[float] = 4.5,):
  """Computes the residue and contact masks using GPU acceleration."""
  
  # Move data to GPU
  pos_gpu = cp.asarray(pos)
  atom_mask_gpu = cp.asarray(atom_mask)
  
  num_res = pos.shape[0]
  residue_mask = cp.zeros(num_res, dtype=bool)
  contact_mask = cp.zeros((num_res, num_res), dtype=bool)

  def get_min_pairwise_dist(a, b, mask_a, mask_b):
    a = a[mask_a > 0.5]
    b = b[mask_b > 0.5]
    # Efficient GPU pairwise distance computation
    diff = a[:, None, :] - b[None, :, :]
    dist_sq = cp.sum(diff * diff, axis=2)
    return cp.sqrt(cp.min(dist_sq))

  for i in range(num_res):
      for j in range(i+1, num_res):  # use symmetry
          if asym_id[i] != asym_id[j]:
              if atom_mask_gpu[i].sum() == 0 or atom_mask_gpu[j].sum() == 0:
                  continue
              dist = get_min_pairwise_dist(
                  pos_gpu[i], pos_gpu[j],
                  atom_mask_gpu[i], atom_mask_gpu[j])
              is_contact = dist < distance_threshold
              residue_mask[i] = residue_mask[i] or is_contact
              residue_mask[j] = residue_mask[j] or is_contact
              contact_mask[i, j] = contact_mask[i, j] or is_contact

  return cp.asnumpy(residue_mask), cp.asnumpy(contact_mask)

################################################################################
def cluster_analysis(
    asym_id: np.ndarray,
    pos: np.ndarray,
    atom_mask: np.ndarray,
    distance_threshold: Optional[float] = 4.5,
    edge_contacts_thres: Optional[int] = 10,
    superid2chainids: Optional[Dict[int, List[int]]] = None,
) -> Dict[str, Union[int, List[int], List[List[int]]]]:
  """Computes information about clusters using GPU acceleration."""
  asym_id = asym_id.astype(int)
  res_mask, contact_mask = get_residue_and_contact_masks(
      asym_id, pos, atom_mask, distance_threshold)

  num_chains = int(asym_id.max() + 1)
  num_res = len(asym_id)
  chain_adj_count = cp.zeros((num_chains, num_chains))

  resid2asymid = {k: v for k, v in enumerate(asym_id)}

  # Compute chain adjacency matrix on GPU
  for i in range(num_res):
      for j in range(i+1, num_res):
          asym_id_a = resid2asymid[i]
          asym_id_b = resid2asymid[j]
          if asym_id_a != asym_id_b:
              chain_adj_count[asym_id_a, asym_id_b] += contact_mask[i, j]

  chain_adj_mat = cp.asnumpy(chain_adj_count > edge_contacts_thres)
  chain_adj_mat = np.logical_or(chain_adj_mat, chain_adj_mat.T)

  # Create CuGraph from adjacency matrix
  G = cugraph.Graph()
  edges = np.where(chain_adj_mat)
  G.from_cudf_edgelist(
      cudf.DataFrame({
          'src': edges[0],
          'dst': edges[1]
      })
  )

  # Find connected components
  components = cugraph.connected_components(G)
  labels = components.values_host

  # Process results
  unique_labels = np.unique(labels)
  num_clusters = len(unique_labels)
  clusters = []
  cluster_size = []

  for label in unique_labels:
      cluster = np.where(labels == label)[0].tolist()
      clusters.append(cluster)
      if superid2chainids:
          size = sum(len(superid2chainids[i]) for i in cluster)
      else:
          size = len(cluster)
      cluster_size.append(size)

  return {
      'num_clusters': num_clusters,
      'cluster_size': cluster_size,
      'clusters': clusters,
  }
