"""Tests for zero-allocation preallocated memory buffers during match turns."""

import torch
import numpy as np
from inference.agent_worker import PreallocatedTensorBuffer


def test_preallocated_buffer_retains_data_pointer():
    buffers = PreallocatedTensorBuffer()

    raw_ptr_before = buffers.raw_features.data_ptr()
    mask_ptr_before = buffers.legality_mask.data_ptr()
    cmicro_ptr_before = buffers.c_micro.data_ptr()
    zmatch_ptr_before = buffers.z_match.data_ptr()

    # Simulate 50 turn updates
    for i in range(50):
        dummy_raw = np.full((64, 256), float(i), dtype=np.float32)
        dummy_mask = np.zeros(64, dtype=np.float32)
        dummy_cmicro = np.zeros(32, dtype=np.float32)
        dummy_zmatch = np.zeros(64, dtype=np.float32)

        buffers.update_from_numpy(
            raw_feats_np=dummy_raw,
            card_ids_np=None,
            mask_np=dummy_mask,
            c_micro_np=dummy_cmicro,
            z_match_np=dummy_zmatch,
        )

        # Assert data pointer remains invariant (no re-allocation!)
        assert buffers.raw_features.data_ptr() == raw_ptr_before
        assert buffers.legality_mask.data_ptr() == mask_ptr_before
        assert buffers.c_micro.data_ptr() == cmicro_ptr_before
        assert buffers.z_match.data_ptr() == zmatch_ptr_before

        # Verify in-place values were correctly updated
        assert buffers.raw_features[0, 0, 0].item() == float(i)
