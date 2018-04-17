import os

import torch, cv2

from davisinteractive.session import DavisInteractiveSession
from davisinteractive import utils as interactive_utils
from davisinteractive.dataset import Davis

from osvos_scribble import OsvosScribble
from mypath import Path

# General parameters
gpu_id = 0

# Interactive parameters
max_nb_interactions = 5
max_time = None  # Maximum time for each interaction
subset = 'val'
host = 'localhost'  # 'localhost' for subsets train and val.

# OSVOS parameters
time_budget_per_object = 60
parent_model = 'osvos_parent.pth'
prev_mask = True  # Use previous mask as no-care area when fine-tuning

save_model_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'models')
report_save_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'results')
save_result_dir = report_save_dir  # 'None' to not save the results

model = OsvosScribble(parent_model, save_model_dir, gpu_id, time_budget_per_object, save_result_dir=save_result_dir)
davis = Davis(davis_root=Path.db_root_dir())

seen_seq = {}
with DavisInteractiveSession(host='localhost', davis_root=Path.db_root_dir(), subset=subset,
                             report_save_dir=report_save_dir, max_nb_interactions=max_nb_interactions,
                             max_time=max_time) as sess:
    while sess.next():
        # Get the current iteration scribbles
        sequence, scribbles, first_scribble = sess.get_scribbles()
        if first_scribble:
            n_interaction = 1
            n_objects = davis.dataset['sequences'][sequence]['num_objects']
            first_frame = interactive_utils.scribbles.annotated_frames(scribbles)[0]
            seen_seq[sequence] = 1 if sequence not in seen_seq.keys() else seen_seq[sequence]+1
        else:
            n_interaction += 1
        pred_masks = []
        print('\nRunning sequence {} in interaction {} and scribble iteration {}'
              .format(sequence, n_interaction, seen_seq[sequence]))
        for obj_id in range(1, n_objects+1):
            model.train(first_frame, n_interaction, obj_id, scribbles, seen_seq[sequence], use_previous_mask=prev_mask)
            pred_masks.append(model.test(sequence, n_interaction, obj_id, seen_seq[sequence]))

        final_masks = interactive_utils.mask.combine_masks(pred_masks)

        # Submit your prediction
        sess.submit_masks(final_masks)
